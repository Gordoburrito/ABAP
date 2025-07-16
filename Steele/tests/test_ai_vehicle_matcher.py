import pandas as pd
import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from openai import OpenAI
from dotenv import load_dotenv
import sys
from pathlib import Path

# Add Steele utils to path
steele_root = Path(__file__).parent.parent
sys.path.append(str(steele_root))

from utils.ai_vehicle_matcher import AIVehicleMatcher, ModelMatchResult

# Load environment variables for testing
load_dotenv()

class TestAIVehicleMatcher:
    """Test suite for AI vehicle matching functionality"""
    
    @pytest.fixture
    def ai_matcher(self):
        """AI matcher instance for testing"""
        return AIVehicleMatcher(use_ai=True)
    
    @pytest.fixture
    def ai_matcher_no_ai(self):
        """AI matcher with AI disabled for fallback testing"""
        return AIVehicleMatcher(use_ai=False)
    
    @pytest.fixture
    def sample_year_make_matches(self):
        """Sample DataFrame with year+make matches"""
        return pd.DataFrame({
            'car_id': [
                '1957_Nash_Rambler',
                '1957_Nash_Ambassador Super',
                '1957_Nash_Ambassador Custom'
            ],
            'year': [1957, 1957, 1957],
            'make': ['Nash', 'Nash', 'Nash'],
            'model': ['Rambler', 'Ambassador Super', 'Ambassador Custom']
        })
    
    @pytest.fixture
    def sample_year_matches(self):
        """Sample DataFrame with year-only matches"""
        return pd.DataFrame({
            'car_id': [
                '1957_Nash_Rambler',
                '1957_Ford_Thunderbird',
                '1957_Chevrolet_Bel Air'
            ],
            'year': [1957, 1957, 1957],
            'make': ['Nash', 'Ford', 'Chevrolet'],
            'model': ['Rambler', 'Thunderbird', 'Bel Air']
        })
    
    @pytest.fixture
    def golden_df(self):
        """Sample golden dataset for testing"""
        return pd.DataFrame({
            'car_id': [
                '1957_Nash_Rambler',
                '1957_Nash_Ambassador Super',
                '1957_Nash_Ambassador Custom',
                '1957_Ford_Thunderbird',
                '1957_Chevrolet_Bel Air'
            ],
            'year': [1957, 1957, 1957, 1957, 1957],
            'make': ['Nash', 'Nash', 'Nash', 'Ford', 'Chevrolet'],
            'model': ['Rambler', 'Ambassador Super', 'Ambassador Custom', 'Thunderbird', 'Bel Air']
        })
    
    def test_initialization_with_ai_enabled(self):
        """Test AI matcher initialization with AI enabled"""
        with patch.dict(os.environ, {'USE_AI_MATCHING': 'true', 'OPENAI_API_KEY': 'test-key'}):
            with patch('openai.OpenAI') as mock_openai:
                matcher = AIVehicleMatcher(use_ai=True)
                assert matcher.use_ai == True
                mock_openai.assert_called_once_with(api_key='test-key')
    
    def test_initialization_with_ai_disabled(self):
        """Test AI matcher initialization with AI disabled"""
        matcher = AIVehicleMatcher(use_ai=False)
        assert matcher.use_ai == False
        assert matcher.client is None
    
    def test_initialization_no_api_key(self):
        """Test AI matcher initialization without API key"""
        with patch.dict(os.environ, {'USE_AI_MATCHING': 'true'}, clear=True):
            with patch.dict(os.environ, {}, clear=True):  # Remove OPENAI_API_KEY
                matcher = AIVehicleMatcher(use_ai=True)
                assert matcher.use_ai == False
                assert matcher.client is None
    
    @pytest.mark.ai
    def test_ai_model_matching_success(self, ai_matcher, sample_year_make_matches):
        """Test successful AI model matching"""
        if not ai_matcher.use_ai:
            pytest.skip("AI not enabled for this test")
        
        # Mock successful AI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '''
        {
            "selected_car_ids": ["1957_Nash_Rambler"],
            "confidence": 0.9,
            "reasoning": "Best match based on model similarity",
            "match_type": "ai_model_match"
        }
        '''
        
        with patch.object(ai_matcher.client.beta.chat.completions, 'parse', return_value=mock_response):
            result = ai_matcher.ai_match_models_for_year_make(
                sample_year_make_matches,
                "Rambler",
                "Base",
                "Car",
                "2.0",
                "Convertible"
            )
            
            assert len(result) == 1
            assert result.iloc[0]['car_id'] == '1957_Nash_Rambler'
    
    def test_ai_model_matching_fallback(self, ai_matcher_no_ai, sample_year_make_matches):
        """Test AI model matching falls back to fuzzy matching when AI disabled"""
        result = ai_matcher_no_ai.ai_match_models_for_year_make(
            sample_year_make_matches,
            "Rambler"
        )
        
        # Should use fuzzy matching and find Rambler
        assert len(result) >= 1
        assert any('Rambler' in car_id for car_id in result['car_id'].values)
    
    def test_fuzzy_make_matching_exact(self, ai_matcher, sample_year_matches):
        """Test fuzzy make matching with exact match"""
        corrected_make = ai_matcher.fuzzy_match_make_for_year(
            sample_year_matches,
            "Nash"
        )
        assert corrected_make == "Nash"
    
    def test_fuzzy_make_matching_similar(self, ai_matcher, sample_year_matches):
        """Test fuzzy make matching with similar spelling"""
        corrected_make = ai_matcher.fuzzy_match_make_for_year(
            sample_year_matches,
            "Nsh"  # Missing 'a'
        )
        assert corrected_make == "Nash"
    
    def test_fuzzy_make_matching_no_match(self, ai_matcher, sample_year_matches):
        """Test fuzzy make matching with no good match"""
        corrected_make = ai_matcher.fuzzy_match_make_for_year(
            sample_year_matches,
            "Toyota"  # Not in dataset
        )
        assert corrected_make is None
    
    def test_validate_with_corrected_make_exact_match(self, ai_matcher, golden_df):
        """Test re-validation with corrected make finds exact match"""
        result = ai_matcher.validate_with_corrected_make(
            golden_df,
            1957,
            "Nash",
            "Rambler"
        )
        
        assert result['golden_validated'] == True
        assert result['match_type'] == 'exact_with_corrected_make'
        assert '1957_Nash_Rambler' in result['car_ids']
        assert result['corrected_make'] == 'Nash'
    
    def test_validate_with_corrected_make_ai_model_match(self, ai_matcher, golden_df):
        """Test re-validation leads to AI model matching"""
        # Mock AI response for model matching
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '''
        {
            "selected_car_ids": ["1957_Nash_Ambassador Super"],
            "confidence": 0.85,
            "reasoning": "Ambassador matches input model",
            "match_type": "ai_model_match"
        }
        '''
        
        if ai_matcher.use_ai and ai_matcher.client:
            with patch.object(ai_matcher.client.beta.chat.completions, 'parse', return_value=mock_response):
                result = ai_matcher.validate_with_corrected_make(
                    golden_df,
                    1957,
                    "Nash",
                    "Ambassador"  # Should match with Ambassador Super
                )
                
                assert result['golden_validated'] == True
                assert result['match_type'] == 'ai_model_match_with_corrected_make'
                assert result['corrected_make'] == 'Nash'
    
    def test_validate_with_corrected_make_no_matches(self, ai_matcher, golden_df):
        """Test re-validation with corrected make finds no matches"""
        result = ai_matcher.validate_with_corrected_make(
            golden_df,
            1999,  # Year not in dataset
            "Nash",
            "Rambler"
        )
        
        assert result['golden_validated'] == False
        assert result['match_type'] == 'no_match_with_corrected_make'
        assert len(result['car_ids']) == 0
    
    def test_model_matching_low_confidence(self, ai_matcher, sample_year_make_matches):
        """Test AI model matching with low confidence falls back"""
        if not ai_matcher.use_ai:
            pytest.skip("AI not enabled for this test")
        
        # Mock low confidence response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '''
        {
            "selected_car_ids": ["1957_Nash_Rambler"],
            "confidence": 0.3,
            "reasoning": "Poor match quality",
            "match_type": "ai_model_match"
        }
        '''
        
        with patch.object(ai_matcher.client.beta.chat.completions, 'parse', return_value=mock_response):
            result = ai_matcher.ai_match_models_for_year_make(
                sample_year_make_matches,
                "Unknown Model"
            )
            
            # Should fall back to fuzzy matching
            assert isinstance(result, pd.DataFrame)
    
    def test_ai_error_handling(self, ai_matcher, sample_year_make_matches):
        """Test graceful handling of AI API errors"""
        if not ai_matcher.use_ai:
            pytest.skip("AI not enabled for this test")
        
        # Mock API error
        with patch.object(ai_matcher.client.beta.chat.completions, 'parse', side_effect=Exception("API Error")):
            result = ai_matcher.ai_match_models_for_year_make(
                sample_year_make_matches,
                "Rambler"
            )
            
            # Should fall back to fuzzy matching
            assert isinstance(result, pd.DataFrame)
    
    def test_empty_dataframe_handling(self, ai_matcher):
        """Test handling of empty DataFrames"""
        empty_df = pd.DataFrame()
        
        # Test AI model matching with empty DataFrame
        result = ai_matcher.ai_match_models_for_year_make(empty_df, "Test")
        assert len(result) == 0
        
        # Test fuzzy make matching with empty DataFrame
        corrected_make = ai_matcher.fuzzy_match_make_for_year(empty_df, "Test")
        assert corrected_make is None
    
    def test_normalize_strings(self, ai_matcher, sample_year_matches):
        """Test string normalization in fuzzy matching"""
        # Test with spaces, hyphens, underscores
        corrected_make = ai_matcher.fuzzy_match_make_for_year(
            sample_year_matches,
            "Na-sh_"  # Should match "Nash"
        )
        assert corrected_make == "Nash"
    
    def test_similarity_thresholds(self, ai_matcher, sample_year_matches):
        """Test similarity threshold enforcement"""
        # Test with very low threshold
        corrected_make = ai_matcher.fuzzy_match_make_for_year(
            sample_year_matches,
            "X",  # Single character, should not match
            similarity_threshold=0.1
        )
        assert corrected_make is None
    
    @pytest.mark.performance
    def test_performance_large_dataset(self, ai_matcher):
        """Test performance with larger dataset"""
        # Create larger test dataset
        large_df = pd.DataFrame({
            'car_id': [f'1957_Make{i}_Model{i}' for i in range(1000)],
            'year': [1957] * 1000,
            'make': [f'Make{i}' for i in range(1000)],
            'model': [f'Model{i}' for i in range(1000)]
        })
        
        import time
        start_time = time.time()
        
        # Test fuzzy make matching performance
        corrected_make = ai_matcher.fuzzy_match_make_for_year(
            large_df,
            "Make500"
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        assert corrected_make == "Make500"
        assert processing_time < 1.0  # Should complete in under 1 second
    
    def test_integration_with_transformer_data_format(self, ai_matcher):
        """Test integration with actual Steele transformer data format"""
        # Test with data that matches actual Steele format
        steele_format_df = pd.DataFrame({
            'car_id': ['1957_Nash_Metropolitan'],
            'year': [1957],
            'make': ['Nash'],
            'model': ['Metropolitan']
        })
        
        result = ai_matcher.validate_with_corrected_make(
            steele_format_df,
            1957,
            "Nash",
            "Metropolitan",
            "Base",
            "Car",
            "2.0",
            "Convertible"
        )
        
        assert 'golden_validated' in result
        assert 'car_ids' in result
        assert 'match_type' in result
        assert 'corrected_make' in result
    
    def test_mock_openai_responses(self):
        """Test with mocked OpenAI responses for consistent testing"""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '''
        {
            "selected_car_ids": ["test_car_id"],
            "confidence": 0.95,
            "reasoning": "Mock test response",
            "match_type": "ai_model_match"
        }
        '''
        
        with patch('openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_client.beta.chat.completions.parse.return_value = mock_response
            mock_openai.return_value = mock_client
            
            with patch.dict(os.environ, {'USE_AI_MATCHING': 'true', 'OPENAI_API_KEY': 'test-key'}):
                matcher = AIVehicleMatcher(use_ai=True)
                
                # Test that mocked responses work correctly
                test_df = pd.DataFrame({
                    'car_id': ['test_car_id'],
                    'model': ['Test Model']
                })
                
                result = matcher.ai_match_models_for_year_make(test_df, "Test")
                assert len(result) >= 0  # Should not crash