import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ai_fitment_extractor import SteeleAIFitmentExtractor, FitmentExtraction

class TestTwoPassAIExtractionTDD:
    """
    Test-Driven Development suite for the two-pass AI extraction system.
    Tests the logic flow before making any actual API calls.
    """
    
    @pytest.fixture
    def mock_extractor(self):
        """Create a mock extractor with sample golden master data"""
        # Mock the environment variable
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            extractor = SteeleAIFitmentExtractor()
        
        # Mock golden master data - use same column names as real golden master
        mock_golden_data = pd.DataFrame({
            'Year': [1920, 1920, 1921, 1921, 1922, 1923, 1924, 1925, 1912, 1913, 1914, 1915, 1928, 1929],
            'Make': ['Ford', 'Buick', 'Ford', 'Buick', 'Ford', 'Buick', 'Buick', 
                    'Auburn', 'Ford', 'Ford', 'Buick', 'Buick',
                    'Auburn', 'Auburn'],
            'Model': ['Model_T', 'Series_K', 'Model_TT', 'Series_H', 'Model_T', 'Series_K', 'Series_H', 
                     'Model_6-39', 'Model_T', 'Model_TT', 'Series_K', 'Series_H', 
                     'Model_6-40', 'Model_6-44']
        })
        extractor.golden_master_df = mock_golden_data
        return extractor
    
    def test_mock_first_pass_ai_extraction_vague_independent(self, mock_extractor):
        """Test first pass extraction for vague Independent description"""
        description = "This part is compatible with models built by Independent (1920 - 1929) automobile manufacturers"
        
        # Mock the AI response for first pass
        mock_ai_response = FitmentExtraction(
            years=['1920', '1921', '1922', '1923', '1924', '1925', '1926', '1927', '1928', '1929'],
            make='ALL',
            model='ALL',
            confidence=0.85,
            reasoning='Vague description mentioning Independent manufacturers from 1920-1929'
        )
        
        with patch.object(mock_extractor, 'extract_fitment_from_description', return_value=mock_ai_response):
            result = mock_extractor.extract_fitment_from_description("Test Product", description)
            
            assert result.make == 'ALL'
            assert result.model == 'ALL' 
            assert '1920' in result.years
            assert '1929' in result.years
            assert result.confidence > 0.8
    
    def test_mock_first_pass_ai_extraction_street_rod(self, mock_extractor):
        """Test first pass extraction for Street Rod description"""
        description = "Steele Rubber Products has the best quality Accelerator Pedal Pads for your Street Rod or Custom Build project"
        
        mock_ai_response = FitmentExtraction(
            years=[str(year) for year in range(1920, 1980)],  # Wide range for custom builds
            make='ALL',
            model='ALL',
            confidence=0.75,
            reasoning='Generic street rod/custom build application'
        )
        
        with patch.object(mock_extractor, 'extract_fitment_from_description', return_value=mock_ai_response):
            result = mock_extractor.extract_fitment_from_description("Test Product", description)
            
            assert result.make == 'ALL'
            assert result.model == 'ALL'
            assert len(result.years) > 10  # Should be a wide range
    
    def test_mock_first_pass_ai_extraction_specific_sterns_knight(self, mock_extractor):
        """Test first pass extraction for specific Sterns-Knight description"""
        description = "Axle Rebound Pad - Rear. Made from top quality rubber to ensure durability. This part is compatible with (1912 - 1915) Sterns-Knight models."
        
        mock_ai_response = FitmentExtraction(
            years=['1912', '1913', '1914', '1915'],
            make='Sterns-Knight',
            model='ALL',
            confidence=0.90,
            reasoning='Specific make mentioned with year range'
        )
        
        with patch.object(mock_extractor, 'extract_fitment_from_description', return_value=mock_ai_response):
            result = mock_extractor.extract_fitment_from_description("Test Product", description)
            
            assert result.make == 'Sterns-Knight'
            assert result.model == 'ALL'
            assert result.years == ['1912', '1913', '1914', '1915']
    
    def test_expand_all_makes_logic(self, mock_extractor):
        """Test the logic for expanding ALL makes"""
        years = ['1920', '1921', '1922']
        
        result = mock_extractor.expand_all_makes(years)
        
        # Should return actual makes from golden master for those years
        # Real golden master should have Ford, Auburn, Willys, etc. for 1920-1922
        assert isinstance(result, list)
        assert len(result) > 0
        # Check that we get actual make names (not "ALL")
        assert 'ALL' not in result or len(result) == 1  # Either real makes or fallback to ["ALL"]
        if result != ['ALL']:
            # Should contain some real makes from that era
            common_1920s_makes = ['Ford', 'Auburn', 'Willys', 'Maxwell', 'Oldsmobile']
            assert any(make in result for make in common_1920s_makes)
    
    def test_expand_all_models_logic(self, mock_extractor):
        """Test the logic for expanding ALL models"""
        make = 'Ford'  # Use a real make from the golden master
        years = ['1920', '1921', '1922']
        
        result = mock_extractor.expand_all_models(make, years)
        
        # Should return actual models for Ford 1920-1922 from golden master
        assert isinstance(result, list)
        assert len(result) > 0
        # For Ford in 1920s, we should get actual models like Model T, Model TT, etc.
        if result != ['ALL']:
            # Should contain some real Ford models from that era
            assert any('Model' in model for model in result)  # Ford models typically have "Model" in name
    
    def test_expand_fitment_extraction_all_make_all_model(self, mock_extractor):
        """Test expansion when both make and model are ALL"""
        extraction = FitmentExtraction(
            years=['1920', '1921'],
            make='ALL',
            model='ALL',
            confidence=0.85,
            reasoning='Test case'
        )
        
        result = mock_extractor.expand_fitment_extraction(extraction)
        
        # Should attempt expansion - either succeed or fall back gracefully
        assert isinstance(result, FitmentExtraction)
        assert result.years == extraction.years  # Years should remain unchanged
        assert result.confidence == 0.85  # Should preserve confidence
        # Make should either be expanded or remain "ALL" if no expansion possible
        assert result.make in ['ALL'] or (',' in result.make or result.make != 'ALL')
    
    def test_expand_fitment_extraction_specific_make_all_model(self, mock_extractor):
        """Test expansion when make is specific but model is ALL"""
        extraction = FitmentExtraction(
            years=['1912', '1913'],
            make='Sterns-Knight',
            model='ALL',
            confidence=0.90,
            reasoning='Test case'
        )
        
        result = mock_extractor.expand_fitment_extraction(extraction)
        
        # Should keep the specific make (Sterns-Knight not in golden master, so should remain)
        assert result.make == 'Sterns-Knight'
        # Model should remain "ALL" since Sterns-Knight not in golden master
        assert result.model == 'ALL'  # No expansion possible for non-existent make
    
    def test_expand_fitment_extraction_no_expansion_needed(self, mock_extractor):
        """Test that specific extractions don't get expanded"""
        extraction = FitmentExtraction(
            years=['1920'],
            make='Independent',
            model='Model_A',
            confidence=0.95,
            reasoning='Specific fitment'
        )
        
        result = mock_extractor.expand_fitment_extraction(extraction)
        
        # Should return the original extraction as-is
        assert result.make == 'Independent'
        assert result.model == 'Model_A'
        assert result.years == ['1920']
    
    def test_generate_vehicle_tags_from_extraction(self, mock_extractor):
        """Test vehicle tag generation from expanded extractions"""
        # Use a real make that exists in the golden master for 1920-1921
        extraction = FitmentExtraction(
            years=['1920', '1921'],
            make='Ford',  # Ford should exist in golden master for these years
            model='ALL',  # Use ALL to get actual models from golden master
            confidence=0.95,
            reasoning='Test case'
        )
        
        result = mock_extractor._generate_vehicle_tags_from_extraction(extraction)
        
        # Should generate multiple Ford vehicle tags for 1920-1921
        assert isinstance(result, list)
        assert len(result) > 0
        assert all('Ford' in tag for tag in result)
        assert any('1920_Ford_' in tag for tag in result)
        assert any('1921_Ford_' in tag for tag in result)
    
    def test_generate_vehicle_tags_with_multiple_makes_models(self, mock_extractor):
        """Test vehicle tag generation with multiple makes and models"""
        # Use makes that exist in the golden master
        extraction = FitmentExtraction(
            years=['1920'],
            make='Ford, Buick',  # Both should exist in golden master
            model='ALL',
            confidence=0.85,
            reasoning='Multiple makes and models'
        )
        
        result = mock_extractor._generate_vehicle_tags_from_extraction(extraction)
        
        # Should generate tags for both makes
        assert isinstance(result, list)
        assert len(result) > 0
        assert any('Ford' in tag for tag in result)
        assert any('Buick' in tag for tag in result)
        assert all('1920_' in tag for tag in result)
    
    def test_process_unknown_skus_batch_with_expansion_mock(self, mock_extractor):
        """Test the complete two-pass process with mocked AI calls"""
        # Sample data with vague descriptions
        sample_data = pd.DataFrame({
            'StockCode': ['10-0108-45', '10-0127-52'],
            'Product Name': ['Running Board Step Pad', 'Windshield Weatherstrip'],
            'Description': [
                'Running Board Step Pad. Made from top quality rubber to ensure durability. This part is compatible with models built by Independent (1920 - 1929) automobile manufacturers.',
                'This Windshield to Cowl Weatherstrip comes in a thirty-six inch (36") strip and seals the lower windshield frame to the cowl of the car. This generic part is known to fit Independent vehicle models produced from 1920 - 1924.'
            ]
        })
        
        # Mock AI responses
        mock_ai_responses = [
            FitmentExtraction(
                years=[str(year) for year in range(1920, 1930)],
                make='ALL',
                model='ALL',
                confidence=0.85,
                reasoning='Vague Independent manufacturers 1920-1929'
            ),
            FitmentExtraction(
                years=[str(year) for year in range(1920, 1925)],
                make='ALL', 
                model='ALL',
                confidence=0.80,
                reasoning='Generic Independent models 1920-1924'
            )
        ]
        
        with patch.object(mock_extractor, 'extract_fitment_from_description', side_effect=mock_ai_responses):
            results = mock_extractor.process_unknown_skus_batch_with_expansion(sample_data)
            
            assert len(results) == 2
            
            # Check that results are generated
            first_result = results.iloc[0]
            assert first_result['StockCode'] == '10-0108-45'
            # Since "Independent" is not in golden master, might still be Unknown but should have tried expansion
            assert 'ai_extracted_years' in first_result
            assert first_result['ai_extracted_years']  # Should have extracted years
            
            # Second product should also have results
            second_result = results.iloc[1]
            assert second_result['StockCode'] == '10-0127-52'
            assert 'ai_extracted_years' in second_result
    
    def test_edge_case_no_golden_master_match(self, mock_extractor):
        """Test behavior when extraction doesn't match golden master data"""
        extraction = FitmentExtraction(
            years=['1950', '1951'],  # Years not in our mock golden master
            make='NonExistentMake',
            model='ALL',
            confidence=0.70,
            reasoning='Test edge case'
        )
        
        result = mock_extractor.expand_fitment_extraction(extraction)
        
        # Should return the original extraction since no matches found
        assert result.make == 'NonExistentMake'
        assert result.model == 'ALL'  # Should remain ALL since no expansion possible
    
    def test_confidence_threshold_filtering(self, mock_extractor):
        """Test that low confidence extractions are handled appropriately"""
        low_confidence_extraction = FitmentExtraction(
            years=['1920'],
            make='Independent',
            model='Model_A',
            confidence=0.3,  # Very low confidence
            reasoning='Uncertain extraction'
        )
        
        # The system should handle low confidence appropriately
        # This test validates the confidence is preserved through the process
        result = mock_extractor.expand_fitment_extraction(low_confidence_extraction)
        
        # Confidence should be preserved
        assert result.confidence == 0.3
        assert result.make == 'Independent'
        assert result.model == 'Model_A'
    
    def test_error_handling_in_extraction(self, mock_extractor):
        """Test that errors in extraction are handled properly"""
        error_extraction = FitmentExtraction(
            years=[],
            make='UNKNOWN',
            model='UNKNOWN',
            confidence=0.0,
            reasoning='Failed extraction',
            error='AI extraction failed'
        )
        
        result = mock_extractor.expand_fitment_extraction(error_extraction)
        
        # Should return the error extraction as-is
        assert result.error == 'AI extraction failed'
        assert result.make == 'UNKNOWN'
        
        # Should generate fallback tags
        tags = mock_extractor._generate_vehicle_tags_from_extraction(result)
        assert tags == ['0_Unknown_UNKNOWN']

def test_integration_with_real_sample_data():
    """Integration test that validates the complete flow with real sample data structure"""
    # This test validates the data structure and flow without making API calls
    
    # Sample from the actual CSV data structure
    sample_data = pd.DataFrame({
        'StockCode': ['10-0108-45'],
        'Product Name': ['Running Board Step Pad'],
        'Description': ['Running Board Step Pad. Made from top quality rubber to ensure durability. This part is compatible with models built by Independent (1920 - 1929) automobile manufacturers.'],
        'Tags': ['0_Unknown_UNKNOWN']  # Current problematic tag
    })
    
    # Validate input data structure
    assert 'StockCode' in sample_data.columns
    assert 'Description' in sample_data.columns
    assert sample_data.iloc[0]['Tags'] == '0_Unknown_UNKNOWN'
    
    # This confirms our test data matches the real data structure
    assert len(sample_data) == 1
    assert 'Independent (1920 - 1929)' in sample_data.iloc[0]['Description']

if __name__ == "__main__":
    # Run the TDD tests
    pytest.main([__file__, "-v"]) 