import pytest
import pandas as pd
from unittest.mock import Mock, patch
from utils.model_refinement import ModelRefinementEngine, BodyTypeProcessor
from utils.ai_extraction import ProductData
from utils.exceptions import GoldenMasterValidationError


class TestBodyTypeProcessor:
    """Test suite for body type processing functionality"""
    
    @pytest.fixture
    def processor(self):
        """Create a BodyTypeProcessor instance for testing"""
        return BodyTypeProcessor()
    
    @pytest.fixture
    def sample_golden_df(self):
        """Create sample golden master data with body types"""
        return pd.DataFrame({
            'year': [1965, 1966, 1967, 1968, 1969, 1970, 1971, 1972],
            'make': ['Ford', 'Ford', 'Ford', 'Ford', 'Chevrolet', 'Chevrolet', 'Dodge', 'Dodge'],
            'model': ['Mustang', 'Mustang', 'Mustang', 'Mustang', 'Camaro', 'Camaro', 'Challenger', 'Challenger'],
            'body_type': ['Coupe', 'Coupe', 'Fastback', 'Fastback', 'Coupe', 'Coupe', 'Coupe', 'Coupe'],
            'make_normalized': ['ford', 'ford', 'ford', 'ford', 'chevrolet', 'chevrolet', 'dodge', 'dodge'],
            'model_normalized': ['mustang', 'mustang', 'mustang', 'mustang', 'camaro', 'camaro', 'challenger', 'challenger'],
            'body_type_normalized': ['coupe', 'coupe', 'fastback', 'fastback', 'coupe', 'coupe', 'coupe', 'coupe']
        })
    
    def test_process_body_type_all_2_door(self, processor, sample_golden_df):
        """Test processing of 'ALL (2-Door)' body type specification"""
        description = "Compatible with ALL (2-Door) models"
        
        result = processor.process_body_type_specification(description, sample_golden_df)
        
        assert isinstance(result, dict)
        assert 'body_type_filter' in result
        assert result['body_type_filter'] == '2-Door'
        assert 'applicable_models' in result
    
    def test_process_body_type_all_4_door(self, processor, sample_golden_df):
        """Test processing of 'ALL (4-Door)' body type specification"""
        description = "Compatible with ALL (4-Door) models"
        
        result = processor.process_body_type_specification(description, sample_golden_df)
        
        assert isinstance(result, dict)
        assert result['body_type_filter'] == '4-Door'
    
    def test_process_body_type_a_body(self, processor, sample_golden_df):
        """Test processing of 'ALL (A-Body)' chassis specification"""
        description = "Fits ALL (A-Body) chassis vehicles"
        
        result = processor.process_body_type_specification(description, sample_golden_df)
        
        assert isinstance(result, dict)
        assert result['body_type_filter'] == 'A-Body'
    
    def test_process_combined_body_type(self, processor, sample_golden_df):
        """Test processing of combined body type specifications"""
        description = "Compatible with 2 & 4-Door Sedan models"
        
        result = processor.process_body_type_specification(description, sample_golden_df)
        
        assert isinstance(result, dict)
        assert 'body_type_filter' in result
        assert isinstance(result['body_type_filter'], list)
        assert '2-Door' in result['body_type_filter']
        assert '4-Door' in result['body_type_filter']
    
    def test_validate_body_type_against_historical_data(self, processor, sample_golden_df):
        """Test validation of body type against historical model configurations"""
        # Test valid body type for model
        assert processor.validate_body_type_for_model(1965, 'Ford', 'Mustang', 'Coupe', sample_golden_df) == True
        
        # Test invalid body type for model
        assert processor.validate_body_type_for_model(1965, 'Ford', 'Mustang', 'Truck', sample_golden_df) == False
    
    def test_get_available_body_types_for_model(self, processor, sample_golden_df):
        """Test getting available body types for specific model"""
        body_types = processor.get_available_body_types_for_model(1965, 'Ford', 'Mustang', sample_golden_df)
        
        assert isinstance(body_types, list)
        assert 'Coupe' in body_types
        assert len(body_types) > 0
    
    def test_filter_models_by_body_type(self, processor, sample_golden_df):
        """Test filtering models by body type specification"""
        # Test filtering by specific body type
        filtered_models = processor.filter_models_by_body_type(
            sample_golden_df, 'Ford', 1965, 'Coupe'
        )
        
        assert isinstance(filtered_models, list)
        assert 'Mustang' in filtered_models
    
    def test_expand_all_specification(self, processor, sample_golden_df):
        """Test expansion of 'ALL' specification to specific models"""
        expanded_models = processor.expand_all_specification(
            sample_golden_df, 'Ford', 1965, None
        )
        
        assert isinstance(expanded_models, list)
        assert 'Mustang' in expanded_models
    
    def test_body_type_pattern_matching(self, processor):
        """Test body type pattern matching from descriptions"""
        test_cases = [
            ("ALL (2-Door)", "2-Door"),
            ("ALL (4-Door)", "4-Door"),
            ("ALL (A-Body)", "A-Body"),
            ("ALL (B-Body)", "B-Body"),
            ("2 & 4-Door Sedan", ["2-Door", "4-Door"]),
            ("Coupe models only", "Coupe"),
            ("Convertible and Hardtop", ["Convertible", "Hardtop"])
        ]
        
        for description, expected in test_cases:
            result = processor.extract_body_type_from_description(description)
            assert result == expected, f"Failed for: {description}"


class TestModelRefinementEngine:
    """Test suite for model refinement engine functionality"""
    
    @pytest.fixture
    def engine(self):
        """Create a ModelRefinementEngine instance for testing"""
        mock_client = Mock()
        sample_golden_df = pd.DataFrame({
            'year': [1965, 1966, 1967, 1968, 1969, 1970],
            'make': ['Ford', 'Ford', 'Ford', 'Ford', 'Chevrolet', 'Chevrolet'],
            'model': ['Mustang', 'Mustang', 'Mustang', 'Mustang', 'Camaro', 'Camaro'],
            'body_type': ['Coupe', 'Coupe', 'Fastback', 'Fastback', 'Coupe', 'Coupe'],
            'make_normalized': ['ford', 'ford', 'ford', 'ford', 'chevrolet', 'chevrolet'],
            'model_normalized': ['mustang', 'mustang', 'mustang', 'mustang', 'camaro', 'camaro']
        })
        return ModelRefinementEngine(mock_client, sample_golden_df)
    
    @pytest.fixture
    def sample_product_data(self):
        """Create sample product data for testing"""
        return pd.DataFrame({
            'title': ['Accelerator Pedal Pad', 'Brake Pad Set', 'Universal Mirror'],
            'year_min': [1965, 1969, 1900],
            'year_max': [1970, 1970, 2024],
            'make': ['Ford', 'Chevrolet', 'Universal'],
            'model': ['Mustang', 'Camaro', 'All'],
            'mpn': ['10-0001-40', '10-0002-35', '10-0003-35'],
            'description': ['For 1965-1970 Ford Mustang', 'For 1969-1970 Chevrolet Camaro', 'Universal fit for all vehicles']
        })
    
    def test_refine_models_with_ai_success(self, engine, sample_product_data):
        """Test successful AI model refinement"""
        # Mock AI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Mustang, Fastback"
        
        engine.client.chat.completions.create.return_value = mock_response
        
        refined_df = engine.refine_models_with_ai(sample_product_data)
        
        assert isinstance(refined_df, pd.DataFrame)
        assert 'refined_models' in refined_df.columns
        assert len(refined_df) == len(sample_product_data)
    
    def test_get_refined_models_from_ai(self, engine):
        """Test getting refined models from AI"""
        # Mock AI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Mustang, Fastback"
        
        engine.client.chat.completions.create.return_value = mock_response
        
        refined_models = engine.get_refined_models_from_ai(
            "Accelerator Pedal Pad", 1965, "Ford", "Mustang"
        )
        
        assert isinstance(refined_models, str)
        assert "Mustang" in refined_models
        assert "Fastback" in refined_models
    
    def test_expand_models_to_car_ids(self, engine):
        """Test expansion of models to car IDs"""
        # Create sample row
        sample_row = pd.Series({
            'year_min': 1965,
            'year_max': 1970,
            'make': 'Ford',
            'model': 'Mustang',
            'refined_models': 'Mustang, Fastback'
        })
        
        car_ids = engine.expand_models_to_car_ids(sample_row)
        
        assert isinstance(car_ids, list)
        # Should return empty list or actual car IDs depending on golden master data
    
    def test_validate_model_selection(self, engine):
        """Test validation of model selection"""
        models = ['Mustang', 'Fastback']
        
        is_valid = engine.validate_model_selection(models, 1965, 'Ford')
        
        assert isinstance(is_valid, bool)
        # Should validate against golden master
    
    def test_create_refinement_prompt(self, engine):
        """Test creation of AI refinement prompt"""
        prompt = engine.create_refinement_prompt(
            "Accelerator Pedal Pad", 1965, "Ford", "Mustang"
        )
        
        assert isinstance(prompt, str)
        assert "Accelerator Pedal Pad" in prompt
        assert "Ford" in prompt
        assert "Mustang" in prompt
        assert "1965" in prompt
    
    def test_handle_universal_compatibility(self, engine):
        """Test handling of universal compatibility products"""
        sample_row = pd.Series({
            'year_min': 1900,
            'year_max': 2024,
            'make': 'Universal',
            'model': 'All',
            'title': 'Universal Mirror',
            'description': 'Universal fit for all vehicles'
        })
        
        result = engine.handle_universal_compatibility(sample_row)
        
        assert isinstance(result, str)
        assert result == "Universal"
    
    def test_ai_refinement_error_handling(self, engine):
        """Test error handling in AI refinement"""
        # Mock AI error
        engine.client.chat.completions.create.side_effect = Exception("API Error")
        
        refined_models = engine.get_refined_models_from_ai(
            "Test Product", 1965, "Ford", "Mustang"
        )
        
        # Should return fallback value
        assert isinstance(refined_models, str)
        assert refined_models == "Mustang"  # Fallback to original
    
    def test_confidence_scoring_for_refinement(self, engine):
        """Test confidence scoring for AI refinement"""
        # Mock AI response with confidence indicators
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Mustang (high confidence), Fastback (medium confidence)"
        
        engine.client.chat.completions.create.return_value = mock_response
        
        result = engine.get_refined_models_with_confidence(
            "Accelerator Pedal Pad", 1965, "Ford", "Mustang"
        )
        
        assert isinstance(result, dict)
        assert 'models' in result
        assert 'confidence' in result
    
    def test_historical_accuracy_validation(self, engine):
        """Test validation against historical accuracy"""
        # Test with historically accurate combination
        is_accurate = engine.validate_historical_accuracy(1965, 'Ford', 'Mustang')
        assert isinstance(is_accurate, bool)
        
        # Test with historically inaccurate combination
        is_accurate = engine.validate_historical_accuracy(1960, 'Ford', 'Mustang')
        assert is_accurate == False
    
    def test_batch_model_refinement(self, engine, sample_product_data):
        """Test batch processing of model refinement"""
        # Mock successful AI responses
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Mustang, Fastback"
        
        engine.client.chat.completions.create.return_value = mock_response
        
        refined_df = engine.process_batch_refinement(sample_product_data)
        
        assert isinstance(refined_df, pd.DataFrame)
        assert len(refined_df) == len(sample_product_data)
        assert 'refined_models' in refined_df.columns
        assert 'refinement_confidence' in refined_df.columns


class TestModelRefinementIntegration:
    """Integration tests for model refinement functionality"""
    
    @pytest.fixture
    def engine(self):
        mock_client = Mock()
        golden_master_path = Path(__file__).parent.parent.parent / "shared" / "data" / "master_ultimate_golden.csv"
        
        # Load real golden master for integration tests
        from utils.golden_master_validation import GoldenMasterValidator
        validator = GoldenMasterValidator(str(golden_master_path))
        golden_df = validator.load_golden_master()
        
        return ModelRefinementEngine(mock_client, golden_df)
    
    def test_real_world_model_refinement(self, engine):
        """Test model refinement with real-world data"""
        # Mock AI response for real-world scenario
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Mustang, Fastback, Coupe"
        
        engine.client.chat.completions.create.return_value = mock_response
        
        refined_models = engine.get_refined_models_from_ai(
            "Accelerator Pedal Pad", 1965, "Ford", "Mustang"
        )
        
        assert isinstance(refined_models, str)
        assert len(refined_models) > 0
    
    def test_performance_with_large_dataset(self, engine):
        """Test performance with large dataset"""
        import time
        
        # Create large test dataset
        large_df = pd.DataFrame({
            'title': ['Test Product'] * 100,
            'year_min': [1965] * 100,
            'year_max': [1970] * 100,
            'make': ['Ford'] * 100,
            'model': ['Mustang'] * 100,
            'mpn': [f'TEST-{i:03d}' for i in range(100)]
        })
        
        # Mock AI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Mustang"
        
        engine.client.chat.completions.create.return_value = mock_response
        
        start_time = time.time()
        refined_df = engine.refine_models_with_ai(large_df)
        elapsed_time = time.time() - start_time
        
        # Should process reasonably quickly
        assert elapsed_time < 10.0  # Less than 10 seconds for 100 items
        assert len(refined_df) == 100
    
    def test_error_recovery_in_batch_processing(self, engine):
        """Test error recovery in batch processing"""
        # Create test data with some problematic entries
        test_df = pd.DataFrame({
            'title': ['Good Product', 'Bad Product', 'Good Product'],
            'year_min': [1965, None, 1969],  # One with missing year
            'year_max': [1970, None, 1970],
            'make': ['Ford', 'InvalidMake', 'Chevrolet'],
            'model': ['Mustang', 'InvalidModel', 'Camaro'],
            'mpn': ['GOOD-001', 'BAD-002', 'GOOD-003']
        })
        
        # Mock AI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Mustang"
        
        engine.client.chat.completions.create.return_value = mock_response
        
        # Should handle errors gracefully
        refined_df = engine.refine_models_with_ai(test_df)
        
        assert isinstance(refined_df, pd.DataFrame)
        assert len(refined_df) == 3  # Should process all rows
        assert 'refined_models' in refined_df.columns