import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch
from utils.golden_master_validation import GoldenMasterValidator
from utils.exceptions import GoldenMasterValidationError


class TestGoldenMasterValidator:
    """Test suite for golden master validation functionality"""
    
    @pytest.fixture
    def validator(self):
        """Create a GoldenMasterValidator instance for testing"""
        golden_master_path = Path(__file__).parent.parent.parent / "shared" / "data" / "master_ultimate_golden.csv"
        return GoldenMasterValidator(str(golden_master_path))
    
    @pytest.fixture
    def sample_golden_df(self):
        """Create sample golden master data for testing"""
        df = pd.DataFrame({
            'year': [1965, 1966, 1967, 1968, 1969, 1970, 1971, 1972],
            'make': ['Ford', 'Ford', 'Ford', 'Ford', 'Chevrolet', 'Chevrolet', 'Dodge', 'Dodge'],
            'model': ['Mustang', 'Mustang', 'Mustang', 'Mustang', 'Camaro', 'Camaro', 'Challenger', 'Challenger'],
            'car_id': ['001', '002', '003', '004', '005', '006', '007', '008'],
            'body_type': ['Coupe', 'Coupe', 'Fastback', 'Fastback', 'Coupe', 'Coupe', 'Coupe', 'Coupe'],
            'engine': ['289 V8', '289 V8', '302 V8', '302 V8', '350 V8', '350 V8', '340 V8', '340 V8']
        })
        # Add normalized columns that the validator creates
        df['make_normalized'] = df['make'].str.lower()
        df['model_normalized'] = df['model'].str.lower()
        df['body_type_normalized'] = df['body_type'].str.lower()
        return df
    
    def test_load_golden_master_success(self, validator):
        """Test successful loading of golden master data"""
        df = validator.load_golden_master()
        
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert 'year' in df.columns
        assert 'make' in df.columns
        assert 'model' in df.columns
        assert len(df) > 0
    
    def test_load_golden_master_file_not_found(self):
        """Test handling of missing golden master file"""
        validator = GoldenMasterValidator("nonexistent_file.csv")
        
        with pytest.raises(FileNotFoundError):
            validator.load_golden_master()
    
    def test_load_golden_master_empty_file(self, tmp_path):
        """Test handling of empty golden master file"""
        empty_file = tmp_path / "empty_golden.csv"
        empty_file.write_text("")
        
        validator = GoldenMasterValidator(str(empty_file))
        
        with pytest.raises(Exception):
            validator.load_golden_master()
    
    def test_validate_combination_valid(self, validator, sample_golden_df):
        """Test validation of valid year/make/model combinations"""
        # Mock the golden master data
        validator.golden_df = sample_golden_df
        
        # Test valid combinations
        assert validator.validate_combination(1965, 'Ford', 'Mustang') == True
        assert validator.validate_combination(1969, 'Chevrolet', 'Camaro') == True
        assert validator.validate_combination(1971, 'Dodge', 'Challenger') == True
    
    def test_validate_combination_invalid_year(self, validator, sample_golden_df):
        """Test validation of invalid year"""
        validator.golden_df = sample_golden_df
        
        # Test invalid year
        assert validator.validate_combination(1960, 'Ford', 'Mustang') == False
        assert validator.validate_combination(1980, 'Ford', 'Mustang') == False
    
    def test_validate_combination_invalid_make(self, validator, sample_golden_df):
        """Test validation of invalid make"""
        validator.golden_df = sample_golden_df
        
        # Test invalid make
        assert validator.validate_combination(1965, 'Toyota', 'Camry') == False
        assert validator.validate_combination(1965, 'InvalidMake', 'Mustang') == False
    
    def test_validate_combination_invalid_model(self, validator, sample_golden_df):
        """Test validation of invalid model"""
        validator.golden_df = sample_golden_df
        
        # Test invalid model
        assert validator.validate_combination(1965, 'Ford', 'Camaro') == False
        assert validator.validate_combination(1965, 'Ford', 'InvalidModel') == False
    
    def test_validate_combination_case_insensitive(self, validator, sample_golden_df):
        """Test case-insensitive validation"""
        validator.golden_df = sample_golden_df
        
        # Test case variations
        assert validator.validate_combination(1965, 'ford', 'mustang') == True
        assert validator.validate_combination(1965, 'FORD', 'MUSTANG') == True
        assert validator.validate_combination(1965, 'Ford', 'MUSTANG') == True
    
    def test_validate_year_range_valid(self, validator, sample_golden_df):
        """Test validation of valid year ranges"""
        validator.golden_df = sample_golden_df
        
        # Test valid year ranges
        assert validator.validate_year_range(1965, 1970, 'Ford', 'Mustang') is True
        assert validator.validate_year_range(1969, 1970, 'Chevrolet', 'Camaro') is True
    
    def test_validate_year_range_invalid(self, validator, sample_golden_df):
        """Test validation of invalid year ranges"""
        validator.golden_df = sample_golden_df
        
        # Test invalid year ranges
        assert validator.validate_year_range(1960, 1964, 'Ford', 'Mustang') is False
        assert validator.validate_year_range(1975, 1980, 'Ford', 'Mustang') is False
    
    def test_validate_year_range_partial_overlap(self, validator, sample_golden_df):
        """Test validation of partially overlapping year ranges"""
        validator.golden_df = sample_golden_df
        
        # Test partial overlap (some years valid, some not)
        result = validator.validate_year_range(1964, 1966, 'Ford', 'Mustang')
        assert isinstance(result, (bool, dict))  # May return detailed info
    
    def test_get_valid_options_all(self, validator, sample_golden_df):
        """Test getting all valid options"""
        validator.golden_df = sample_golden_df
        
        options = validator.get_valid_options()
        
        assert 'makes' in options
        assert 'models' in options
        assert 'years' in options
        
        assert 'Ford' in options['makes']
        assert 'Chevrolet' in options['makes']
        assert 'Dodge' in options['makes']
        
        assert 'Mustang' in options['models']
        assert 'Camaro' in options['models']
        assert 'Challenger' in options['models']
        
        assert 1965 in options['years']
        assert 1972 in options['years']
    
    def test_get_valid_options_filtered_by_year(self, validator, sample_golden_df):
        """Test getting valid options filtered by year range"""
        validator.golden_df = sample_golden_df
        
        options = validator.get_valid_options(year_range=(1965, 1967))
        
        assert 'makes' in options
        assert 'models' in options
        
        # Should only include makes/models available in 1965-1967
        assert 'Ford' in options['makes']
        # Chevrolet Camaro starts in 1969, so shouldn't be included
        assert 'Chevrolet' not in options['makes']
    
    def test_get_valid_options_filtered_by_make(self, validator, sample_golden_df):
        """Test getting valid options filtered by make"""
        validator.golden_df = sample_golden_df
        
        options = validator.get_valid_options(make='Ford')
        
        assert 'models' in options
        assert 'years' in options
        
        # Should only include Ford models
        assert 'Mustang' in options['models']
        assert 'Camaro' not in options['models']
        assert 'Challenger' not in options['models']
    
    def test_get_valid_models_for_make_and_year(self, validator, sample_golden_df):
        """Test getting valid models for specific make and year"""
        validator.golden_df = sample_golden_df
        
        models = validator.get_valid_models_for_make_and_year('Ford', 1965)
        
        assert isinstance(models, list)
        assert 'Mustang' in models
        assert 'Camaro' not in models
    
    def test_get_valid_car_ids(self, validator, sample_golden_df):
        """Test getting valid car IDs for combinations"""
        validator.golden_df = sample_golden_df
        
        car_ids = validator.get_valid_car_ids(1965, 'Ford', 'Mustang')
        
        assert isinstance(car_ids, list)
        assert len(car_ids) > 0
        assert '001' in car_ids
    
    def test_generate_validation_report(self, validator, sample_golden_df):
        """Test generation of validation report"""
        validator.golden_df = sample_golden_df
        
        # Create test data to validate
        test_data = pd.DataFrame({
            'year_min': [1965, 1969, 1980],
            'year_max': [1970, 1970, 1985],
            'make': ['Ford', 'Chevrolet', 'Toyota'],
            'model': ['Mustang', 'Camaro', 'Camry'],
            'mpn': ['TEST-001', 'TEST-002', 'TEST-003']
        })
        
        report = validator.generate_validation_report(test_data)
        
        assert isinstance(report, dict)
        assert 'total_items' in report
        assert 'valid_items' in report
        assert 'invalid_items' in report
        assert 'validation_rate' in report
        assert 'invalid_details' in report
        
        assert report['total_items'] == 3
        assert report['valid_items'] == 2  # Ford Mustang and Chevrolet Camaro
        assert report['invalid_items'] == 1  # Toyota Camry
        assert report['validation_rate'] == 2/3
    
    def test_suggest_alternatives_for_invalid_combination(self, validator, sample_golden_df):
        """Test suggestion of alternatives for invalid combinations"""
        validator.golden_df = sample_golden_df
        
        suggestions = validator.suggest_alternatives('Ford', 'InvalidModel', 1965)
        
        assert isinstance(suggestions, dict)
        assert 'valid_models_for_make' in suggestions
        assert 'similar_models' in suggestions
        assert 'valid_years_for_make' in suggestions
        
        assert 'Mustang' in suggestions['valid_models_for_make']
    
    def test_validate_body_type_compatibility(self, validator, sample_golden_df):
        """Test body type compatibility validation"""
        validator.golden_df = sample_golden_df
        
        # Test valid body type
        assert validator.validate_body_type(1965, 'Ford', 'Mustang', 'Coupe') is True
        
        # Test invalid body type
        assert validator.validate_body_type(1965, 'Ford', 'Mustang', 'Truck') is False
    
    def test_get_available_body_types(self, validator, sample_golden_df):
        """Test getting available body types for make/model/year"""
        validator.golden_df = sample_golden_df
        
        body_types = validator.get_available_body_types(1965, 'Ford', 'Mustang')
        
        assert isinstance(body_types, list)
        assert 'Coupe' in body_types
    
    def test_validate_universal_compatibility(self, validator, sample_golden_df):
        """Test validation of universal compatibility claims"""
        validator.golden_df = sample_golden_df
        
        # Test universal compatibility
        result = validator.validate_universal_compatibility(1900, 2024, 'Universal', 'All')
        
        assert isinstance(result, dict)
        assert result['is_universal'] is True
        assert result['applicable_count'] > 0
    
    def test_get_validation_statistics(self, validator, sample_golden_df):
        """Test getting validation statistics"""
        validator.golden_df = sample_golden_df
        
        stats = validator.get_validation_statistics()
        
        assert isinstance(stats, dict)
        assert 'total_combinations' in stats
        assert 'unique_makes' in stats
        assert 'unique_models' in stats
        assert 'year_range' in stats
        assert 'most_common_makes' in stats
        assert 'most_common_models' in stats
    
    def test_validate_with_confidence_scoring(self, validator, sample_golden_df):
        """Test validation with confidence scoring"""
        validator.golden_df = sample_golden_df
        
        # Test exact match (high confidence)
        result = validator.validate_with_confidence(1965, 'Ford', 'Mustang')
        assert result['is_valid'] is True
        assert result['confidence'] > 0.9
        
        # Test fuzzy match (lower confidence)
        result = validator.validate_with_confidence(1965, 'ford', 'mustang')
        assert result['is_valid'] is True
        assert result['confidence'] > 0.8
        
        # Test invalid combination (low confidence)
        result = validator.validate_with_confidence(1965, 'Toyota', 'Camry')
        assert result['is_valid'] is False
        assert result['confidence'] < 0.5


class TestGoldenMasterValidationIntegration:
    """Integration tests for golden master validation"""
    
    @pytest.fixture
    def validator(self):
        golden_master_path = Path(__file__).parent.parent.parent / "shared" / "data" / "master_ultimate_golden.csv"
        return GoldenMasterValidator(str(golden_master_path))
    
    def test_real_golden_master_loading(self, validator):
        """Test loading real golden master data"""
        df = validator.load_golden_master()
        
        # Should have substantial data
        assert len(df) > 100000
        assert 'year' in df.columns
        assert 'make' in df.columns
        assert 'model' in df.columns
    
    def test_real_validation_scenarios(self, validator):
        """Test validation with real golden master data"""
        # Load real data
        validator.load_golden_master()
        
        # Test known valid combinations
        assert validator.validate_combination(1965, 'Ford', 'Mustang') is True
        assert validator.validate_combination(1969, 'Chevrolet', 'Camaro') is True
        assert validator.validate_combination(1970, 'Dodge', 'Challenger') is True
        
        # Test known invalid combinations
        assert validator.validate_combination(1960, 'Ford', 'Mustang') is False
        assert validator.validate_combination(1965, 'Toyota', 'Camry') is False
    
    def test_performance_with_large_dataset(self, validator):
        """Test performance with large golden master dataset"""
        import time
        
        # Load real data
        validator.load_golden_master()
        
        # Test validation performance
        start_time = time.time()
        
        test_combinations = [
            (1965, 'Ford', 'Mustang'),
            (1969, 'Chevrolet', 'Camaro'),
            (1970, 'Dodge', 'Challenger'),
            (1975, 'Honda', 'Civic'),
            (1980, 'Toyota', 'Corolla')
        ]
        
        for year, make, model in test_combinations:
            validator.validate_combination(year, make, model)
        
        elapsed_time = time.time() - start_time
        
        # Should be fast enough for real-time validation
        assert elapsed_time < 1.0  # Less than 1 second for 5 validations
    
    def test_memory_efficiency(self, validator):
        """Test memory efficiency with large dataset"""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Load golden master
        validator.load_golden_master()
        
        # Get memory usage after loading
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Memory increase should be reasonable for dataset size
        memory_increase = final_memory - initial_memory
        assert memory_increase < 500  # Less than 500MB increase
    
    def test_batch_validation_performance(self, validator):
        """Test batch validation performance"""
        import time
        
        # Load real data
        validator.load_golden_master()
        
        # Create batch test data
        test_data = pd.DataFrame({
            'year_min': [1965, 1969, 1970, 1975, 1980] * 100,
            'year_max': [1970, 1970, 1972, 1980, 1985] * 100,
            'make': ['Ford', 'Chevrolet', 'Dodge', 'Honda', 'Toyota'] * 100,
            'model': ['Mustang', 'Camaro', 'Challenger', 'Civic', 'Corolla'] * 100,
            'mpn': [f'TEST-{i:03d}' for i in range(500)]
        })
        
        # Test batch validation performance
        start_time = time.time()
        report = validator.generate_validation_report(test_data)
        elapsed_time = time.time() - start_time
        
        # Should process 500 items quickly
        assert elapsed_time < 5.0  # Less than 5 seconds for 500 items
        assert report['total_items'] == 500