import pytest
import pandas as pd
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from utils.data_loader import RawDataLoader, AIFriendlyConverter
from utils.exceptions import DataValidationError


class TestRawDataLoader:
    """Test suite for raw data loading functionality"""
    
    @pytest.fixture
    def loader(self):
        """Create a RawDataLoader instance for testing"""
        return RawDataLoader()
    
    @pytest.fixture
    def sample_data_path(self):
        """Path to sample test data"""
        return Path(__file__).parent.parent / "data" / "raw" / "steele.xlsx"
    
    def test_load_data_success(self, loader, sample_data_path):
        """Test loading raw data successfully"""
        # Test loading Excel file
        df = loader.load_data(str(sample_data_path))
        
        # Verify data loaded correctly
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert df.shape[0] > 0
        assert df.shape[1] > 0
        
        # Check for expected columns (based on Steele data structure)
        expected_columns = ['StockCode', 'Product Name', 'Description', 'StockUom', 'UPC Code', 'MAP', 'Dealer Price']
        for col in expected_columns:
            assert col in df.columns
    
    def test_load_data_file_not_found(self, loader):
        """Test handling of missing files"""
        with pytest.raises(FileNotFoundError):
            loader.load_data("nonexistent_file.xlsx")
    
    def test_load_data_invalid_format(self, loader, tmp_path):
        """Test handling of invalid file formats"""
        # Create a text file with invalid format
        invalid_file = tmp_path / "invalid.txt"
        invalid_file.write_text("not a valid data file")
        
        with pytest.raises(Exception):
            loader.load_data(str(invalid_file))
    
    def test_validate_structure_valid_data(self, loader, sample_data_path):
        """Test structure validation for valid data"""
        df = loader.load_data(str(sample_data_path))
        validation_result = loader.validate_structure(df)
        
        assert validation_result['is_valid'] is True
        assert 'missing_columns' in validation_result
        assert 'data_types' in validation_result
        assert 'column_count' in validation_result
        assert validation_result['column_count'] > 0
    
    def test_validate_structure_missing_columns(self, loader):
        """Test structure validation with missing required columns"""
        # Create DataFrame with missing columns
        df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        validation_result = loader.validate_structure(df)
        
        assert validation_result['is_valid'] is False
        assert len(validation_result['missing_columns']) > 0
    
    def test_validate_structure_empty_dataframe(self, loader):
        """Test structure validation with empty DataFrame"""
        df = pd.DataFrame()
        validation_result = loader.validate_structure(df)
        
        assert validation_result['is_valid'] is False
        assert validation_result['column_count'] == 0
    
    def test_generate_quality_report(self, loader, sample_data_path):
        """Test quality report generation"""
        df = loader.load_data(str(sample_data_path))
        quality_report = loader.generate_quality_report(df)
        
        # Check report structure
        assert 'total_rows' in quality_report
        assert 'total_columns' in quality_report
        assert 'completeness' in quality_report
        assert 'duplicates' in quality_report
        assert 'data_types' in quality_report
        
        # Check values
        assert quality_report['total_rows'] > 0
        assert quality_report['total_columns'] > 0
        assert isinstance(quality_report['completeness'], dict)
        assert isinstance(quality_report['duplicates'], dict)
    
    def test_generate_quality_report_duplicates(self, loader):
        """Test quality report with duplicate detection"""
        # Create DataFrame with duplicates
        df = pd.DataFrame({
            'StockCode': ['A001', 'A001', 'A002'],
            'Product Name': ['Product 1', 'Product 1', 'Product 2'],
            'Description': ['Desc 1', 'Desc 1', 'Desc 2']
        })
        
        quality_report = loader.generate_quality_report(df)
        
        assert quality_report['duplicates']['duplicate_count'] > 0
        assert quality_report['duplicates']['duplicate_percentage'] > 0
    
    def test_generate_quality_report_missing_data(self, loader):
        """Test quality report with missing data"""
        # Create DataFrame with missing values
        df = pd.DataFrame({
            'StockCode': ['A001', 'A002', None],
            'Product Name': ['Product 1', None, 'Product 3'],
            'Description': ['Desc 1', 'Desc 2', 'Desc 3']
        })
        
        quality_report = loader.generate_quality_report(df)
        
        # Check completeness metrics
        assert 'StockCode' in quality_report['completeness']
        assert 'Product Name' in quality_report['completeness']
        assert quality_report['completeness']['StockCode'] < 100.0
        assert quality_report['completeness']['Product Name'] < 100.0


class TestAIFriendlyConverter:
    """Test suite for AI-friendly format conversion"""
    
    @pytest.fixture
    def converter(self):
        """Create an AIFriendlyConverter instance for testing"""
        return AIFriendlyConverter()
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame for testing"""
        return pd.DataFrame({
            'StockCode': ['10-0001-40', '10-0002-35', '10-0003-35'],
            'Product Name': ['Accelerator Pedal Pad', 'Axle Rebound Pad', 'Axle Rebound Pad Service'],
            'Description': ['For 1965-1970 Ford Mustang', 'Universal fit for all vehicles', '1934/64 Chevrolet 2-Door'],
            'StockUom': ['EA', 'EA', 'EA'],
            'UPC Code': ['123456789', '987654321', '456789123'],
            'MAP': [75.49, 127.79, 132.49],
            'Dealer Price': [43.76, 81.97, 84.50]
        })
    
    def test_convert_to_ai_format(self, converter, sample_dataframe):
        """Test conversion to AI-friendly format"""
        ai_df = converter.convert_to_ai_format(sample_dataframe)
        
        # Check that conversion was successful
        assert isinstance(ai_df, pd.DataFrame)
        assert not ai_df.empty
        assert 'product_info' in ai_df.columns
        assert 'stock_code' in ai_df.columns
        assert 'cost' in ai_df.columns
        assert 'price' in ai_df.columns
        
        # Check that product_info contains relevant information
        for _, row in ai_df.iterrows():
            assert row['stock_code'] in row['product_info']
            assert len(row['product_info']) > 0
    
    def test_create_product_info_string(self, converter):
        """Test product info string creation"""
        sample_row = pd.Series({
            'StockCode': '10-0001-40',
            'Product Name': 'Accelerator Pedal Pad',
            'Description': 'For 1965-1970 Ford Mustang',
            'StockUom': 'EA',
            'UPC Code': '123456789'
        })
        
        product_info = converter.create_product_info_string(sample_row)
        
        # Check that all relevant information is included
        assert '10-0001-40' in product_info
        assert 'Accelerator Pedal Pad' in product_info
        assert 'For 1965-1970 Ford Mustang' in product_info
        assert isinstance(product_info, str)
        assert len(product_info) > 0
    
    def test_create_product_info_string_missing_fields(self, converter):
        """Test product info string creation with missing fields"""
        sample_row = pd.Series({
            'StockCode': '10-0001-40',
            'Product Name': 'Accelerator Pedal Pad',
            'Description': None,  # Missing description
            'StockUom': 'EA'
        })
        
        product_info = converter.create_product_info_string(sample_row)
        
        # Should handle missing fields gracefully
        assert '10-0001-40' in product_info
        assert 'Accelerator Pedal Pad' in product_info
        assert isinstance(product_info, str)
        assert len(product_info) > 0
    
    def test_estimate_tokens(self, converter):
        """Test token estimation functionality"""
        test_text = "This is a test string for token estimation"
        token_count = converter.estimate_tokens(test_text)
        
        assert isinstance(token_count, int)
        assert token_count > 0
        assert token_count < 100  # Should be reasonable for short text
    
    def test_estimate_tokens_empty_string(self, converter):
        """Test token estimation with empty string"""
        token_count = converter.estimate_tokens("")
        assert token_count == 0
    
    def test_estimate_tokens_long_text(self, converter):
        """Test token estimation with long text"""
        long_text = "This is a very long text " * 100
        token_count = converter.estimate_tokens(long_text)
        
        assert isinstance(token_count, int)
        assert token_count > 100  # Should be substantial for long text
    
    def test_estimate_batch_cost(self, converter, sample_dataframe):
        """Test batch cost estimation"""
        ai_df = converter.convert_to_ai_format(sample_dataframe)
        cost_estimate = converter.estimate_batch_cost(ai_df)
        
        # Check cost estimate structure
        assert isinstance(cost_estimate, dict)
        assert 'total_input_tokens' in cost_estimate
        assert 'estimated_output_tokens' in cost_estimate
        assert 'total_cost' in cost_estimate
        assert 'cost_per_item' in cost_estimate
        
        # Check values
        assert cost_estimate['total_input_tokens'] > 0
        assert cost_estimate['estimated_output_tokens'] > 0
        assert cost_estimate['total_cost'] > 0
        assert cost_estimate['cost_per_item'] > 0
    
    def test_estimate_batch_cost_empty_dataframe(self, converter):
        """Test batch cost estimation with empty DataFrame"""
        empty_df = pd.DataFrame()
        cost_estimate = converter.estimate_batch_cost(empty_df)
        
        assert cost_estimate['total_input_tokens'] == 0
        assert cost_estimate['estimated_output_tokens'] == 0
        assert cost_estimate['total_cost'] == 0
        assert cost_estimate['cost_per_item'] == 0
    
    def test_estimate_batch_cost_various_models(self, converter, sample_dataframe):
        """Test batch cost estimation with different AI models"""
        ai_df = converter.convert_to_ai_format(sample_dataframe)
        
        # Test with different models
        models = ['gpt-4.1-mini', 'gpt-4o']
        for model in models:
            cost_estimate = converter.estimate_batch_cost(ai_df, model=model)
            
            assert isinstance(cost_estimate, dict)
            assert cost_estimate['total_cost'] > 0
            assert 'model' in cost_estimate
            assert cost_estimate['model'] == model


class TestDataLoaderIntegration:
    """Integration tests for data loader and converter"""
    
    @pytest.fixture
    def loader(self):
        return RawDataLoader()
    
    @pytest.fixture
    def converter(self):
        return AIFriendlyConverter()
    
    @pytest.fixture
    def sample_data_path(self):
        return Path(__file__).parent.parent / "data" / "raw" / "steele.xlsx"
    
    def test_full_conversion_pipeline(self, loader, converter, sample_data_path):
        """Test complete data loading and conversion pipeline"""
        # Load raw data
        df = loader.load_data(str(sample_data_path))
        
        # Validate structure
        validation_result = loader.validate_structure(df)
        assert validation_result['is_valid'] is True
        
        # Generate quality report
        quality_report = loader.generate_quality_report(df)
        assert quality_report['total_rows'] > 0
        
        # Convert to AI-friendly format
        ai_df = converter.convert_to_ai_format(df)
        assert not ai_df.empty
        assert 'product_info' in ai_df.columns
        
        # Estimate processing cost
        cost_estimate = converter.estimate_batch_cost(ai_df)
        assert cost_estimate['total_cost'] > 0
    
    def test_error_handling_pipeline(self, loader, converter):
        """Test error handling in the complete pipeline"""
        # Test with invalid file
        with pytest.raises(FileNotFoundError):
            loader.load_data("nonexistent_file.xlsx")
        
        # Test with invalid DataFrame
        invalid_df = pd.DataFrame()
        validation_result = loader.validate_structure(invalid_df)
        assert validation_result['is_valid'] is False
        
        # Test conversion with empty DataFrame
        ai_df = converter.convert_to_ai_format(invalid_df)
        assert ai_df.empty