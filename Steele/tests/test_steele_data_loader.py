import pandas as pd
import pytest
import os
from pathlib import Path

class TestSteeleDataLoader:
    """Test suite for Steele data loading and validation"""
    
    @pytest.fixture
    def sample_data_path(self):
        """Path to Steele sample data"""
        return "data/samples/steele_sample.csv"
    
    @pytest.fixture
    def sample_df(self, sample_data_path):
        """Load sample dataframe for testing"""
        return pd.read_csv(sample_data_path)
    
    def test_sample_data_file_exists(self, sample_data_path):
        """Test that sample data file exists"""
        assert os.path.exists(sample_data_path), f"Sample data file not found: {sample_data_path}"
    
    def test_sample_data_not_empty(self, sample_df):
        """Test that sample data is not empty"""
        assert len(sample_df) > 0, "Sample data should not be empty"
        assert len(sample_df.columns) > 0, "Sample data should have columns"
    
    def test_expected_columns_present(self, sample_df):
        """Test that all expected Steele columns are present"""
        expected_columns = [
            'StockCode', 'Product Name', 'Description', 'StockUom',
            'UPC Code', 'MAP', 'Dealer Price', 'PartNumber',
            'Year', 'Make', 'Model', 'Submodel', 'Type', 'Doors', 'BodyType'
        ]
        
        missing_columns = set(expected_columns) - set(sample_df.columns)
        assert not missing_columns, f"Missing expected columns: {missing_columns}"
    
    def test_critical_columns_not_null(self, sample_df):
        """Test that critical columns don't have all null values"""
        critical_columns = ['StockCode', 'Product Name', 'Year', 'Make', 'Model']
        
        for col in critical_columns:
            assert not sample_df[col].isnull().all(), f"Critical column {col} is all null"
    
    def test_price_columns_data_types(self, sample_df):
        """Test that price columns are numeric"""
        price_columns = ['MAP', 'Dealer Price']
        
        for col in price_columns:
            assert pd.api.types.is_numeric_dtype(sample_df[col]), f"Price column {col} should be numeric"
    
    def test_price_values_positive(self, sample_df):
        """Test that price values are positive"""
        price_columns = ['MAP', 'Dealer Price']
        
        for col in price_columns:
            non_null_prices = sample_df[col].dropna()
            if len(non_null_prices) > 0:
                assert (non_null_prices > 0).all(), f"All {col} values should be positive"
    
    def test_year_values_valid_range(self, sample_df):
        """Test that year values are in valid range"""
        years = sample_df['Year'].dropna()
        if len(years) > 0:
            assert years.between(1900, 2030).all(), "Year values should be between 1900 and 2030"
    
    def test_make_model_not_empty(self, sample_df):
        """Test that Make and Model are not empty strings"""
        non_null_makes = sample_df['Make'].dropna()
        non_null_models = sample_df['Model'].dropna()
        
        if len(non_null_makes) > 0:
            assert not (non_null_makes == '').any(), "Make should not be empty string"
        
        if len(non_null_models) > 0:
            assert not (non_null_models == '').any(), "Model should not be empty string"
    
    def test_stock_code_uniqueness_per_vehicle(self, sample_df):
        """Test that StockCode uniqueness is maintained properly"""
        # Group by vehicle details and check if stock codes are consistent
        grouped = sample_df.groupby(['StockCode', 'Product Name'])
        
        for (stock_code, product_name), group in grouped:
            # Same stock code should have same product name
            assert len(group['Product Name'].unique()) == 1, f"StockCode {stock_code} has multiple product names"
    
    def test_upc_code_format(self, sample_df):
        """Test UPC code format if present"""
        upc_codes = sample_df['UPC Code'].dropna()
        
        for upc in upc_codes:
            if upc and str(upc) != 'nan':
                # UPC should be numeric and appropriate length
                upc_str = str(upc).replace('.0', '')  # Remove decimal if present
                assert upc_str.isdigit(), f"UPC Code {upc} should be numeric"
                assert len(upc_str) in [12, 13, 14], f"UPC Code {upc} should be 12-14 digits"
    
    def test_description_length(self, sample_df):
        """Test that descriptions are reasonable length"""
        descriptions = sample_df['Description'].dropna()
        
        for desc in descriptions:
            if desc and str(desc) != 'nan':
                assert len(str(desc)) > 10, f"Description too short: {desc}"
                assert len(str(desc)) < 1000, f"Description too long: {desc}"

    @pytest.mark.performance
    def test_data_loading_performance(self, sample_data_path):
        """Test that data loading performance is acceptable"""
        import time
        
        start_time = time.time()
        df = pd.read_csv(sample_data_path)
        load_time = time.time() - start_time
        
        # Should load sample data in under 1 second
        assert load_time < 1.0, f"Data loading took too long: {load_time} seconds"
        assert len(df) > 0, "No data loaded" 