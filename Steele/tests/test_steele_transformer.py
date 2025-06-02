import pandas as pd
import pytest
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

# Add the utils directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.steele_data_transformer import SteeleDataTransformer, ProductData

class TestSteeleDataTransformer:
    """Test suite for Steele data transformer following new workflow"""
    
    @pytest.fixture
    def transformer(self):
        """Basic transformer instance without AI"""
        return SteeleDataTransformer(use_ai=False)
    
    @pytest.fixture
    def sample_steele_data(self):
        """Sample Steele data for testing"""
        return pd.DataFrame({
            'StockCode': ['10-0001-40', '10-0002-35', '10-0003-35'],
            'Product Name': ['Accelerator Pedal Pad', 'Axle Rebound Pad', 'Fuel Tank Cover'],
            'Description': ['Pad, accelerator pedal', 'Pad, front axle rebound', 'Cover, fuel tank'],
            'StockUom': ['ea.', 'ea.', 'ea.'],
            'UPC Code': [706072000022, 706072000023, 706072000024],
            'MAP': [75.49, 127.79, 45.99],
            'Dealer Price': [43.76, 81.97, 25.49],
            'PartNumber': ['10-0001-40', '10-0002-35', '10-0003-35'],
            'Year': [1928, 1930, 1932],
            'Make': ['Stutz', 'Stutz', 'Ford'],
            'Model': ['Stutz', 'Stutz', 'Model A'],
            'Submodel': ['Base', 'Base', 'Base'],
            'Type': ['Car', 'Car', 'Car'],
            'Doors': [0.0, 0.0, 4.0],
            'BodyType': ['U/K', 'U/K', 'Sedan']
        })
    
    @pytest.fixture
    def sample_validation_df(self):
        """Sample validation DataFrame"""
        return pd.DataFrame({
            'steele_row_index': [0, 1, 2],
            'stock_code': ['10-0001-40', '10-0002-35', '10-0003-35'],
            'golden_validated': [True, True, False],
            'golden_matches': [2, 1, 0]
        })

    def test_transformer_initialization(self):
        """Test basic transformer initialization"""
        transformer = SteeleDataTransformer(use_ai=False)
        assert transformer.vendor_name == "Steele"
        assert transformer.use_ai == False

    def test_transformer_initialization_with_ai(self):
        """Test transformer initializes with AI when API key available"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            with patch('utils.steele_data_transformer.OpenAI') as mock_openai:
                transformer = SteeleDataTransformer(use_ai=True)
                assert transformer.use_ai == True
                mock_openai.assert_called_once()

    def test_load_sample_data_success(self, transformer, sample_steele_data):
        """Test successful sample data loading"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            sample_steele_data.to_csv(temp_file.name, index=False)
            
            try:
                loaded_df = transformer.load_sample_data(temp_file.name)
                assert len(loaded_df) == 3
                assert 'StockCode' in loaded_df.columns
                assert 'Product Name' in loaded_df.columns
            finally:
                os.unlink(temp_file.name)

    def test_load_sample_data_file_not_found(self, transformer):
        """Test handling of missing sample file"""
        with pytest.raises(FileNotFoundError):
            transformer.load_sample_data("nonexistent_file.csv")

    def test_validate_input_data_success(self, transformer, sample_steele_data):
        """Test validation with valid input data"""
        # Should not raise exception
        transformer._validate_input_data(sample_steele_data)

    def test_validate_input_data_missing_columns(self, transformer):
        """Test validation with missing required columns"""
        incomplete_data = pd.DataFrame({
            'StockCode': ['10-0001-40'],
            'Product Name': ['Test Product']
            # Missing other required columns
        })
        
        with pytest.raises(ValueError, match="Missing required columns"):
            transformer._validate_input_data(incomplete_data)

    def test_validate_input_data_empty_dataframe(self, transformer):
        """Test validation with empty DataFrame"""
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="Input data is empty"):
            transformer._validate_input_data(empty_df)

    def test_validate_input_data_null_critical_fields(self, transformer, sample_steele_data):
        """Test validation with null critical fields"""
        null_data = sample_steele_data.copy()
        null_data['StockCode'] = None
        
        with pytest.raises(ValueError, match="Critical field 'StockCode' is completely empty"):
            transformer._validate_input_data(null_data)

    def test_validate_against_golden_dataset(self, transformer, sample_steele_data):
        """Test golden dataset validation"""
        # Mock golden dataset
        mock_golden = pd.DataFrame({
            'year': [1928, 1930, 1965],
            'make': ['Stutz', 'Stutz', 'Ford'],
            'model': ['Stutz', 'Stutz', 'Mustang'],
            'car_id': ['1928_Stutz_Stutz', '1930_Stutz_Stutz', '1965_Ford_Mustang']
        })
        transformer.golden_df = mock_golden
        
        validation_df = transformer.validate_against_golden_dataset(sample_steele_data)
        
        assert len(validation_df) == 3
        assert 'golden_validated' in validation_df.columns
        assert validation_df.iloc[0]['golden_validated'] == True  # 1928 Stutz should match
        assert validation_df.iloc[1]['golden_validated'] == True  # 1930 Stutz should match

    def test_transform_to_ai_friendly_format(self, transformer, sample_steele_data, sample_validation_df):
        """Test transformation to AI-friendly format"""
        ai_friendly_products = transformer.transform_to_ai_friendly_format(sample_steele_data, sample_validation_df)
        
        assert len(ai_friendly_products) == 3
        assert all(isinstance(product, ProductData) for product in ai_friendly_products)
        
        # Check first product
        first_product = ai_friendly_products[0]
        assert first_product.title == "Accelerator Pedal Pad"
        assert first_product.year_min == "1928"
        assert first_product.make == "Stutz"  # Should be populated since golden_validated=True
        assert first_product.price == 75.49

    def test_transform_to_final_tagged_format(self, transformer):
        """Test transformation to final Shopify format"""
        # Create sample ProductData
        sample_products = [
            ProductData(
                title="Test Product",
                year_min="1928",
                year_max="1928",
                make="Stutz",
                model="Stutz",
                mpn="10-0001-40",
                cost=43.76,
                price=75.49,
                body_html="Test description",
                meta_title="Test Meta Title",
                meta_description="Test meta description"
            )
        ]
        
        final_df = transformer.transform_to_final_tagged_format(sample_products)
        
        assert len(final_df) == 1
        assert 'Title' in final_df.columns
        assert 'Vendor' in final_df.columns
        assert 'Tags' in final_df.columns
        assert final_df.iloc[0]['Title'] == "Test Product"
        assert final_df.iloc[0]['Vendor'] == "Steele"
        assert final_df.iloc[0]['Tags'] == "1928_Stutz_Stutz"

    def test_generate_vehicle_tag(self, transformer):
        """Test vehicle tag generation"""
        product_data = ProductData(
            title="Test Product",
            year_min="1928",
            make="Stutz",
            model="Stutz"
        )
        
        tag = transformer._generate_vehicle_tag(product_data)
        assert tag == "1928_Stutz_Stutz"

    def test_generate_vehicle_tag_with_missing_data(self, transformer):
        """Test vehicle tag generation with missing data"""
        product_data = ProductData(
            title="Test Product",
            make="NONE",  # Missing make
            model="Model"
        )
        
        tag = transformer._generate_vehicle_tag(product_data)
        assert tag == ""  # Should return empty string for incomplete data

    def test_generate_basic_meta_title(self, transformer):
        """Test basic meta title generation"""
        product_data = ProductData(
            title="Accelerator Pedal Pad",
            year_min="1928",
            make="Stutz",
            model="Stutz"
        )
        
        meta_title = transformer._generate_basic_meta_title(product_data)
        assert "Accelerator Pedal Pad" in meta_title
        assert "1928 Stutz" in meta_title
        assert len(meta_title) <= 60

    def test_generate_basic_meta_description(self, transformer):
        """Test basic meta description generation"""
        product_data = ProductData(
            title="Accelerator Pedal Pad",
            year_min="1928",
            make="Stutz",
            model="Stutz"
        )
        
        meta_description = transformer._generate_basic_meta_description(product_data)
        assert "Accelerator Pedal Pad" in meta_description
        assert "1928 Stutz" in meta_description
        assert len(meta_description) <= 160

    def test_process_complete_pipeline(self, transformer, sample_steele_data):
        """Test complete pipeline processing"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            sample_steele_data.to_csv(temp_file.name, index=False)
            
            try:
                # Mock golden dataset
                mock_golden = pd.DataFrame({
                    'year': [1928, 1930, 1932],
                    'make': ['Stutz', 'Stutz', 'Ford'],
                    'model': ['Stutz', 'Stutz', 'Model A'],
                    'car_id': ['1928_Stutz_Stutz', '1930_Stutz_Stutz', '1932_Ford_Model A']
                })
                transformer.golden_df = mock_golden
                
                final_df = transformer.process_complete_pipeline(temp_file.name)
                
                assert len(final_df) == 3
                assert 'Title' in final_df.columns
                assert 'Vendor' in final_df.columns
                assert 'Tags' in final_df.columns
                assert 'Variant Price' in final_df.columns
                assert 'Variant Cost' in final_df.columns
                
            finally:
                os.unlink(temp_file.name)

    def test_validate_output_success(self, transformer):
        """Test output validation for valid transformed data"""
        valid_df = pd.DataFrame({
            'Title': ['Test Product'],
            'Body HTML': ['Test description'],
            'Vendor': ['Steele'],
            'Tags': ['1928_Stutz_Stutz'],
            'Variant Price': [75.49],
            'Variant Cost': [43.76],
            'Metafield: title_tag [string]': ['Test Meta Title'],
            'Metafield: description_tag [string]': ['Test meta description']
        })
        
        validation_results = transformer.validate_output(valid_df)
        
        assert len(validation_results['errors']) == 0
        assert 'Processed 1 products' in validation_results['info']

    def test_validate_output_missing_columns(self, transformer):
        """Test output validation with missing required columns"""
        incomplete_df = pd.DataFrame({
            'Title': ['Test Product']
            # Missing other required columns
        })
        
        validation_results = transformer.validate_output(incomplete_df)
        
        assert len(validation_results['errors']) > 0
        assert any('Missing required columns' in error for error in validation_results['errors'])

    def test_validate_output_invalid_prices(self, transformer):
        """Test output validation with invalid prices"""
        invalid_price_df = pd.DataFrame({
            'Title': ['Test Product'],
            'Body HTML': ['Test description'],
            'Vendor': ['Steele'],
            'Tags': ['1928_Stutz_Stutz'],
            'Variant Price': [0],  # Invalid price
            'Variant Cost': [43.76],
            'Metafield: title_tag [string]': ['Test Meta Title'],
            'Metafield: description_tag [string]': ['Test meta description']
        })
        
        validation_results = transformer.validate_output(invalid_price_df)
        
        assert len(validation_results['warnings']) > 0
        assert any('invalid prices' in warning for warning in validation_results['warnings'])

    def test_save_output(self, transformer):
        """Test saving transformed output to file"""
        test_df = pd.DataFrame({
            'Title': ['Test Product'],
            'Vendor': ['Steele']
        })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, 'test_output.csv')
            saved_path = transformer.save_output(test_df, output_path)
            
            assert os.path.exists(saved_path)
            loaded_df = pd.read_csv(saved_path)
            assert len(loaded_df) == 1
            assert loaded_df.iloc[0]['Title'] == 'Test Product'

    def test_save_output_creates_directory(self, transformer):
        """Test that save_output creates output directory if it doesn't exist"""
        test_df = pd.DataFrame({
            'Title': ['Test Product'],
            'Vendor': ['Steele']
        })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_output_path = os.path.join(temp_dir, 'nested', 'dir', 'test_output.csv')
            saved_path = transformer.save_output(test_df, nested_output_path)
            
            assert os.path.exists(saved_path)
            assert os.path.exists(os.path.dirname(saved_path))

    @pytest.mark.integration
    def test_complete_pipeline_integration(self, transformer, sample_steele_data):
        """Test complete pipeline integration"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as input_file:
            sample_steele_data.to_csv(input_file.name, index=False)

            try:
                # Mock golden dataset for consistent results
                mock_golden = pd.DataFrame({
                    'year': [1928, 1930, 1932],
                    'make': ['Stutz', 'Stutz', 'Ford'],
                    'model': ['Stutz', 'Stutz', 'Model A'],
                    'car_id': ['1928_Stutz_Stutz', '1930_Stutz_Stutz', '1932_Ford_Model A']
                })
                transformer.golden_df = mock_golden

                # Test complete pipeline
                final_df = transformer.process_complete_pipeline(input_file.name)
                
                # Validate results
                validation_results = transformer.validate_output(final_df)
                assert len(validation_results['errors']) == 0
                
                # Test saving
                with tempfile.TemporaryDirectory() as temp_dir:
                    output_path = os.path.join(temp_dir, 'pipeline_output.csv')
                    saved_path = transformer.save_output(final_df, output_path)
                    assert os.path.exists(saved_path)

            finally:
                os.unlink(input_file.name)

    @pytest.mark.performance
    def test_transformation_performance(self, transformer):
        """Test transformation performance with larger dataset"""
        import time

        # Create larger sample dataset
        large_sample = pd.concat([pd.DataFrame({
            'StockCode': [f'10-000{i}-40'],
            'Product Name': [f'Test Product {i}'],
            'Description': [f'Description for product {i}'],
            'StockUom': ['ea.'],
            'UPC Code': [706072000022 + i],
            'MAP': [75.49 + i],
            'Dealer Price': [43.76 + i],
            'PartNumber': [f'10-000{i}-40'],
            'Year': [1920 + (i % 50)],
            'Make': [f'Make{i % 10}'],
            'Model': [f'Model{i % 20}'],
            'Submodel': ['Base'],
            'Type': ['Car'],
            'Doors': [4.0],
            'BodyType': ['Sedan']
        }) for i in range(100)], ignore_index=True)

        # Mock golden dataset
        mock_golden = pd.DataFrame({
            'year': list(range(1920, 1970)),
            'make': [f'Make{i % 10}' for i in range(50)],
            'model': [f'Model{i % 20}' for i in range(50)],
            'car_id': [f'{1920+i}_Make{i%10}_Model{i%20}' for i in range(50)]
        })
        transformer.golden_df = mock_golden

        start_time = time.time()
        
        # Test AI-friendly transformation
        validation_df = transformer.validate_against_golden_dataset(large_sample)
        ai_friendly_products = transformer.transform_to_ai_friendly_format(large_sample, validation_df)
        final_df = transformer.transform_to_final_tagged_format(ai_friendly_products)
        
        processing_time = time.time() - start_time

        assert len(final_df) == 100
        assert processing_time < 5.0  # Should complete within 5 seconds

    def test_handle_missing_optional_fields(self, transformer):
        """Test transformation handles missing optional fields gracefully"""
        minimal_data = pd.DataFrame({
            'StockCode': ['10-0001-40'],
            'Product Name': ['Test Product'],
            'Description': ['Test Description'],
            'MAP': [75.49],
            'Dealer Price': [43.76],
            'PartNumber': ['10-0001-40'],
            'Year': [1930],
            'Make': ['Stutz'],
            'Model': ['Stutz']
            # Missing UPC Code and other optional fields
        })

        # Mock golden dataset
        mock_golden = pd.DataFrame({
            'year': [1930],
            'make': ['Stutz'],
            'model': ['Stutz'],
            'car_id': ['1930_Stutz_Stutz']
        })
        transformer.golden_df = mock_golden

        # Should not raise exception
        validation_df = transformer.validate_against_golden_dataset(minimal_data)
        ai_friendly_products = transformer.transform_to_ai_friendly_format(minimal_data, validation_df)
        final_df = transformer.transform_to_final_tagged_format(ai_friendly_products)
        
        assert len(final_df) == 1
        assert final_df.iloc[0]['Title'] == 'Test Product' 