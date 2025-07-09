import pandas as pd
import pytest
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

# Add the utils directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.steele_data_transformer import SteeleDataTransformer, ProductData, TemplateGenerator

class TestSteeleDataTransformer:
    """Test suite for Steele data transformer following @completed-data.mdc (NO AI)"""
    
    @pytest.fixture
    def transformer(self):
        """Basic transformer instance with NO AI (following @completed-data.mdc)"""
        return SteeleDataTransformer(use_ai=False)
    
    @pytest.fixture
    def template_generator(self):
        """Template generator for testing template-based enhancement"""
        return TemplateGenerator()
    
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

    def test_transformer_initialization_no_ai(self):
        """Test transformer initialization with NO AI (following @completed-data.mdc)"""
        transformer = SteeleDataTransformer(use_ai=False)
        assert transformer.vendor_name == "Steele"
        assert transformer.use_ai == False
        assert isinstance(transformer.template_generator, TemplateGenerator)

    def test_transformer_warns_about_ai_usage(self, capsys):
        """Test transformer warns when AI is requested for complete fitment data"""
        transformer = SteeleDataTransformer(use_ai=True)
        captured = capsys.readouterr()
        assert "WARNING: AI usage disabled for complete fitment data" in captured.out
        assert transformer.use_ai == False  # Should be forced to False

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
        """Test golden dataset validation (ONLY CRITICAL STEP)"""
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

    def test_transform_to_standard_format(self, transformer, sample_steele_data, sample_validation_df):
        """Test transformation to standard format (preserving existing fitment)"""
        standard_products = transformer.transform_to_standard_format(sample_steele_data, sample_validation_df)
        
        assert len(standard_products) == 3
        assert all(isinstance(product, ProductData) for product in standard_products)
        
        # Check first product
        first_product = standard_products[0]
        assert first_product.title == "Accelerator Pedal Pad"
        assert first_product.year_min == "1928"
        assert first_product.make == "Stutz"  # Should preserve existing fitment
        assert first_product.price == 75.49
        assert first_product.golden_validated == True
        assert first_product.fitment_source == "vendor_provided"
        assert first_product.processing_method == "template_based"

    def test_enhance_with_templates(self, transformer):
        """Test template-based enhancement (NO AI)"""
        sample_products = [
            ProductData(
                title="Accelerator Pedal Pad",
                year_min="1928",
                year_max="1928",
                make="Stutz",
                model="Stutz",
                golden_validated=True,
                processing_method="template_based"
            )
        ]
        
        enhanced_products = transformer.enhance_with_templates(sample_products)
        
        assert len(enhanced_products) == 1
        product = enhanced_products[0]
        assert product.meta_title == "Accelerator Pedal Pad - 1928 Stutz Stutz"
        assert "Quality Accelerator Pedal Pad for 1928 Stutz Stutz vehicles" in product.meta_description
        assert product.collection == "Brakes"  # "pad" keyword triggers brake categorization

    def test_template_generator_meta_title(self, template_generator):
        """Test template-based meta title generation"""
        title = template_generator.generate_meta_title("Brake Pad", "1965", "Ford", "Mustang")
        assert title == "Brake Pad - 1965 Ford Mustang"

    def test_template_generator_meta_description(self, template_generator):
        """Test template-based meta description generation"""
        desc = template_generator.generate_meta_description("Brake Pad", "1965", "Ford", "Mustang")
        assert desc == "Quality Brake Pad for 1965 Ford Mustang vehicles. OEM replacement part."

    def test_template_generator_categorization(self, template_generator):
        """Test rule-based product categorization"""
        assert template_generator.categorize_product("Brake Pad") == "Brakes"
        assert template_generator.categorize_product("Engine Mount") == "Engine"
        assert template_generator.categorize_product("Headlight") == "Lighting"
        assert template_generator.categorize_product("Unknown Part") == "Accessories"

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
                meta_description="Test meta description",
                collection="Engine",
                product_type="Automotive Part"
            )
        ]
        
        final_df = transformer.transform_to_final_tagged_format(sample_products)
        
        assert len(final_df) == 1
        assert 'Title' in final_df.columns
        assert 'Vendor' in final_df.columns
        assert 'Tags' in final_df.columns
        assert 'Collection' in final_df.columns
        assert final_df.iloc[0]['Title'] == "Test Product"
        assert final_df.iloc[0]['Vendor'] == "Steele"
        assert final_df.iloc[0]['Tags'] == "1928_Stutz_Stutz"
        assert final_df.iloc[0]['Collection'] == "Engine"

    def test_generate_vehicle_tag(self, transformer):
        """Test vehicle tag generation from existing fitment data"""
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

    @pytest.mark.integration  
    def test_complete_no_ai_pipeline(self, transformer, sample_steele_data):
        """Test complete NO-AI pipeline following @completed-data.mdc"""
        # Mock golden dataset
        mock_golden = pd.DataFrame({
            'year': [1928, 1930, 1932],
            'make': ['Stutz', 'Stutz', 'Ford'],
            'model': ['Stutz', 'Stutz', 'Model A'],
            'car_id': ['1928_Stutz_Stutz', '1930_Stutz_Stutz', '1932_Ford_Model_A']
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as sample_file:
            sample_steele_data.to_csv(sample_file.name, index=False)
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as golden_file:
                mock_golden.to_csv(golden_file.name, index=False)
                
                try:
                    # Mock the load_golden_dataset method
                    transformer.golden_df = mock_golden
                    
                    final_df = transformer.process_complete_pipeline_no_ai(sample_file.name)
                    
                    assert len(final_df) == 3
                    assert 'Title' in final_df.columns
                    assert 'Tags' in final_df.columns
                    assert 'Collection' in final_df.columns
                    
                    # Check that vehicle tags are generated
                    tags = final_df['Tags'].tolist()
                    assert '1928_Stutz_Stutz' in tags
                    assert '1930_Stutz_Stutz' in tags
                    
                finally:
                    os.unlink(sample_file.name)
                    os.unlink(golden_file.name)

    @pytest.mark.performance
    def test_template_processing_performance(self, transformer):
        """Test that template processing is ultra-fast (>1000 products/sec)"""
        import time
        
        # Create large dataset for performance testing
        large_dataset = []
        for i in range(1000):
            large_dataset.append(ProductData(
                title=f"Product {i}",
                year_min="1965",
                make="Ford",
                model="Mustang",
                golden_validated=True
            ))
        
        start_time = time.time()
        enhanced_products = transformer.enhance_with_templates(large_dataset)
        end_time = time.time()
        
        processing_time = end_time - start_time
        products_per_second = len(enhanced_products) / processing_time
        
        assert products_per_second > 1000, f"Processing too slow: {products_per_second:.2f} products/sec"
        assert len(enhanced_products) == 1000 