import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch
from utils.shopify_format import ShopifyFormatGenerator, SEOContentGenerator
from utils.ai_extraction import ProductData
from utils.exceptions import ShopifyFormatError


class TestShopifyFormatGenerator:
    """Test suite for Shopify format generation functionality"""
    
    @pytest.fixture
    def generator(self):
        """Create a ShopifyFormatGenerator instance for testing"""
        column_requirements_path = Path(__file__).parent.parent.parent / "shared" / "data" / "product_import" / "product_import-column-requirements.py"
        return ShopifyFormatGenerator(str(column_requirements_path))
    
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
            'cost': [43.76, 81.97, 30.87],
            'price': [75.49, 127.79, 45.69],
            'body_html': ['<p>High-quality accelerator pedal pad</p>', '<p>Premium brake pads</p>', '<p>Universal mirror</p>'],
            'collection': ['Ford Parts', 'Chevrolet Parts', 'Universal Parts'],
            'product_type': ['Pedal Pad', 'Brake Pad', 'Mirror'],
            'meta_title': ['Ford Mustang Accelerator Pedal Pad 1965-1970', 'Chevrolet Camaro Brake Pad Set 1969-1970', 'Universal Mirror'],
            'meta_description': ['Premium accelerator pedal pad for 1965-1970 Ford Mustang', 'High-quality brake pads for 1969-1970 Chevrolet Camaro', 'Universal mirror for all vehicles']
        })
    
    def test_load_column_requirements_success(self, generator):
        """Test successful loading of column requirements"""
        requirements = generator.load_column_requirements()
        
        assert isinstance(requirements, dict)
        assert 'required_columns' in requirements
        assert 'column_order' in requirements
        assert len(requirements['required_columns']) == 65
    
    def test_load_column_requirements_file_not_found(self):
        """Test handling of missing column requirements file"""
        generator = ShopifyFormatGenerator("nonexistent_file.py")
        
        with pytest.raises(FileNotFoundError):
            generator.load_column_requirements()
    
    def test_generate_shopify_format_success(self, generator, sample_product_data):
        """Test successful generation of Shopify format"""
        shopify_df = generator.generate_shopify_format(sample_product_data)
        
        # Check that all 65 columns are present
        assert len(shopify_df.columns) == 65
        
        # Check core required columns that should be present
        core_required_columns = [
            'ID', 'Title', 'Body HTML', 'Vendor', 'Tags', 'Option1 Name', 'Option1 Value', 
            'Option2 Name', 'Option2 Value', 'Option3 Name', 'Option3 Value', 'Variant SKU', 
            'Variant Price', 'Variant Cost', 'Variant Inventory Tracker', 'Variant Fulfillment Service'
        ]
        
        for col in core_required_columns:
            assert col in shopify_df.columns, f"Missing required column: {col}"
        
        # Check data integrity
        assert len(shopify_df) == len(sample_product_data)
        assert not shopify_df['Title'].isna().any()
        assert not shopify_df['Variant SKU'].isna().any()
        assert not shopify_df['Variant Price'].isna().any()
    
    def test_generate_shopify_format_empty_dataframe(self, generator):
        """Test generation with empty DataFrame"""
        empty_df = pd.DataFrame()
        shopify_df = generator.generate_shopify_format(empty_df)
        
        # Should return empty DataFrame with correct columns
        assert len(shopify_df.columns) == 65
        assert len(shopify_df) == 0
    
    def test_validate_column_compliance_success(self, generator, sample_product_data):
        """Test successful column compliance validation"""
        shopify_df = generator.generate_shopify_format(sample_product_data)
        validation_result = generator.validate_column_compliance(shopify_df)
        
        assert validation_result['is_compliant'] is True
        assert validation_result['column_count'] == 65
        assert len(validation_result['missing_columns']) == 0
        assert len(validation_result['extra_columns']) == 0
    
    def test_validate_column_compliance_missing_columns(self, generator):
        """Test column compliance validation with missing columns"""
        # Create DataFrame with missing columns
        incomplete_df = pd.DataFrame({
            'Title': ['Test Product'],
            'Variant Price': [10.0]
        })
        
        validation_result = generator.validate_column_compliance(incomplete_df)
        
        assert validation_result['is_compliant'] is False
        assert validation_result['column_count'] < 65
        assert len(validation_result['missing_columns']) > 0
    
    def test_map_product_data_to_shopify_columns(self, generator, sample_product_data):
        """Test mapping of product data to Shopify columns"""
        mapped_data = generator.map_product_data_to_shopify_columns(sample_product_data.iloc[0])
        
        assert isinstance(mapped_data, dict)
        assert 'Title' in mapped_data
        assert 'Variant SKU' in mapped_data
        assert 'Variant Price' in mapped_data
        assert 'Variant Cost' in mapped_data or 'Cost per item' in mapped_data
        assert 'Body (HTML)' in mapped_data
        assert 'SEO Title' in mapped_data
        assert 'SEO Description' in mapped_data
        
        # Check values
        assert mapped_data['Title'] == 'Accelerator Pedal Pad'
        assert mapped_data['Variant SKU'] == '10-0001-40'
        assert mapped_data['Variant Price'] == 75.49
    
    def test_generate_handle_from_title(self, generator):
        """Test handle generation from product title"""
        test_cases = [
            ("Accelerator Pedal Pad", "accelerator-pedal-pad"),
            ("Brake Pad Set - Premium", "brake-pad-set-premium"),
            ("Universal Mirror (Chrome)", "universal-mirror-chrome"),
            ("1965-1970 Ford Mustang Part", "1965-1970-ford-mustang-part")
        ]
        
        for title, expected_handle in test_cases:
            handle = generator.generate_handle_from_title(title)
            assert handle == expected_handle, f"Failed for title: {title}"
    
    def test_generate_tags_from_product_data(self, generator):
        """Test tag generation from product data"""
        product_data = pd.Series({
            'make': 'Ford',
            'model': 'Mustang',
            'year_min': 1965,
            'year_max': 1970,
            'product_type': 'Pedal Pad',
            'collection': 'Ford Parts'
        })
        
        tags = generator.generate_tags_from_product_data(product_data)
        
        assert isinstance(tags, str)
        assert 'Ford' in tags
        assert 'Mustang' in tags
        assert '1965-1970' in tags
        assert 'Pedal Pad' in tags
    
    def test_set_default_shopify_values(self, generator):
        """Test setting of default Shopify values"""
        defaults = generator.set_default_shopify_values()
        
        assert isinstance(defaults, dict)
        assert 'Published' in defaults
        assert 'Variant Inventory Tracker' in defaults
        assert 'Variant Fulfillment Service' in defaults
        assert 'Variant Requires Shipping' in defaults
        assert 'Variant Taxable' in defaults
        assert 'Gift Card' in defaults
        
        # Check default values
        assert defaults['Published'] is True
        assert defaults['Gift Card'] is False
        assert defaults['Variant Requires Shipping'] is True
        assert defaults['Variant Taxable'] is True
    
    def test_handle_single_variant_products(self, generator, sample_product_data):
        """Test handling of single variant products"""
        shopify_df = generator.generate_shopify_format(sample_product_data)
        
        # Check that option fields are empty for single variants
        assert shopify_df['Option1 Name'].isna().all()
        assert shopify_df['Option1 Value'].isna().all()
        assert shopify_df['Option2 Name'].isna().all()
        assert shopify_df['Option2 Value'].isna().all()
        assert shopify_df['Option3 Name'].isna().all()
        assert shopify_df['Option3 Value'].isna().all()
    
    def test_generate_google_shopping_fields(self, generator, sample_product_data):
        """Test generation of Google Shopping fields"""
        shopify_df = generator.generate_shopify_format(sample_product_data)
        
        # Check Google Shopping fields
        assert 'Google Shopping / MPN' in shopify_df.columns
        assert 'Google Shopping / Condition' in shopify_df.columns
        assert 'Google Shopping / Google Product Category' in shopify_df.columns
        
        # Check values
        assert shopify_df['Google Shopping / MPN'].iloc[0] == '10-0001-40'
        assert shopify_df['Google Shopping / Condition'].iloc[0] == 'new'
    
    def test_handle_missing_required_fields(self, generator):
        """Test handling of products with missing required fields"""
        incomplete_data = pd.DataFrame({
            'title': ['Test Product'],
            'mpn': ['TEST-001'],
            # Missing price, cost, etc.
        })
        
        shopify_df = generator.generate_shopify_format(incomplete_data)
        
        # Should handle gracefully with defaults
        assert len(shopify_df) == 1
        assert shopify_df['Title'].iloc[0] == 'Test Product'
        assert shopify_df['Variant Price'].iloc[0] == 0  # Default value


class TestSEOContentGenerator:
    """Test suite for SEO content generation functionality"""
    
    @pytest.fixture
    def seo_generator(self):
        """Create an SEOContentGenerator instance for testing"""
        return SEOContentGenerator()
    
    def test_generate_meta_title_standard(self, seo_generator):
        """Test generation of standard meta titles"""
        product_data = {
            'title': 'Accelerator Pedal Pad',
            'make': 'Ford',
            'model': 'Mustang',
            'year_min': 1965,
            'year_max': 1970
        }
        
        meta_title = seo_generator.generate_meta_title(product_data)
        
        assert isinstance(meta_title, str)
        assert len(meta_title) <= 60  # SEO best practice
        assert 'Ford' in meta_title
        assert 'Mustang' in meta_title
        assert 'Accelerator Pedal Pad' in meta_title
    
    def test_generate_meta_title_universal(self, seo_generator):
        """Test generation of meta titles for universal products"""
        product_data = {
            'title': 'Universal Mirror',
            'make': 'Universal',
            'model': 'All',
            'year_min': 1900,
            'year_max': 2024
        }
        
        meta_title = seo_generator.generate_meta_title(product_data)
        
        assert isinstance(meta_title, str)
        assert len(meta_title) <= 60
        assert 'Universal' in meta_title
        assert 'Mirror' in meta_title
    
    def test_generate_meta_description_standard(self, seo_generator):
        """Test generation of standard meta descriptions"""
        product_data = {
            'title': 'Accelerator Pedal Pad',
            'make': 'Ford',
            'model': 'Mustang',
            'year_min': 1965,
            'year_max': 1970,
            'mpn': '10-0001-40'
        }
        
        meta_description = seo_generator.generate_meta_description(product_data)
        
        assert isinstance(meta_description, str)
        assert len(meta_description) <= 160  # SEO best practice
        assert 'Ford' in meta_description
        assert 'Mustang' in meta_description
        assert '1965-1970' in meta_description
        assert '10-0001-40' in meta_description
    
    def test_generate_meta_description_universal(self, seo_generator):
        """Test generation of meta descriptions for universal products"""
        product_data = {
            'title': 'Universal Mirror',
            'make': 'Universal',
            'model': 'All',
            'year_min': 1900,
            'year_max': 2024,
            'mpn': '10-0003-35'
        }
        
        meta_description = seo_generator.generate_meta_description(product_data)
        
        assert isinstance(meta_description, str)
        assert len(meta_description) <= 160
        assert 'Universal' in meta_description
        assert '10-0003-35' in meta_description
    
    def test_generate_body_html_standard(self, seo_generator):
        """Test generation of HTML body content"""
        product_data = {
            'title': 'Accelerator Pedal Pad',
            'make': 'Ford',
            'model': 'Mustang',
            'year_min': 1965,
            'year_max': 1970,
            'mpn': '10-0001-40'
        }
        
        body_html = seo_generator.generate_body_html(product_data)
        
        assert isinstance(body_html, str)
        assert '<p>' in body_html
        assert '</p>' in body_html
        assert 'Ford' in body_html
        assert 'Mustang' in body_html
        assert '1965-1970' in body_html
        assert '10-0001-40' in body_html
    
    def test_generate_body_html_with_features(self, seo_generator):
        """Test generation of HTML body content with product features"""
        product_data = {
            'title': 'Brake Pad Set',
            'make': 'Chevrolet',
            'model': 'Camaro',
            'year_min': 1969,
            'year_max': 1970,
            'mpn': '10-0002-35',
            'product_type': 'Brake Pad'
        }
        
        body_html = seo_generator.generate_body_html(product_data)
        
        assert '<ul>' in body_html
        assert '<li>' in body_html
        assert 'Direct fit replacement' in body_html
        assert 'Quality tested' in body_html
    
    def test_optimize_for_seo_length_limits(self, seo_generator):
        """Test SEO optimization with length limits"""
        # Test very long title
        long_title = "Very Long Product Title That Exceeds The Recommended SEO Length Limits For Meta Titles"
        optimized = seo_generator.optimize_for_seo_length(long_title, max_length=60)
        
        assert len(optimized) <= 60
        assert optimized.endswith('...')
        
        # Test normal length title
        normal_title = "Normal Product Title"
        optimized = seo_generator.optimize_for_seo_length(normal_title, max_length=60)
        
        assert len(optimized) <= 60
        assert optimized == normal_title
    
    def test_generate_year_range_string(self, seo_generator):
        """Test generation of year range strings"""
        test_cases = [
            (1965, 1965, "1965"),
            (1965, 1970, "1965-1970"),
            (1900, 2024, "Universal"),
            (2000, 2024, "2000-2024")
        ]
        
        for year_min, year_max, expected in test_cases:
            result = seo_generator.generate_year_range_string(year_min, year_max)
            assert result == expected, f"Failed for years {year_min}-{year_max}"
    
    def test_generate_product_features_list(self, seo_generator):
        """Test generation of product features list"""
        product_data = {
            'product_type': 'Brake Pad',
            'make': 'Ford',
            'model': 'Mustang'
        }
        
        features = seo_generator.generate_product_features_list(product_data)
        
        assert isinstance(features, list)
        assert len(features) > 0
        assert any('replacement' in feature.lower() for feature in features)
        assert any('quality' in feature.lower() for feature in features)


class TestShopifyFormatIntegration:
    """Integration tests for Shopify format generation"""
    
    @pytest.fixture
    def generator(self):
        column_requirements_path = Path(__file__).parent.parent.parent / "shared" / "data" / "product_import" / "product_import-column-requirements.py"
        return ShopifyFormatGenerator(str(column_requirements_path))
    
    def test_complete_format_generation_pipeline(self, generator):
        """Test complete format generation pipeline"""
        # Create realistic test data
        product_data = pd.DataFrame({
            'title': ['Ford Mustang Accelerator Pedal Pad', 'Chevrolet Camaro Brake Pad Set'],
            'year_min': [1965, 1969],
            'year_max': [1970, 1970],
            'make': ['Ford', 'Chevrolet'],
            'model': ['Mustang', 'Camaro'],
            'mpn': ['10-0001-40', '10-0002-35'],
            'cost': [43.76, 81.97],
            'price': [75.49, 127.79],
            'body_html': ['<p>High-quality pedal pad</p>', '<p>Premium brake pads</p>'],
            'collection': ['Ford Parts', 'Chevrolet Parts'],
            'product_type': ['Pedal Pad', 'Brake Pad'],
            'meta_title': ['Ford Mustang Pedal Pad 1965-1970', 'Chevrolet Camaro Brake Pads 1969-1970'],
            'meta_description': ['Premium pedal pad for 1965-1970 Ford Mustang', 'High-quality brake pads for 1969-1970 Chevrolet Camaro']
        })
        
        # Generate Shopify format
        shopify_df = generator.generate_shopify_format(product_data)
        
        # Validate output
        validation_result = generator.validate_column_compliance(shopify_df)
        
        assert validation_result['is_compliant'] is True
        assert len(shopify_df) == 2
        assert len(shopify_df.columns) == 65
        
        # Check specific values
        assert shopify_df['Title'].iloc[0] == 'Ford Mustang Accelerator Pedal Pad'
        assert shopify_df['Variant SKU'].iloc[0] == '10-0001-40'
        assert shopify_df['Variant Price'].iloc[0] == 75.49
        assert shopify_df['SEO Title'].iloc[0] == 'Ford Mustang Pedal Pad 1965-1970'
    
    def test_performance_with_large_dataset(self, generator):
        """Test performance with large dataset"""
        import time
        
        # Create large test dataset
        large_data = pd.DataFrame({
            'title': [f'Test Product {i}' for i in range(1000)],
            'year_min': [1965] * 1000,
            'year_max': [1970] * 1000,
            'make': ['Ford'] * 1000,
            'model': ['Mustang'] * 1000,
            'mpn': [f'TEST-{i:04d}' for i in range(1000)],
            'cost': [10.0] * 1000,
            'price': [20.0] * 1000,
            'body_html': ['<p>Test product</p>'] * 1000,
            'collection': ['Test Collection'] * 1000,
            'product_type': ['Test Type'] * 1000,
            'meta_title': ['Test Meta Title'] * 1000,
            'meta_description': ['Test meta description'] * 1000
        })
        
        start_time = time.time()
        shopify_df = generator.generate_shopify_format(large_data)
        elapsed_time = time.time() - start_time
        
        # Should process reasonably quickly
        assert elapsed_time < 30.0  # Less than 30 seconds for 1000 items
        assert len(shopify_df) == 1000
        assert len(shopify_df.columns) == 65
    
    def test_error_handling_with_malformed_data(self, generator):
        """Test error handling with malformed data"""
        # Create data with various issues
        malformed_data = pd.DataFrame({
            'title': ['Good Product', None, ''],  # Missing and empty titles
            'year_min': [1965, 'invalid', 1969],  # Invalid year
            'year_max': [1970, 1970, None],  # Missing year
            'make': ['Ford', 'Chevrolet', 'Dodge'],
            'model': ['Mustang', 'Camaro', 'Challenger'],
            'mpn': ['GOOD-001', None, 'GOOD-003'],  # Missing MPN
            'cost': [10.0, -5.0, 'invalid'],  # Negative and invalid cost
            'price': [20.0, 15.0, None],  # Missing price
        })
        
        # Should handle gracefully without crashing
        shopify_df = generator.generate_shopify_format(malformed_data)
        
        assert len(shopify_df) == 3  # Should process all rows
        assert len(shopify_df.columns) == 65
        
        # Check that fallback values were used
        assert not shopify_df['Title'].iloc[0] == ''  # Should have valid title
        assert shopify_df['Variant Price'].iloc[2] == 0.0  # Should use default price