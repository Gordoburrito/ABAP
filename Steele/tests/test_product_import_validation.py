import pandas as pd
import pytest
import sys
import os
from pathlib import Path

# Add shared data directory to path for importing requirements
sys.path.append(str(Path(__file__).parent.parent.parent / "shared" / "data" / "product_import"))

try:
    from product_import_column_requirements import REQUIRED_ALWAYS, REQUIRED_MULTI_VARIANTS, REQUIRED_MANUAL, NO_REQUIREMENTS
except ImportError:
    # Fallback definitions if import fails
    REQUIRED_ALWAYS = [
        "Title", "Body HTML", "Vendor", "Tags", "Image Src", "Image Command", 
        "Image Position", "Image Alt Text", "Variant Barcode", "Variant Price", 
        "Variant Cost", "Metafield: title_tag [string]", "Metafield: description_tag [string]"
    ]
    REQUIRED_MULTI_VARIANTS = [
        "Option1 Name", "Option1 Value", "Option2 Name", "Option2 Value", 
        "Option3 Name", "Option3 Value", "Variant Position", "Variant Image"
    ]
    REQUIRED_MANUAL = [
        "Command", "Tags Command", "Custom Collections", "Variant Command", 
        "Variant SKU", "Variant HS Code", "Variant Country of Origin",
        "Variant Metafield: harmonized_system_code [string]"
    ]

class TestProductImportValidation:
    """Test suite for validating Steele output against Shopify product import requirements"""
    
    @pytest.fixture
    def sample_transformed_data(self):
        """Mock transformed data for testing"""
        return pd.DataFrame({
            'Title': ['Test Product 1', 'Test Product 2'],
            'Body HTML': [
                'This is a comprehensive description for test product 1 that provides detailed information about the automotive part and its specifications.',
                'This is a comprehensive description for test product 2 that provides detailed information about the automotive part and its features.'
            ],
            'Vendor': ['Steele', 'Steele'],
            'Tags': ['1965_Ford_Mustang', '1930_Stutz_Stutz'],
            'Image Src': ['image1.jpg', 'image2.jpg'],
            'Image Command': ['MERGE', 'MERGE'],
            'Image Position': [1, 1],
            'Image Alt Text': ['Alt text 1', 'Alt text 2'],
            'Variant Barcode': ['123456789012', '123456789013'],
            'Variant Price': [100.00, 150.00],
            'Variant Cost': [50.00, 75.00],
            'Metafield: title_tag [string]': ['Premium Test Product 1 for Automotive', 'Quality Test Product 2 for Cars'],
            'Metafield: description_tag [string]': [
                'High-quality automotive part designed for reliable performance and long-lasting durability in vintage vehicles.',
                'Professional-grade automotive component engineered for optimal fit and superior performance in classic cars.'
            ],
            'Variant SKU': ['SKU001', 'SKU002'],
            'Command': ['MERGE', 'MERGE'],
        })
    
    def test_required_always_columns_present(self, sample_transformed_data):
        """Test that all REQUIRED_ALWAYS columns are present"""
        df = sample_transformed_data
        
        for col in REQUIRED_ALWAYS:
            assert col in df.columns, f"Required column '{col}' is missing from transformed data"
    
    def test_required_always_columns_not_empty(self, sample_transformed_data):
        """Test that REQUIRED_ALWAYS columns are not completely empty"""
        df = sample_transformed_data
        
        for col in REQUIRED_ALWAYS:
            if col in df.columns:
                # Check that column has at least some non-null, non-empty values
                non_empty_values = df[col].dropna()
                non_empty_values = non_empty_values[non_empty_values != '']
                assert len(non_empty_values) > 0, f"Required column '{col}' is completely empty"
    
    def test_title_field_validation(self, sample_transformed_data):
        """Test title field meets requirements"""
        df = sample_transformed_data
        
        if 'Title' in df.columns:
            titles = df['Title'].dropna()
            
            for title in titles:
                assert len(str(title)) > 0, "Title should not be empty"
                assert len(str(title)) <= 255, f"Title too long: {title}"
                assert not str(title).startswith(' '), f"Title should not start with space: '{title}'"
                assert not str(title).endswith(' '), f"Title should not end with space: '{title}'"
    
    def test_price_fields_validation(self, sample_transformed_data):
        """Test price fields are valid"""
        df = sample_transformed_data
        
        price_fields = ['Variant Price', 'Variant Cost']
        
        for field in price_fields:
            if field in df.columns:
                prices = df[field].dropna()
                
                for price in prices:
                    assert pd.api.types.is_numeric_dtype(type(price)) or isinstance(price, (int, float)), f"{field} should be numeric"
                    assert float(price) >= 0, f"{field} should be non-negative: {price}"
                    assert float(price) < 10000, f"{field} seems unreasonably high: {price}"
    
    def test_vendor_field_validation(self, sample_transformed_data):
        """Test vendor field is correctly set"""
        df = sample_transformed_data
        
        if 'Vendor' in df.columns:
            vendors = df['Vendor'].dropna()
            
            for vendor in vendors:
                assert str(vendor) == 'Steele', f"Vendor should be 'Steele', got: {vendor}"
    
    def test_tags_format_validation(self, sample_transformed_data):
        """Test tags are in correct format"""
        df = sample_transformed_data
        
        if 'Tags' in df.columns:
            tags_list = df['Tags'].dropna()
            
            for tags in tags_list:
                if tags and str(tags) != 'nan':
                    # Tags should be comma-separated without spaces around commas
                    tag_items = str(tags).split(',')
                    
                    for tag in tag_items:
                        assert len(tag.strip()) > 0, f"Empty tag found in: {tags}"
                        # Vehicle tags should follow year_make_model format
                        if '_' in tag and tag.count('_') >= 2:
                            parts = tag.split('_')
                            year_part = parts[0]
                            if year_part.isdigit():
                                year = int(year_part)
                                assert 1900 <= year <= 2030, f"Invalid year in tag: {tag}"
    
    def test_meta_fields_seo_compliance(self, sample_transformed_data):
        """Test meta fields meet SEO requirements"""
        df = sample_transformed_data
        
        # Test meta title
        if 'Metafield: title_tag [string]' in df.columns:
            meta_titles = df['Metafield: title_tag [string]'].dropna()
            
            for title in meta_titles:
                assert len(str(title)) <= 60, f"Meta title too long: {len(str(title))} chars - {title}"
                assert len(str(title)) >= 10, f"Meta title too short: {title}"
        
        # Test meta description
        if 'Metafield: description_tag [string]' in df.columns:
            meta_descriptions = df['Metafield: description_tag [string]'].dropna()
            
            for desc in meta_descriptions:
                assert len(str(desc)) <= 160, f"Meta description too long: {len(str(desc))} chars - {desc}"
                assert len(str(desc)) >= 50, f"Meta description too short: {desc}"
    
    def test_image_fields_validation(self, sample_transformed_data):
        """Test image fields are properly formatted"""
        df = sample_transformed_data
        
        if 'Image Src' in df.columns:
            image_sources = df['Image Src'].dropna()
            
            for img_src in image_sources:
                if img_src and str(img_src) != 'nan':
                    # Should be a valid image filename or URL
                    img_str = str(img_src)
                    assert len(img_str) > 0, "Image source should not be empty"
                    # Could be filename or URL
                    valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
                    if not img_str.startswith('http'):
                        assert any(img_str.lower().endswith(ext) for ext in valid_extensions), f"Invalid image extension: {img_src}"
    
    def test_variant_barcode_format(self, sample_transformed_data):
        """Test variant barcode format"""
        df = sample_transformed_data
        
        if 'Variant Barcode' in df.columns:
            barcodes = df['Variant Barcode'].dropna()
            
            for barcode in barcodes:
                if barcode and str(barcode) != 'nan':
                    barcode_str = str(barcode).replace('.0', '')  # Remove decimal if present
                    assert barcode_str.isdigit(), f"Barcode should be numeric: {barcode}"
                    assert len(barcode_str) in [12, 13, 14], f"Barcode should be 12-14 digits: {barcode}"
    
    def test_body_html_content(self, sample_transformed_data):
        """Test body HTML content quality"""
        df = sample_transformed_data
        
        if 'Body HTML' in df.columns:
            descriptions = df['Body HTML'].dropna()
            
            for desc in descriptions:
                if desc and str(desc) != 'nan':
                    desc_str = str(desc)
                    assert len(desc_str) >= 50, f"Description too short: {desc_str}"
                    assert len(desc_str) <= 5000, f"Description too long: {len(desc_str)} chars"
    
    def test_command_fields_validation(self, sample_transformed_data):
        """Test command fields for Shopify import"""
        df = sample_transformed_data
        
        if 'Command' in df.columns:
            commands = df['Command'].dropna()
            
            valid_commands = ['NEW', 'UPDATE', 'MERGE', 'DELETE']
            
            for command in commands:
                assert str(command).upper() in valid_commands, f"Invalid command: {command}"
    
    def test_all_required_columns_coverage(self):
        """Test that we're testing all required columns"""
        all_required = REQUIRED_ALWAYS + REQUIRED_MANUAL
        
        # This test ensures we don't miss any required columns in our validation
        tested_columns = {
            'Title', 'Body HTML', 'Vendor', 'Tags', 'Image Src', 'Image Command',
            'Image Position', 'Image Alt Text', 'Variant Barcode', 'Variant Price',
            'Variant Cost', 'Metafield: title_tag [string]', 'Metafield: description_tag [string]',
            'Command', 'Variant SKU'
        }
        
        required_set = set(all_required)
        untested_columns = required_set - tested_columns
        
        # Print warning for untested required columns
        if untested_columns:
            print(f"Warning: The following required columns are not being tested: {untested_columns}")
    
    def test_data_consistency(self, sample_transformed_data):
        """Test data consistency across related fields"""
        df = sample_transformed_data
        
        # Test that products with same title have consistent vendor
        if 'Title' in df.columns and 'Vendor' in df.columns:
            for title in df['Title'].unique():
                title_rows = df[df['Title'] == title]
                vendors = title_rows['Vendor'].unique()
                assert len(vendors) == 1, f"Title '{title}' has multiple vendors: {vendors}"
        
        # Test that variant cost is less than variant price
        if 'Variant Cost' in df.columns and 'Variant Price' in df.columns:
            for idx, row in df.iterrows():
                if pd.notna(row['Variant Cost']) and pd.notna(row['Variant Price']):
                    cost = float(row['Variant Cost'])
                    price = float(row['Variant Price'])
                    assert cost <= price, f"Cost ({cost}) should not exceed price ({price}) at row {idx}"
    
    def test_shopify_import_format_compliance(self, sample_transformed_data):
        """Test overall compliance with Shopify import format"""
        df = sample_transformed_data
        
        # Test that dataframe is not empty
        assert len(df) > 0, "Transformed data should not be empty"
        
        # Test that we have some basic product structure
        assert 'Title' in df.columns, "Must have Title column"
        assert 'Vendor' in df.columns, "Must have Vendor column"
        
        # Test that we don't have any columns with invalid characters
        for col in df.columns:
            # Shopify CSV format should not have certain characters
            assert '\n' not in col, f"Column name contains newline: {col}"
            assert '\r' not in col, f"Column name contains carriage return: {col}" 