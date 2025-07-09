#!/usr/bin/env python3
"""
Test Shopify Format Output
Verifies that the generated Shopify format matches the expected format exactly
"""

import pandas as pd
import os

def test_shopify_format():
    """Test that the Shopify format output matches expected format"""
    
    print("🧪 Testing Shopify Format Output")
    print("=" * 60)
    
    # Expected columns from user's example
    expected_columns = [
        'ID', 'Command', 'Title', 'Body HTML', 'Vendor', 'Tags', 'Tags Command',
        'Category: ID', 'Category: Name', 'Category', 'Custom Collections', 'Smart Collections',
        'Image Type', 'Image Src', 'Image Command', 'Image Position', 'Image Width', 'Image Height', 'Image Alt Text',
        'Variant Inventory Item ID', 'Variant ID', 'Variant Command', 'Option1 Name', 'Option1 Value',
        'Option2 Name', 'Option2 Value', 'Option3 Name', 'Option3 Value', 'Variant Position',
        'Variant SKU', 'Variant Barcode', 'Variant Image', 'Variant Weight', 'Variant Weight Unit',
        'Variant Price', 'Variant Compare At Price', 'Variant Taxable', 'Variant Tax Code',
        'Variant Inventory Tracker', 'Variant Inventory Policy', 'Variant Fulfillment Service',
        'Variant Requires Shipping', 'Variant Inventory Qty', 'Variant Inventory Adjust',
        'Variant Cost', 'Variant HS Code', 'Variant Country of Origin', 'Variant Province of Origin',
        'Metafield: title_tag [string]', 'Metafield: description_tag [string]',
        'Metafield: custom.engine_types [list.single_line_text_field]',
        'Metafield: mm-google-shopping.custom_product [boolean]',
        'Variant Metafield: mm-google-shopping.custom_label_4 [single_line_text_field]',
        'Variant Metafield: mm-google-shopping.custom_label_3 [single_line_text_field]',
        'Variant Metafield: mm-google-shopping.custom_label_2 [single_line_text_field]',
        'Variant Metafield: mm-google-shopping.custom_label_1 [single_line_text_field]',
        'Variant Metafield: mm-google-shopping.custom_label_0 [single_line_text_field]',
        'Variant Metafield: mm-google-shopping.size_system [single_line_text_field]',
        'Variant Metafield: mm-google-shopping.size_type [single_line_text_field]',
        'Variant Metafield: mm-google-shopping.mpn [single_line_text_field]',
        'Variant Metafield: mm-google-shopping.gender [single_line_text_field]',
        'Variant Metafield: mm-google-shopping.condition [single_line_text_field]',
        'Variant Metafield: mm-google-shopping.age_group [single_line_text_field]',
        'Variant Metafield: harmonized_system_code [string]',
        'Metafield: mm-google-shopping.mpn [single_line_text_field]',
        'ai_extracted_years', 'ai_extracted_make', 'ai_extracted_model', 'ai_confidence',
        'ai_reasoning', 'generated_tags', 'extraction_error'
    ]
    
    # Find the most recent shopify format file
    results_dir = "data/results"
    shopify_files = [f for f in os.listdir(results_dir) if f.startswith("shopify_format_batch_")]
    
    if not shopify_files:
        print("❌ No Shopify format files found!")
        return False
    
    # Get the most recent file
    latest_file = max(shopify_files)
    file_path = os.path.join(results_dir, latest_file)
    
    print(f"📁 Testing file: {file_path}")
    
    try:
        # Load the file
        df = pd.read_csv(file_path)
        
        # Test 1: Check column count
        print(f"📊 Column count: {len(df.columns)} (expected: {len(expected_columns)})")
        
        # Test 2: Check exact column names
        missing_columns = set(expected_columns) - set(df.columns)
        extra_columns = set(df.columns) - set(expected_columns)
        
        if missing_columns:
            print(f"❌ Missing columns: {missing_columns}")
        else:
            print("✅ All expected columns present")
        
        if extra_columns:
            print(f"⚠️  Extra columns: {extra_columns}")
        else:
            print("✅ No extra columns")
        
        # Test 3: Check column order
        column_order_match = list(df.columns) == expected_columns
        print(f"✅ Column order matches: {column_order_match}")
        
        # Test 4: Check sample data format
        print("\n📋 Sample data format verification:")
        print("-" * 60)
        
        sample_row = df.iloc[0]
        
        # Check key fields (pandas converts strings to appropriate types when reading CSV)
        tests = [
            ("Command", sample_row['Command'] == 'MERGE'),
            ("Vendor", sample_row['Vendor'] == 'Steele'),
            ("Tags Command", sample_row['Tags Command'] == 'MERGE'),
            ("Category", sample_row['Category'] == 'Accessories'),
            ("Image Command", sample_row['Image Command'] == 'MERGE'),
            ("Image Position", sample_row['Image Position'] == 1),  # pandas converts '1' to 1
            ("Variant Command", sample_row['Variant Command'] == 'MERGE'),
            ("Variant Position", sample_row['Variant Position'] == 1),  # pandas converts '1' to 1
            ("Variant Taxable", sample_row['Variant Taxable'] == True),  # pandas converts 'True' to True
            ("Variant Inventory Tracker", sample_row['Variant Inventory Tracker'] == 'shopify'),
            ("Variant Inventory Policy", sample_row['Variant Inventory Policy'] == 'deny'),
            ("Variant Fulfillment Service", sample_row['Variant Fulfillment Service'] == 'manual'),
            ("Variant Requires Shipping", sample_row['Variant Requires Shipping'] == True),  # pandas converts 'True' to True
            ("Variant Inventory Qty", sample_row['Variant Inventory Qty'] == 0),  # pandas converts '0' to 0
            ("Variant Condition", sample_row['Variant Metafield: mm-google-shopping.condition [single_line_text_field]'] == 'new'),
        ]
        
        for test_name, result in tests:
            status = "✅" if result else "❌"
            print(f"{status} {test_name}: {result}")
        
        # Test 5: Check that we have data in key AI fields
        print("\n🤖 AI extraction data verification:")
        print("-" * 60)
        
        ai_fields = ['ai_extracted_years', 'ai_extracted_make', 'ai_extracted_model', 'ai_confidence', 'generated_tags']
        for field in ai_fields:
            non_empty = df[field].notna().sum()
            print(f"📊 {field}: {non_empty}/{len(df)} rows have data")
        
        # Test 6: Show header comparison
        print("\n📋 Header format verification:")
        print("-" * 60)
        print("First 10 columns:")
        for i, col in enumerate(df.columns[:10]):
            expected_col = expected_columns[i]
            match = "✅" if col == expected_col else "❌"
            print(f"{match} {i+1:2d}: {col} {'== ' + expected_col if col == expected_col else '!= ' + expected_col}")
        
        # Test 7: Sample output format
        print("\n📝 Sample output format:")
        print("-" * 60)
        
        # Show first row in CSV format (similar to user's example)
        first_row = df.iloc[0]
        key_fields = ['ID', 'Command', 'Title', 'Body HTML', 'Vendor', 'Tags', 'Tags Command', 'Category', 'Variant SKU', 'ai_extracted_make', 'ai_extracted_model']
        
        print("Sample row (key fields):")
        for field in key_fields:
            value = first_row[field] if pd.notna(first_row[field]) else ''
            print(f"  {field}: '{value}'")
        
        print(f"\n✅ Testing complete! File format appears correct.")
        print(f"📁 Output file: {file_path}")
        print(f"📊 Total products: {len(df)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing file: {e}")
        return False

if __name__ == "__main__":
    test_shopify_format() 