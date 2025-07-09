#!/usr/bin/env python3
"""
Test script to validate the new processed Steele data with the existing pipeline.
Tests both the sample and the full processed dataset.
"""

import pandas as pd
import sys
from pathlib import Path
import os

# Add project paths
project_root = Path(__file__).parent.parent
steele_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.steele_data_transformer import SteeleDataTransformer

def test_processed_sample_data():
    """Test the pipeline with the processed sample data."""
    
    print("🧪 Testing with processed sample data...")
    
    transformer = SteeleDataTransformer(use_ai=False)  # Disable AI for faster testing
    
    try:
        # Load the new processed sample
        sample_df = transformer.load_sample_data("data/samples/steele_processed_sample.csv")
        print(f"✅ Loaded {len(sample_df)} records from processed sample")
        print(f"   Columns: {list(sample_df.columns)}")
        
        # Validate golden dataset
        validation_df = transformer.validate_against_golden_dataset(sample_df)
        validated_count = len(validation_df[validation_df['golden_validated'] == True])
        print(f"✅ Golden validation: {validated_count}/{len(sample_df)} products validated")
        
        # Transform to standard format
        standard_products = transformer.transform_to_standard_format(sample_df, validation_df)
        print(f"✅ Transformed {len(standard_products)} products to standard format")
        
        # Enhance with templates
        enhanced_products = transformer.enhance_with_templates(standard_products)
        print(f"✅ Enhanced {len(enhanced_products)} products with templates")
        
        # Convert to Shopify format
        shopify_df = transformer.transform_to_formatted_shopify_import(enhanced_products)
        print(f"✅ Generated {len(shopify_df)} Shopify-ready records")
        
        # Consolidate products
        final_df = transformer.consolidate_products_by_unique_id(shopify_df)
        print(f"✅ Consolidated to {len(final_df)} unique products")
        
        # Validate output
        validation_results = transformer.validate_output(final_df)
        errors = len(validation_results['errors'])
        warnings = len(validation_results['warnings'])
        
        print(f"✅ Validation: {errors} errors, {warnings} warnings")
        
        if errors == 0:
            print("🎉 Sample processing test PASSED!")
            return True
        else:
            print("❌ Sample processing test FAILED!")
            for error in validation_results['errors']:
                print(f"   Error: {error}")
            return False
            
    except Exception as e:
        print(f"❌ Sample test failed with error: {e}")
        return False

def test_specific_products():
    """Test specific products from the processed data."""
    
    print("\n🔍 Testing specific products...")
    
    transformer = SteeleDataTransformer(use_ai=True)  # Enable AI for vehicle tag testing
    
    # Load sample data
    sample_df = transformer.load_sample_data("data/samples/steele_processed_sample.csv")
    
    # Test a few specific products
    test_products = [
        {"StockCode": "10-0001-40", "expected_make": "Stutz"},
        {"StockCode": "10-0002-35", "expected_make": "Stutz"},
    ]
    
    for test_product in test_products:
        stock_code = test_product["StockCode"]
        expected_make = test_product["expected_make"]
        
        # Find product in data
        product_rows = sample_df[sample_df['StockCode'] == stock_code]
        
        if len(product_rows) > 0:
            product_row = product_rows.iloc[0]
            print(f"✅ Found {stock_code}: {product_row['Product Name']}")
            print(f"   Vehicle: {product_row['Year']} {product_row['Make']} {product_row['Model']}")
            
            if product_row['Make'] == expected_make:
                print(f"   ✅ Make matches expected: {expected_make}")
            else:
                print(f"   ⚠️  Make mismatch: expected {expected_make}, got {product_row['Make']}")
        else:
            print(f"❌ Product {stock_code} not found in sample")

def test_data_quality():
    """Test data quality of the processed dataset."""
    
    print("\n📊 Testing data quality...")
    
    # Load processed sample
    sample_df = pd.read_csv("data/samples/steele_processed_sample.csv")
    
    print(f"Dataset shape: {sample_df.shape}")
    print(f"Columns: {list(sample_df.columns)}")
    
    # Check for required columns
    required_cols = ['StockCode', 'Product Name', 'Description', 'MAP', 'Dealer Price', 'Year', 'Make', 'Model']
    missing_cols = set(required_cols) - set(sample_df.columns)
    
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        return False
    else:
        print("✅ All required columns present")
    
    # Check data completeness
    for col in required_cols:
        null_count = sample_df[col].isnull().sum()
        total_count = len(sample_df)
        completeness = (total_count - null_count) / total_count * 100
        print(f"   {col}: {completeness:.1f}% complete ({null_count} nulls)")
    
    # Check price data
    valid_prices = sample_df[sample_df['MAP'] > 0]
    print(f"✅ Products with valid prices: {len(valid_prices)}/{len(sample_df)}")
    
    if len(valid_prices) > 0:
        print(f"   Price range: ${valid_prices['MAP'].min():.2f} - ${valid_prices['MAP'].max():.2f}")
    
    # Check vehicle data
    valid_vehicles = sample_df[
        (sample_df['Year'] > 1900) & 
        (sample_df['Make'] != 'Unknown') & 
        (sample_df['Model'] != 'Unknown')
    ]
    print(f"✅ Products with valid vehicle data: {len(valid_vehicles)}/{len(sample_df)}")
    
    if len(valid_vehicles) > 0:
        print(f"   Year range: {valid_vehicles['Year'].min()} - {valid_vehicles['Year'].max()}")
        print(f"   Unique makes: {valid_vehicles['Make'].nunique()}")
        print(f"   Top makes: {list(valid_vehicles['Make'].value_counts().head(3).index)}")
    
    return True

def performance_benchmark():
    """Run a performance benchmark on the pipeline."""
    
    print("\n⏱️  Performance benchmark...")
    
    import time
    
    transformer = SteeleDataTransformer(use_ai=False)  # Disable AI for speed
    
    start_time = time.time()
    
    # Process sample data
    sample_df = transformer.load_sample_data("data/samples/steele_processed_sample.csv")
    validation_df = transformer.validate_against_golden_dataset(sample_df)
    standard_products = transformer.transform_to_standard_format(sample_df, validation_df)
    enhanced_products = transformer.enhance_with_templates(standard_products)
    shopify_df = transformer.transform_to_formatted_shopify_import(enhanced_products)
    final_df = transformer.consolidate_products_by_unique_id(shopify_df)
    
    end_time = time.time()
    duration = end_time - start_time
    
    records_per_second = len(sample_df) / duration
    
    print(f"✅ Processed {len(sample_df)} records in {duration:.2f} seconds")
    print(f"   Rate: {records_per_second:.1f} records/second")
    print(f"   Output: {len(final_df)} final products")
    
    # Estimate full dataset processing time
    full_dataset_size = 2077649  # From processing summary
    estimated_time = full_dataset_size / records_per_second
    print(f"   Estimated time for full dataset: {estimated_time/60:.1f} minutes")

def main():
    """Run all tests."""
    
    print("🚀 TESTING NEW PROCESSED STEELE DATA")
    print("=" * 50)
    
    # Change to Steele directory
    os.chdir(steele_root)
    
    all_tests_passed = True
    
    # Test 1: Data quality
    if not test_data_quality():
        all_tests_passed = False
    
    # Test 2: Pipeline with sample data
    if not test_processed_sample_data():
        all_tests_passed = False
    
    # Test 3: Specific products
    test_specific_products()
    
    # Test 4: Performance benchmark
    performance_benchmark()
    
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED!")
        print("\nThe new processed data is compatible with the existing pipeline.")
        print("Ready to process the full dataset!")
    else:
        print("❌ SOME TESTS FAILED!")
        print("Please review the errors above.")
    
    print("\nNext steps:")
    print("1. If tests passed, run the full pipeline on steele_processed_complete.csv")
    print("2. Use the AI vehicle tag generation for accurate fitment")
    print("3. Validate the final Shopify-ready output")

if __name__ == "__main__":
    main() 