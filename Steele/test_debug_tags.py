#!/usr/bin/env python3
"""
Test script with debugging to show year_min, year_max during tag generation
"""

import logging
import os
import pandas as pd
from utils.integration_pipeline import IncompleteDataPipeline

# Set up detailed logging to see all debug info
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def test_debug_with_real_data():
    """Test with debugging enabled to see year_min/year_max"""
    
    print("🔧 DEBUG TEST - Real Data with Year Range Debugging")
    print("=" * 70)
    
    # Load just 3 products for detailed debugging
    try:
        df = pd.read_excel('data/raw/steele.xlsx')
        sample_df = df.head(3)  # Just 3 products for detailed output
        
        print(f"📋 Loaded {len(sample_df)} products for debugging:")
        for i, row in sample_df.iterrows():
            print(f"   {i+1}. {row['Product Name']} (SKU: {row['StockCode']})")
        
    except Exception as e:
        print(f"❌ Could not load steele.xlsx: {e}")
        return
    
    # Configure pipeline
    config = {
        'openai_api_key': os.getenv('OPENAI_API_KEY', 'test_key'),
        'golden_master_path': '../shared/data/master_ultimate_golden.csv',
        'column_requirements_path': '../shared/data/product_import/product_import-column-requirements.py',
        'batch_size': 10,
        'model': 'gpt-4.1-mini',
        'enable_ai': bool(os.getenv('OPENAI_API_KEY')),
        'max_retries': 3
    }
    
    print(f"\n⚙️  Configuration:")
    print(f"   - AI Processing: {'Enabled' if config['enable_ai'] else 'Disabled (no API key)'}")
    print(f"   - Sample Size: {len(sample_df)} products")
    
    try:
        # Initialize pipeline
        pipeline = IncompleteDataPipeline(config)
        
        # Prepare files
        input_path = 'data/results/debug_test_input.csv'
        output_path = 'data/results/debug_test_output.csv'
        
        os.makedirs('data/results', exist_ok=True)
        sample_df.to_csv(input_path, index=False)
        
        print(f"\n🔄 Processing {len(sample_df)} products with full debugging...")
        print("Look for debug messages showing year_min, year_max, make, and models:")
        print("-" * 50)
        
        # Process pipeline - this will show all the debug output
        results = pipeline.process_complete_pipeline(input_path, output_path)
        
        print("-" * 50)
        print("✅ Processing completed!")
        print(f"\n📊 Results:")
        print(f"   - Total Items: {results['total_items']}")
        print(f"   - Processed: {results['processed_items']}")
        print(f"   - Success Rate: {results['success_rate']:.1%}")
        
        # Show final tags for each product
        if os.path.exists(output_path):
            output_df = pd.read_csv(output_path)
            print(f"\n🏷️ Final Tags Results:")
            for i, row in output_df.iterrows():
                tags = row.get('Tags', '')
                print(f"   Product {i+1}: {row.get('Title', 'Unknown')[:50]}...")
                if pd.notna(tags) and tags.strip():
                    print(f"      Tags: {tags}")
                else:
                    print(f"      Tags: (empty)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🎉 Debug test completed!")


if __name__ == "__main__":
    test_debug_with_real_data()