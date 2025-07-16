#!/usr/bin/env python3
"""Test pipeline with real Steele data (small sample)"""

import pandas as pd
import os
from utils.integration_pipeline import IncompleteDataPipeline

def main():
    print("🔧 REAL DATA TEST")
    print("=" * 50)
    
    # Load real Steele data (first 10 items)
    try:
        df = pd.read_excel('data/raw/steele.xlsx')
        sample_df = df.head(10)  # First 10 products
        
        print(f"📋 Loaded {len(sample_df)} products from steele.xlsx")
        print("Sample products:")
        for i, row in sample_df.head(3).iterrows():
            print(f"   {i+1}. {row['Product Name']} (SKU: {row['StockCode']})")
        print(f"   ... and {len(sample_df)-3} more")
        
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
        'enable_ai': bool(os.getenv('OPENAI_API_KEY')),  # Enable AI if API key available
        'max_retries': 3
    }
    
    print(f"\n⚙️  Configuration:")
    print(f"   - AI Processing: {'Enabled' if config['enable_ai'] else 'Disabled (no API key)'}")
    print(f"   - Model: {config['model']}")
    print(f"   - Sample Size: {len(sample_df)} products")
    
    try:
        # Initialize pipeline
        pipeline = IncompleteDataPipeline(config)
        
        # Estimate cost first
        cost_estimate = pipeline.estimate_processing_cost(sample_df)
        print(f"\n💰 Estimated Cost: ${cost_estimate['total_cost']:.4f}")
        
        # Prepare files
        input_path = 'data/results/real_test_input.csv'
        output_path = 'data/results/real_test_output.csv'
        
        os.makedirs('data/results', exist_ok=True)
        sample_df.to_csv(input_path, index=False)
        
        print(f"\n🔄 Processing {len(sample_df)} real products...")
        
        # Process pipeline
        results = pipeline.process_complete_pipeline(input_path, output_path)
        
        print("✅ Processing completed!")
        print(f"\n📊 Results:")
        print(f"   - Total Items: {results['total_items']}")
        print(f"   - Processed: {results['processed_items']}")
        print(f"   - Failed: {results['failed_items']}")
        print(f"   - Success Rate: {results['success_rate']:.1%}")
        print(f"   - Processing Time: {results['processing_time']:.2f}s")
        print(f"   - Total Cost: ${results['total_cost']:.4f}")
        print(f"   - Cost per Item: ${results['cost_per_item']:.4f}")
        
        # Validate output
        if os.path.exists(output_path):
            output_df = pd.read_csv(output_path)
            print(f"\n✅ Output generated:")
            print(f"   - Rows: {len(output_df)}")
            print(f"   - Columns: {len(output_df.columns)} (Expected: 65)")
            print(f"   - Shopify Compliant: {'✅ Yes' if len(output_df.columns) == 65 else '❌ No'}")
            
            # Show sample
            print(f"\n📋 Sample Output (Product 1):")
            key_cols = ['Title', 'Variant SKU', 'Variant Price', 'Vendor', 'Tags']
            for col in key_cols:
                if col in output_df.columns:
                    print(f"   - {col}: {output_df[col].iloc[0]}")
            
            print(f"\n📁 Output saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n🎉 Real data test completed!")
    print(f"You can open {output_path} to see the full Shopify-format results.")

if __name__ == "__main__":
    main()