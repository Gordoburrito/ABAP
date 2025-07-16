#!/usr/bin/env python3
"""Quick test of the incomplete fitment pipeline with a small dataset"""

import pandas as pd
import os
from utils.integration_pipeline import IncompleteDataPipeline

def main():
    print("🧪 QUICK PIPELINE TEST")
    print("=" * 50)
    
    # Create small test dataset
    test_data = pd.DataFrame({
        'StockCode': ['10-0001-40', '10-0002-35', '10-0003-35'],
        'Product Name': ['Accelerator Pedal Pad', 'Brake Pad Set', 'Universal Mirror'],
        'Description': ['For 1965-1970 Ford Mustang', 'For 1969-1970 Chevrolet Camaro', 'Universal fit for all vehicles'],
        'MAP': [75.49, 127.79, 45.69],
        'Dealer Price': [43.76, 81.97, 30.87],
        'StockUom': ['EA', 'EA', 'EA'],
        'UPC Code': ['123456789', '987654321', '456789123']
    })
    
    print(f"📋 Test data: {len(test_data)} products")
    for i, row in test_data.iterrows():
        print(f"   {i+1}. {row['Product Name']} (SKU: {row['StockCode']})")
    
    # Configure pipeline (AI disabled for quick test)
    config = {
        'openai_api_key': 'test_key',  # Not needed when AI disabled
        'golden_master_path': '../shared/data/master_ultimate_golden.csv',
        'column_requirements_path': '../shared/data/product_import/product_import-column-requirements.py',
        'batch_size': 10,
        'model': 'gpt-4.1-mini',
        'enable_ai': False,  # Disable AI for quick test
        'max_retries': 3
    }
    
    print("\n🔄 Processing pipeline...")
    
    try:
        # Initialize pipeline
        pipeline = IncompleteDataPipeline(config)
        
        # Save test data
        input_path = 'data/results/quick_test_input.csv'
        output_path = 'data/results/quick_test_output.csv'
        
        os.makedirs('data/results', exist_ok=True)
        test_data.to_csv(input_path, index=False)
        
        # Process pipeline
        results = pipeline.process_complete_pipeline(input_path, output_path)
        
        print("✅ Processing completed!")
        print(f"\n📊 Results:")
        print(f"   - Total Items: {results['total_items']}")
        print(f"   - Processed: {results['processed_items']}")
        print(f"   - Success Rate: {results['success_rate']:.1%}")
        print(f"   - Processing Time: {results['processing_time']:.2f}s")
        print(f"   - Output File: {results['output_file']}")
        
        # Check output
        if os.path.exists(output_path):
            output_df = pd.read_csv(output_path)
            print(f"   - Output Columns: {len(output_df.columns)} (Expected: 65)")
            print(f"   - Shopify Compliant: {'✅ Yes' if len(output_df.columns) == 65 else '❌ No'}")
            
            # Show sample output
            print("\n📋 Sample Output:")
            key_cols = ['Title', 'Variant SKU', 'Variant Price', 'Vendor']
            for col in key_cols:
                if col in output_df.columns:
                    print(f"   - {col}: {output_df[col].iloc[0]}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    print("\n🎉 Quick test completed successfully!")

if __name__ == "__main__":
    main()