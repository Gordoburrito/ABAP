#!/usr/bin/env python3
"""
Create Sample Shopify Format File
Takes the large Shopify format file and creates a small sample for review
"""

import pandas as pd
import os

def create_sample_shopify_format(sample_size=5):
    """Create a sample of the Shopify format file for easy review"""
    
    print(f"🔍 Creating Sample Shopify Format File ({sample_size} products)")
    print("=" * 60)
    
    # Find the most recent shopify format file
    results_dir = "data/results"
    shopify_files = [f for f in os.listdir(results_dir) if f.startswith("shopify_format_batch_")]
    
    if not shopify_files:
        print("❌ No Shopify format files found!")
        return
    
    # Get the most recent file
    latest_file = max(shopify_files)
    input_file = os.path.join(results_dir, latest_file)
    
    print(f"📁 Source file: {input_file}")
    
    try:
        # Read the file
        df = pd.read_csv(input_file)
        print(f"📊 Total products in source: {len(df)}")
        
        # Take first few rows as sample
        sample_df = df.head(sample_size)
        
        # Create sample filename
        sample_file = os.path.join(results_dir, f"shopify_format_SAMPLE_{sample_size}_products.csv")
        
        # Save sample
        sample_df.to_csv(sample_file, index=False)
        
        print(f"✅ Sample created: {sample_file}")
        print(f"📊 Sample contains: {len(sample_df)} products")
        
        # Show preview of sample data
        print("\n📋 Sample Preview:")
        print("-" * 60)
        
        key_cols = ['Command', 'Title', 'Vendor', 'Tags', 'Variant SKU', 'ai_extracted_make', 'ai_extracted_model', 'ai_confidence']
        
        for idx, row in sample_df.iterrows():
            print(f"\n🔧 Product {idx + 1}:")
            for col in key_cols:
                value = row[col] if pd.notna(row[col]) else 'N/A'
                print(f"  {col}: {value}")
        
        print(f"\n✅ Sample file ready for review!")
        print(f"📁 File location: {sample_file}")
        
        return sample_file
        
    except Exception as e:
        print(f"❌ Error creating sample: {e}")
        return None

if __name__ == "__main__":
    create_sample_shopify_format() 