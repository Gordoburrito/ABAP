#!/usr/bin/env python3
"""
Create Sample from Complete Shopify Format
Extracts a manageable sample from the complete Shopify format file
"""

import pandas as pd
import os

def create_complete_sample(sample_size=25):
    """Create a sample of the complete Shopify format file"""
    
    print(f"🔍 Creating Complete Sample ({sample_size} products)")
    print("=" * 60)
    
    # Load the complete file
    complete_file = "data/results/complete_shopify_format_20250705_160409.csv"
    
    if not os.path.exists(complete_file):
        print(f"❌ Complete file not found: {complete_file}")
        return
    
    print(f"📁 Loading complete file: {complete_file}")
    df = pd.read_csv(complete_file)
    print(f"   📊 Total products: {len(df):,}")
    
    # Create sample
    sample_df = df.head(sample_size)
    
    # Save sample
    output_file = f"data/results/COMPLETE_SAMPLE_{sample_size}_products.csv"
    sample_df.to_csv(output_file, index=False)
    
    print(f"✅ Sample created successfully!")
    print(f"   📁 Output file: {output_file}")
    print(f"   📊 Sample size: {len(sample_df)} products")
    print(f"   📊 File size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
    
    # Show sample data
    print("\n🔍 Sample Preview:")
    print(f"   Title: {sample_df.iloc[0]['Title']}")
    print(f"   Body HTML: {sample_df.iloc[0]['Body HTML'][:100]}...")
    print(f"   Tags: {sample_df.iloc[0]['Tags'][:100]}...")
    print(f"   Variant SKU: {sample_df.iloc[0]['Variant SKU']}")
    print(f"   Variant Price: {sample_df.iloc[0]['Variant Price']}")
    
    # Show a few more examples
    print("\n🎯 More Examples:")
    for i in range(min(5, len(sample_df))):
        print(f"   {i+1}. {sample_df.iloc[i]['Title']} - SKU: {sample_df.iloc[i]['Variant SKU']} - Price: ${sample_df.iloc[i]['Variant Price']}")
    
    return output_file

if __name__ == "__main__":
    create_complete_sample() 