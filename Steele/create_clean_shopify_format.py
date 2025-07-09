#!/usr/bin/env python3
"""
Create Clean Shopify Format
Removes only the AI analysis columns while keeping standard Shopify format
"""

import pandas as pd
import os
from datetime import datetime

def create_clean_shopify_format():
    """Remove AI analysis columns from complete Shopify format"""
    
    print("🧹 Creating Clean Shopify Format")
    print("=" * 60)
    
    # Load the complete file
    input_file = "data/results/complete_shopify_format_20250705_160409.csv"
    
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return
    
    print(f"📁 Loading complete file: {input_file}")
    df = pd.read_csv(input_file)
    print(f"   📊 Total products: {len(df):,}")
    print(f"   📊 Total columns: {len(df.columns)}")
    print(f"   📊 Original size: {os.path.getsize(input_file) / (1024*1024):.1f} MB")
    
    # Remove only the AI analysis columns at the end
    ai_columns_to_remove = [
        'ai_extracted_years',
        'ai_extracted_make', 
        'ai_extracted_model',
        'ai_confidence',
        'ai_reasoning',
        'generated_tags',  # This duplicates the Tags field
        'extraction_error'
    ]
    
    print(f"\n🗑️  Removing AI analysis columns:")
    columns_removed = []
    for col in ai_columns_to_remove:
        if col in df.columns:
            columns_removed.append(col)
            print(f"   ✅ Removing: {col}")
    
    # Create clean dataframe
    clean_df = df.drop(columns=columns_removed)
    
    # Save clean version
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"data/results/clean_shopify_format_{timestamp}.csv"
    clean_df.to_csv(output_file, index=False)
    
    # Show results
    print(f"\n✅ Clean Shopify format created!")
    print(f"   📁 Output file: {output_file}")
    print(f"   📊 Products: {len(clean_df):,}")
    print(f"   📊 Columns: {len(clean_df.columns)} (was {len(df.columns)})")
    print(f"   📊 Columns removed: {len(columns_removed)}")
    
    original_size = os.path.getsize(input_file) / (1024*1024)
    new_size = os.path.getsize(output_file) / (1024*1024)
    
    print(f"   📊 Original size: {original_size:.1f} MB")
    print(f"   📊 New size: {new_size:.1f} MB")
    print(f"   📊 Size reduction: {((original_size - new_size) / original_size * 100):.1f}%")
    
    # Show sample data
    print(f"\n🔍 Sample clean data:")
    print(f"   Title: {clean_df.iloc[0]['Title']}")
    print(f"   Body HTML: {clean_df.iloc[0]['Body HTML'][:100]}...")
    print(f"   Tags: {clean_df.iloc[0]['Tags'][:100]}...")
    print(f"   Variant SKU: {clean_df.iloc[0]['Variant SKU']}")
    print(f"   Variant Price: {clean_df.iloc[0]['Variant Price']}")
    
    print(f"\n📋 Standard Shopify columns preserved:")
    standard_columns = [col for col in clean_df.columns if not col.startswith('ai_')]
    print(f"   Total standard columns: {len(standard_columns)}")
    print(f"   Includes: Title, Body HTML, Tags, Variant SKU, Variant Price, etc.")
    
    return output_file

if __name__ == "__main__":
    create_clean_shopify_format() 