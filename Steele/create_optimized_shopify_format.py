#!/usr/bin/env python3
"""
Create Optimized Shopify Format
Reduces file size by optimizing tags, removing duplicates, and keeping only essential columns
"""

import pandas as pd
import os
from datetime import datetime

def optimize_shopify_format():
    """Create an optimized version of the Shopify format that's much smaller"""
    
    print("🔧 Creating Optimized Shopify Format")
    print("=" * 60)
    
    # Load the complete file
    input_file = "data/results/complete_shopify_format_20250705_160409.csv"
    
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return
    
    print(f"📁 Loading complete file: {input_file}")
    print(f"   📊 Original size: {os.path.getsize(input_file) / (1024*1024):.1f} MB")
    
    df = pd.read_csv(input_file)
    print(f"   📊 Total products: {len(df):,}")
    print(f"   📊 Total columns: {len(df.columns)}")
    
    # Analyze current tag lengths
    print("\n🔍 Analyzing tag lengths...")
    tag_lengths = df['Tags'].astype(str).str.len()
    print(f"   Average tag length: {tag_lengths.mean():.0f} characters")
    print(f"   Max tag length: {tag_lengths.max():,} characters")
    print(f"   Tags over 1000 chars: {(tag_lengths > 1000).sum()}")
    
    # Create optimized version
    print("\n🎯 Creating optimized version...")
    
    # 1. Optimize tags - limit to top 20 vehicles or 500 characters max
    def optimize_tags(tags_str):
        """Optimize vehicle tags by limiting length"""
        if pd.isna(tags_str) or tags_str == '' or tags_str == 'nan':
            return ''
        
        tags_str = str(tags_str)
        
        # If it's a simple tag, keep it
        if len(tags_str) <= 200:
            return tags_str
        
        # If it's a long vehicle list, limit it
        if ',' in tags_str:
            tag_list = [tag.strip() for tag in tags_str.split(',')]
            # Keep only first 20 tags or up to 500 characters
            optimized_tags = []
            char_count = 0
            for tag in tag_list[:20]:  # Max 20 tags
                if char_count + len(tag) + 2 <= 500:  # Max 500 chars
                    optimized_tags.append(tag)
                    char_count += len(tag) + 2
                else:
                    break
            return ', '.join(optimized_tags)
        
        # If it's one long tag, truncate
        return tags_str[:500]
    
    # Apply tag optimization
    df['Tags'] = df['Tags'].apply(optimize_tags)
    
    # 2. Remove duplicate columns (generated_tags is same as Tags)
    columns_to_drop = ['generated_tags', 'ai_reasoning', 'extraction_error']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    # 3. Keep only essential Shopify columns
    essential_columns = [
        'ID', 'Command', 'Title', 'Body HTML', 'Vendor', 'Tags', 'Tags Command',
        'Category: ID', 'Category: Name', 'Category', 'Custom Collections', 'Smart Collections',
        'Image Type', 'Image Src', 'Image Command', 'Image Position', 'Image Width', 'Image Height', 'Image Alt Text',
        'Variant Inventory Item ID', 'Variant ID', 'Variant Command', 
        'Option1 Name', 'Option1 Value', 'Option2 Name', 'Option2 Value', 'Option3 Name', 'Option3 Value',
        'Variant Position', 'Variant SKU', 'Variant Barcode', 'Variant Image', 
        'Variant Weight', 'Variant Weight Unit', 'Variant Price', 'Variant Compare At Price',
        'Variant Taxable', 'Variant Tax Code', 'Variant Inventory Tracker', 'Variant Inventory Policy',
        'Variant Fulfillment Service', 'Variant Requires Shipping', 'Variant Inventory Qty', 'Variant Inventory Adjust',
        'Variant Cost', 'Variant HS Code', 'Variant Country of Origin', 'Variant Province of Origin',
        'Metafield: title_tag [string]', 'Metafield: description_tag [string]',
        'Metafield: custom.engine_types [list.single_line_text_field]',
        'Metafield: mm-google-shopping.custom_product [boolean]',
        'Variant Metafield: mm-google-shopping.mpn [single_line_text_field]',
        'Variant Metafield: mm-google-shopping.condition [single_line_text_field]',
        'Metafield: mm-google-shopping.mpn [single_line_text_field]'
    ]
    
    # Keep only columns that exist in the dataframe
    existing_essential = [col for col in essential_columns if col in df.columns]
    optimized_df = df[existing_essential].copy()
    
    # 4. Optimize meta descriptions (limit to 160 chars)
    if 'Metafield: description_tag [string]' in optimized_df.columns:
        optimized_df['Metafield: description_tag [string]'] = optimized_df['Metafield: description_tag [string]'].astype(str).str[:160]
    
    # 5. Optimize meta titles (limit to 60 chars)
    if 'Metafield: title_tag [string]' in optimized_df.columns:
        optimized_df['Metafield: title_tag [string]'] = optimized_df['Metafield: title_tag [string]'].astype(str).str[:60]
    
    # Save optimized version
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"data/results/optimized_shopify_format_{timestamp}.csv"
    optimized_df.to_csv(output_file, index=False)
    
    # Show results
    print(f"\n✅ Optimized Shopify format created!")
    print(f"   📁 Output file: {output_file}")
    print(f"   📊 Products: {len(optimized_df):,}")
    print(f"   📊 Columns: {len(optimized_df.columns)} (was {len(df.columns)})")
    
    original_size = os.path.getsize(input_file) / (1024*1024)
    new_size = os.path.getsize(output_file) / (1024*1024)
    
    print(f"   📊 Original size: {original_size:.1f} MB")
    print(f"   📊 New size: {new_size:.1f} MB")
    print(f"   📊 Size reduction: {((original_size - new_size) / original_size * 100):.1f}%")
    
    # Check optimized tag lengths
    new_tag_lengths = optimized_df['Tags'].astype(str).str.len()
    print(f"\n🎯 Tag optimization results:")
    print(f"   Average tag length: {new_tag_lengths.mean():.0f} characters (was {tag_lengths.mean():.0f})")
    print(f"   Max tag length: {new_tag_lengths.max():,} characters (was {tag_lengths.max():,})")
    print(f"   Tags over 500 chars: {(new_tag_lengths > 500).sum()} (was {(tag_lengths > 1000).sum()})")
    
    # Show sample data
    print(f"\n🔍 Sample optimized data:")
    print(f"   Title: {optimized_df.iloc[0]['Title']}")
    print(f"   Body HTML: {optimized_df.iloc[0]['Body HTML'][:80]}...")
    print(f"   Tags: {optimized_df.iloc[0]['Tags'][:80]}...")
    print(f"   Variant SKU: {optimized_df.iloc[0]['Variant SKU']}")
    print(f"   Variant Price: {optimized_df.iloc[0]['Variant Price']}")
    
    return output_file

if __name__ == "__main__":
    optimize_shopify_format() 