#!/usr/bin/env python3
"""
Merge Batch Results with Product Data for Complete Shopify Format
Combines AI-enhanced batch results with original product data to create complete Shopify import format
"""

import pandas as pd
import os
from datetime import datetime

def merge_batch_with_product_data():
    """Merge batch results with original product data for complete Shopify format"""
    
    print("🔄 Merging Batch Results with Product Data")
    print("=" * 60)
    
    # Load batch results
    batch_file = "data/results/steele_batch_processed_20250705_145447.csv"
    
    if not os.path.exists(batch_file):
        print(f"❌ Batch results file not found: {batch_file}")
        return
    
    print(f"📁 Loading batch results: {batch_file}")
    batch_df = pd.read_csv(batch_file)
    print(f"   📊 Batch results: {len(batch_df):,} products")
    
    # Load original product data  
    product_file = "data/processed/steele_processed_complete.csv"
    
    if not os.path.exists(product_file):
        print(f"❌ Product data file not found: {product_file}")
        return
    
    print(f"📁 Loading product data: {product_file}")
    product_df = pd.read_csv(product_file, low_memory=False)
    print(f"   📊 Product records: {len(product_df):,} records")
    
    # Get unique products from product data (one per StockCode)
    print("🔍 Getting unique products from product data...")
    unique_products = product_df.groupby('StockCode').first().reset_index()
    print(f"   📊 Unique products: {len(unique_products):,}")
    
    # Merge batch results with product data
    print("🔗 Merging batch results with product data...")
    merged_df = pd.merge(
        batch_df, 
        unique_products[['StockCode', 'Product Name', 'Description', 'StockUom', 'UPC Code', 'MAP', 'Dealer Price']], 
        on='StockCode', 
        how='left'
    )
    
    print(f"   📊 Merged records: {len(merged_df):,}")
    
    # Check for missing product data
    missing_products = merged_df[merged_df['Product Name'].isna()]
    if len(missing_products) > 0:
        print(f"⚠️  Warning: {len(missing_products)} products missing product data")
    
    # Create complete Shopify format
    print("🏗️  Creating complete Shopify format...")
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    shopify_records = []
    
    for _, row in merged_df.iterrows():
        stock_code = row['StockCode']
        product_name = row['Product Name'] if pd.notna(row['Product Name']) else f"Product {stock_code}"
        description = row['Description'] if pd.notna(row['Description']) else "Quality automotive part"
        
        # Use AI-generated tags if available, otherwise create basic tag
        if pd.notna(row['generated_tags']) and row['generated_tags'] != '':
            tags = row['generated_tags']
        else:
            tags = f"StockCode_{stock_code}"
        
        # Create meta title and description
        meta_title = f"{product_name} - Quality Automotive Part"[:60]
        meta_description = f"Quality automotive {product_name.lower()}. {description[:100]}..."[:160]
        
        # Create complete Shopify record
        shopify_record = {
            'ID': '',
            'Command': 'MERGE',
            'Title': product_name,
            'Body HTML': description,
            'Vendor': 'Steele',
            'Tags': tags,
            'Tags Command': 'MERGE',
            'Category: ID': '',
            'Category: Name': '',
            'Category': '',
            'Custom Collections': 'Accessories',
            'Smart Collections': '',
            'Image Type': '',
            'Image Src': '',
            'Image Command': 'MERGE',
            'Image Position': '1',
            'Image Width': '',
            'Image Height': '',
            'Image Alt Text': product_name,
            'Variant Inventory Item ID': '',
            'Variant ID': '',
            'Variant Command': 'MERGE',
            'Option1 Name': '',
            'Option1 Value': '',
            'Option2 Name': '',
            'Option2 Value': '',
            'Option3 Name': '',
            'Option3 Value': '',
            'Variant Position': '1',
            'Variant SKU': stock_code,
            'Variant Barcode': '',
            'Variant Image': '',
            'Variant Weight': '',
            'Variant Weight Unit': '',
            'Variant Price': row['MAP'] if pd.notna(row['MAP']) else '',
            'Variant Compare At Price': '',
            'Variant Taxable': 'True',
            'Variant Tax Code': '',
            'Variant Inventory Tracker': 'shopify',
            'Variant Inventory Policy': 'deny',
            'Variant Fulfillment Service': 'manual',
            'Variant Requires Shipping': 'True',
            'Variant Inventory Qty': '0',
            'Variant Inventory Adjust': '',
            'Variant Cost': row['Dealer Price'] if pd.notna(row['Dealer Price']) else '',
            'Variant HS Code': '',
            'Variant Country of Origin': '',
            'Variant Province of Origin': '',
            'Metafield: title_tag [string]': meta_title,
            'Metafield: description_tag [string]': meta_description,
            'Metafield: custom.engine_types [list.single_line_text_field]': '',
            'Metafield: mm-google-shopping.custom_product [boolean]': '',
            'Variant Metafield: mm-google-shopping.custom_label_4 [single_line_text_field]': '',
            'Variant Metafield: mm-google-shopping.custom_label_3 [single_line_text_field]': '',
            'Variant Metafield: mm-google-shopping.custom_label_2 [single_line_text_field]': '',
            'Variant Metafield: mm-google-shopping.custom_label_1 [single_line_text_field]': '',
            'Variant Metafield: mm-google-shopping.custom_label_0 [single_line_text_field]': '',
            'Variant Metafield: mm-google-shopping.size_system [single_line_text_field]': '',
            'Variant Metafield: mm-google-shopping.size_type [single_line_text_field]': '',
            'Variant Metafield: mm-google-shopping.mpn [single_line_text_field]': stock_code,
            'Variant Metafield: mm-google-shopping.gender [single_line_text_field]': '',
            'Variant Metafield: mm-google-shopping.condition [single_line_text_field]': 'new',
            'Variant Metafield: mm-google-shopping.age_group [single_line_text_field]': '',
            'Variant Metafield: harmonized_system_code [string]': '',
            'Metafield: mm-google-shopping.mpn [single_line_text_field]': stock_code,
            'ai_extracted_years': row['ai_extracted_years'] if pd.notna(row['ai_extracted_years']) else '',
            'ai_extracted_make': row['ai_extracted_makes'] if pd.notna(row['ai_extracted_makes']) else '',
            'ai_extracted_model': row['ai_extracted_models'] if pd.notna(row['ai_extracted_models']) else '',
            'ai_confidence': row['ai_confidence'] if pd.notna(row['ai_confidence']) else '',
            'ai_reasoning': '',
            'generated_tags': row['generated_tags'] if pd.notna(row['generated_tags']) else '',
            'extraction_error': row['extraction_error'] if pd.notna(row['extraction_error']) else ''
        }
        
        shopify_records.append(shopify_record)
    
    # Convert to DataFrame
    shopify_df = pd.DataFrame(shopify_records)
    
    # Convert all values to strings for proper CSV formatting
    shopify_df = shopify_df.astype(str)
    
    # Save complete Shopify format
    output_file = f"data/results/complete_shopify_format_{timestamp}.csv"
    shopify_df.to_csv(output_file, index=False)
    
    print(f"✅ Complete Shopify format created!")
    print(f"   📁 Output file: {output_file}")
    print(f"   📊 Total products: {len(shopify_df):,}")
    print(f"   📊 File size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
    
    # Show sample of the data
    print("\n🔍 Sample of complete data:")
    print(f"   Title: {shopify_df.iloc[0]['Title']}")
    print(f"   Body HTML: {shopify_df.iloc[0]['Body HTML'][:100]}...")
    print(f"   Tags: {shopify_df.iloc[0]['Tags'][:100]}...")
    print(f"   Variant SKU: {shopify_df.iloc[0]['Variant SKU']}")
    print(f"   Variant Price: {shopify_df.iloc[0]['Variant Price']}")
    
    return output_file

if __name__ == "__main__":
    merge_batch_with_product_data() 