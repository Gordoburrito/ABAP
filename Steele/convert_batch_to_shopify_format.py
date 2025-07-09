#!/usr/bin/env python3
"""
Convert Batch Results to Shopify Product Import Format
Converts steele_batch_processed_*.csv to the exact format shown by user
"""

import pandas as pd
import os
from datetime import datetime

def convert_batch_to_shopify_format():
    """Convert batch results to Shopify product import format"""
    
    print("🚀 Converting Batch Results to Shopify Format")
    print("=" * 60)
    
    # Load the batch results CSV file
    batch_results_file = "data/results/steele_batch_processed_20250705_145447.csv"
    
    if not os.path.exists(batch_results_file):
        print(f"❌ Error: Batch results file not found: {batch_results_file}")
        return
    
    print(f"📁 Loading batch results from: {batch_results_file}")
    
    try:
        # Read the batch results
        df = pd.read_csv(batch_results_file)
        print(f"📊 Loaded {len(df)} products from batch results")
        
        # Create the Shopify format DataFrame with all required columns
        shopify_columns = [
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
        
        # Create empty DataFrame with all required columns
        shopify_df = pd.DataFrame(columns=shopify_columns)
        
        # Process each product from batch results
        for idx, row in df.iterrows():
            stock_code = row['StockCode']
            
            # Map the data to Shopify format
            shopify_row = {
                'ID': '',  # Empty for new products
                'Command': 'MERGE',
                'Title': f'Steele Part {stock_code}',  # Generic title, can be improved
                'Body HTML': f'Steele automotive part {stock_code}. Professional grade replacement part.',
                'Vendor': 'Steele',
                'Tags': row['generated_tags'] if pd.notna(row['generated_tags']) else '',
                'Tags Command': 'MERGE',
                'Category: ID': '',
                'Category: Name': '',
                'Category': 'Accessories',
                'Custom Collections': '',
                'Smart Collections': '',
                'Image Type': '',
                'Image Src': '',
                'Image Command': 'MERGE',
                'Image Position': '1',
                'Image Width': '',
                'Image Height': '',
                'Image Alt Text': f'Steele Part {stock_code}',
                'Variant Inventory Item ID': '',
                'Variant ID': '',
                'Variant Command': 'MERGE',
                'Option1 Name': '',
                'Option1 Value': '',
                'Option2 Name': '',
                'Option2 Value': '',
                'Option3 Name': '',
                'Option3 Value': '',
                'Variant Position': 1,
                'Variant SKU': stock_code,
                'Variant Barcode': '',
                'Variant Image': '',
                'Variant Weight': '',
                'Variant Weight Unit': '',
                'Variant Price': '',  # Price needs to be set
                'Variant Compare At Price': '',
                'Variant Taxable': True,
                'Variant Tax Code': '',
                'Variant Inventory Tracker': 'shopify',
                'Variant Inventory Policy': 'deny',
                'Variant Fulfillment Service': 'manual',
                'Variant Requires Shipping': True,
                'Variant Inventory Qty': 0,
                'Variant Inventory Adjust': '',
                'Variant Cost': '',
                'Variant HS Code': '',
                'Variant Country of Origin': '',
                'Variant Province of Origin': '',
                'Metafield: title_tag [string]': f'Steele Part {stock_code}',
                'Metafield: description_tag [string]': f'Quality automotive Steele Part {stock_code}.',
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
                'ai_reasoning': '',  # Can be extracted from raw_ai_response if needed
                'generated_tags': row['generated_tags'] if pd.notna(row['generated_tags']) else '',
                'extraction_error': row['extraction_error'] if pd.notna(row['extraction_error']) else ''
            }
            
            # Add row to DataFrame
            shopify_df = pd.concat([shopify_df, pd.DataFrame([shopify_row])], ignore_index=True)
        
        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"data/results/shopify_format_batch_{timestamp}.csv"
        
        # Convert all values to strings for proper CSV formatting
        shopify_df = shopify_df.astype(str)
        
        # Save to CSV
        shopify_df.to_csv(output_file, index=False)
        
        print(f"✅ Successfully converted {len(shopify_df)} products to Shopify format")
        print(f"📁 Output saved to: {output_file}")
        
        # Show sample of first few products
        print("\n📋 Sample of converted products:")
        print("-" * 60)
        sample_cols = ['Command', 'Title', 'Vendor', 'Tags', 'Variant SKU', 'ai_extracted_make', 'ai_extracted_model']
        sample_data = shopify_df[sample_cols].head(3)
        for idx, row in sample_data.iterrows():
            print(f"SKU: {row['Variant SKU']}")
            print(f"  Title: {row['Title']}")
            print(f"  Tags: {row['Tags']}")
            print(f"  AI Make: {row['ai_extracted_make']}")
            print(f"  AI Model: {row['ai_extracted_model']}")
            print()
        
        return output_file
        
    except Exception as e:
        print(f"❌ Error processing batch results: {e}")
        return None

if __name__ == "__main__":
    convert_batch_to_shopify_format() 