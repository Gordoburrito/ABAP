#!/usr/bin/env python3
"""
Fast processor script for the first_100_stock_codes_sample.csv file.
Skips AI and golden master validation for quick testing.
"""

import sys
import os
import time
import pandas as pd
from pathlib import Path

# Add the utils directory to the path
sys.path.append('utils')

from steele_data_transformer import SteeleDataTransformer, ProductData

def quick_process_steele_data(input_file: str, output_file: str):
    """Quick processing without AI or golden master validation."""
    
    print(f"🔄 Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"✅ Loaded {len(df)} records")
    
    print("🔄 Converting to standard format...")
    products = []
    
    for idx, row in df.iterrows():
        product = ProductData(
            title=str(row['Product Name']),
            year_min=str(row['Year']) if pd.notna(row['Year']) else "Unknown",
            year_max=str(row['Year']) if pd.notna(row['Year']) else "Unknown",
            make=str(row['Make']) if pd.notna(row['Make']) else "NONE",
            model=str(row['Model']) if pd.notna(row['Model']) else "NONE",
            mpn=str(row.get('PartNumber', row['StockCode'])),
            cost=float(row['Dealer Price']) if pd.notna(row['Dealer Price']) else 0.0,
            price=float(row['MAP']) if pd.notna(row['MAP']) else 0.0,
            body_html=str(row['Description']),
            golden_validated=False,
            fitment_source="vendor_provided",
            processing_method="template_based"
        )
        products.append(product)
    
    print(f"✅ Converted {len(products)} products")
    
    print("🔄 Creating Shopify format...")
    # Simple format - just the essential columns
    shopify_data = []
    
    for product in products:
        # Create vehicle tag
        if product.make != "NONE" and product.model != "NONE" and product.year_min != "Unknown":
            vehicle_tag = f"{product.year_min}_{product.make.replace(' ', '_')}_{product.model.replace(' ', '_')}"
        else:
            vehicle_tag = ""
        
        record = {
            'Title': product.title,
            'Body HTML': product.body_html,
            'Vendor': 'Steele',
            'Product Type': 'Automotive Part',
            'Tags': vehicle_tag,
            'Variant SKU': product.mpn,
            'Variant Price': product.price,
            'Variant Cost': product.cost,
            'SEO Title': f"{product.title} - {product.year_min} {product.make} {product.model}" if vehicle_tag else product.title,
            'SEO Description': f"Quality {product.title.lower()} for {product.year_min} {product.make} {product.model}" if vehicle_tag else f"Quality {product.title.lower()}"
        }
        shopify_data.append(record)
    
    # Convert to DataFrame
    shopify_df = pd.DataFrame(shopify_data)
    
    print("🔄 Consolidating by SKU...")
    # Group by SKU and combine tags
    consolidated = []
    for sku, group in shopify_df.groupby('Variant SKU'):
        # Take the first record
        record = group.iloc[0].to_dict()
        
        # Combine all unique tags
        all_tags = [tag for tag in group['Tags'].dropna() if tag != '']
        unique_tags = list(dict.fromkeys(all_tags))  # Remove duplicates while preserving order
        record['Tags'] = ', '.join(unique_tags)
        
        consolidated.append(record)
    
    final_df = pd.DataFrame(consolidated)
    
    print(f"✅ Consolidated to {len(final_df)} unique products")
    
    # Save results
    final_df.to_csv(output_file, index=False)
    print(f"💾 Results saved to: {output_file}")
    
    return final_df

def main():
    """Run the fast processor."""
    
    print("🚀 STEELE PROCESSOR - FAST MODE")
    print("   (No AI, No Golden Master Validation)")
    print("=" * 60)
    
    try:
        # Input and output files
        input_file = 'data/samples/first_100_stock_codes_sample.csv'
        
        # Create output directory
        output_dir = Path('data/results')
        output_dir.mkdir(exist_ok=True)
        
        # Output file with timestamp
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f'first_100_products_fast_{timestamp}.csv'
        
        # Process the data
        start_time = time.time()
        final_df = quick_process_steele_data(input_file, str(output_file))
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        print("\n" + "=" * 60)
        print("✅ PROCESSING COMPLETE")
        print(f"📁 Results saved to: {output_file}")
        print(f"📊 Processed {len(final_df)} unique products")
        print(f"⏱️  Processing time: {processing_time:.2f} seconds")
        print(f"🚀 Rate: {len(final_df)/processing_time:.1f} products/second")
        print("=" * 60)
        
        # Show sample of results
        print("\n📋 SAMPLE RESULTS:")
        for i, row in final_df.head(3).iterrows():
            print(f"  {i+1}. {row['Title']} - SKU: {row['Variant SKU']}")
            print(f"     Tags: {row['Tags']}")
            print(f"     Price: ${row['Variant Price']}")
        
        return str(output_file)
        
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 