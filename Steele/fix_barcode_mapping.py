#!/usr/bin/env python3
"""
Fix Barcode Mapping in Shopify Format
Corrects the mapping where barcode should be vendor SKU number
"""

import pandas as pd
import os
from datetime import datetime

def fix_barcode_mapping():
    """Fix the barcode and vendor SKU mapping in the clean Shopify format"""
    
    print("🔧 Fixing Barcode Mapping in Shopify Format")
    print("=" * 60)
    
    # Load the clean file
    input_file = "data/results/clean_shopify_format_20250705_162334.csv"
    
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return
    
    print(f"📁 Loading clean file: {input_file}")
    df = pd.read_csv(input_file)
    print(f"   📊 Total products: {len(df):,}")
    
    # Show current mapping
    print(f"\n📋 Current mapping:")
    print(f"   Variant SKU: {df.iloc[0]['Variant SKU']}")
    print(f"   Variant Barcode: '{df.iloc[0]['Variant Barcode']}'")
    
    # Fix the mapping
    print(f"\n🔧 Fixing barcode mapping...")
    print(f"   Rule: Barcode = Vendor SKU Number")
    
    # Move StockCode from Variant SKU to Variant Barcode
    df['Variant Barcode'] = df['Variant SKU']  # StockCode goes to barcode
    df['Variant SKU'] = ''  # Clear the SKU field
    
    # Show fixed mapping
    print(f"\n✅ Fixed mapping:")
    print(f"   Variant SKU: '{df.iloc[0]['Variant SKU']}'")
    print(f"   Variant Barcode: {df.iloc[0]['Variant Barcode']}")
    
    # Save corrected version
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"data/results/corrected_shopify_format_{timestamp}.csv"
    df.to_csv(output_file, index=False)
    
    # Show results
    print(f"\n✅ Corrected Shopify format created!")
    print(f"   📁 Output file: {output_file}")
    print(f"   📊 Products: {len(df):,}")
    print(f"   📊 File size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
    
    # Show sample of corrected data
    print(f"\n🔍 Sample corrected data:")
    print(f"   Title: {df.iloc[0]['Title']}")
    print(f"   Variant SKU: '{df.iloc[0]['Variant SKU']}'")
    print(f"   Variant Barcode: {df.iloc[0]['Variant Barcode']}")
    print(f"   Variant Price: {df.iloc[0]['Variant Price']}")
    
    print(f"\n📋 Mapping Summary:")
    print(f"   ✅ Variant Barcode: Contains StockCode (vendor SKU)")
    print(f"   ✅ Variant SKU: Empty (as intended)")
    print(f"   ✅ All other fields preserved")
    
    return output_file

if __name__ == "__main__":
    fix_barcode_mapping() 