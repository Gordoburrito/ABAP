#!/usr/bin/env python3
"""
Test script to verify that Tags column replacement is working correctly.
This will test a few products and show before/after Tags column values.
"""

import pandas as pd
import os
from datetime import datetime
from utils.ai_fitment_extractor import SteeleAIFitmentExtractor

def main():
    print("🧪 Testing Tags Column Replacement")
    print("=" * 50)
    
    # Read the first 100 products CSV file
    input_file = "data/results/first_100_products_20250603_110423.csv"
    
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    print(f"📊 Loaded {len(df)} products from {input_file}")
    
    # Take only the first 5 products that have 0_Unknown_UNKNOWN tags
    unknown_products = df[df['Tags'] == '0_Unknown_UNKNOWN'].head(5).copy()
    print(f"🎯 Found {len(unknown_products)} products with 0_Unknown_UNKNOWN tags")
    
    if len(unknown_products) == 0:
        print("❌ No products with 0_Unknown_UNKNOWN tags found")
        return
    
    print("\n📋 BEFORE Processing:")
    print("-" * 60)
    for idx, row in unknown_products.iterrows():
        stock_code = row.get('Variant SKU', 'N/A')
        title = row.get('Title', 'N/A')
        tags = row.get('Tags', 'N/A')
        print(f"• {stock_code}: {title[:50]}...")
        print(f"  Tags: {tags}")
    
    # Initialize AI extractor
    try:
        extractor = SteeleAIFitmentExtractor()
        print("\n✅ AI Fitment Extractor initialized successfully")
    except ValueError as e:
        print(f"❌ Error initializing extractor: {e}")
        return
    
    # Process the sample
    print(f"\n🔄 Processing {len(unknown_products)} products...")
    results_df = extractor.process_unknown_skus_batch_with_expansion(unknown_products)
    
    print("\n📋 AFTER Processing:")
    print("-" * 60)
    for idx, row in results_df.iterrows():
        stock_code = row.get('Variant SKU', row.get('StockCode', 'N/A'))
        title = row.get('Title', row.get('Product Name', 'N/A'))
        original_tags = row.get('Tags', 'N/A')
        generated_tags = row.get('generated_tags', 'N/A')
        
        print(f"• {stock_code}: {title[:50]}...")
        print(f"  Tags: {original_tags[:100]}{'...' if len(str(original_tags)) > 100 else ''}")
        print(f"  Generated: {generated_tags[:100]}{'...' if len(str(generated_tags)) > 100 else ''}")
        
        if original_tags != '0_Unknown_UNKNOWN':
            print(f"  ✅ TAGS REPLACED!")
        else:
            print(f"  ❌ Tags not replaced")
        print()
    
    # Save results with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"data/results/tags_replacement_test_{timestamp}.csv"
    
    results_df.to_csv(output_file, index=False)
    print(f"💾 Results saved to: {output_file}")
    
    # Summary
    replaced_count = len(results_df[results_df['Tags'] != '0_Unknown_UNKNOWN'])
    print(f"\n📊 SUMMARY:")
    print(f"• Total products processed: {len(results_df)}")
    print(f"• Tags successfully replaced: {replaced_count}")
    print(f"• Success rate: {replaced_count/len(results_df)*100:.1f}%")

if __name__ == "__main__":
    main() 