#!/usr/bin/env python3
"""
Test all 100 products with the corrected two-pass AI extraction and Tags replacement.
This will replace the Tags column with generated_tags when Tags is 0_Unknown_UNKNOWN.
"""

import pandas as pd
import os
from datetime import datetime
from utils.ai_fitment_extractor import SteeleAIFitmentExtractor

def main():
    print("🧪 Testing 100 Products with Two-Pass AI Extraction + Tags Replacement")
    print("=" * 70)
    
    # Read the first 100 products CSV file
    input_file = "data/results/first_100_products_20250603_110423.csv"
    
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    print(f"📊 Loaded {len(df)} products from {input_file}")
    
    # Count products with 0_Unknown_UNKNOWN tags
    unknown_count = len(df[df['Tags'] == '0_Unknown_UNKNOWN'])
    print(f"🎯 Found {unknown_count} products with 0_Unknown_UNKNOWN tags to potentially replace")
    
    # Initialize AI extractor
    try:
        extractor = SteeleAIFitmentExtractor()
        print("✅ AI Fitment Extractor initialized successfully")
    except ValueError as e:
        print(f"❌ Error initializing extractor: {e}")
        return
    
    # Process all 100 products
    print(f"\n🔄 Processing all {len(df)} products...")
    results_df = extractor.process_unknown_skus_batch_with_expansion(df)
    
    # Save results with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"data/results/first_100_products_WITH_TAGS_REPLACEMENT_{timestamp}.csv"
    
    results_df.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to: {output_file}")
    
    # Analyze results
    original_unknown = len(df[df['Tags'] == '0_Unknown_UNKNOWN'])
    final_unknown = len(results_df[results_df['Tags'] == '0_Unknown_UNKNOWN'])
    replaced_count = original_unknown - final_unknown
    
    # Count products with generated tags
    has_generated_tags = len(results_df[results_df['generated_tags'] != '0_Unknown_UNKNOWN'])
    
    print(f"\n📊 FINAL SUMMARY:")
    print(f"• Total products processed: {len(results_df)}")
    print(f"• Original products with 0_Unknown_UNKNOWN: {original_unknown}")
    print(f"• Final products with 0_Unknown_UNKNOWN: {final_unknown}")
    print(f"• Tags successfully replaced: {replaced_count}")
    print(f"• Products with AI-generated tags: {has_generated_tags}")
    print(f"• Tags replacement success rate: {replaced_count/original_unknown*100:.1f}%")
    
    # Show some examples of successful replacements
    successful_replacements = results_df[
        (results_df['Tags'] != '0_Unknown_UNKNOWN') & 
        (results_df['generated_tags'] != '0_Unknown_UNKNOWN')
    ]
    
    if len(successful_replacements) > 0:
        print(f"\n🏆 SUCCESSFUL TAG REPLACEMENTS ({len(successful_replacements)} products):")
        for idx, row in successful_replacements.head(3).iterrows():
            stock_code = row.get('Variant SKU', row.get('StockCode', 'N/A'))
            title = row.get('Title', row.get('Product Name', 'N/A'))
            tags_count = len(str(row['Tags']).split(', ')) if row['Tags'] != '0_Unknown_UNKNOWN' else 0
            print(f"• {stock_code}: {title[:50]}...")
            print(f"  → Generated {tags_count} vehicle tags")

if __name__ == "__main__":
    main() 