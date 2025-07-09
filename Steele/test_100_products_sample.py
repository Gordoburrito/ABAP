#!/usr/bin/env python3
"""
Test first 100 products with the corrected two-pass AI extraction and save results to CSV.
"""

import pandas as pd
import os
from datetime import datetime
from utils.ai_fitment_extractor import SteeleAIFitmentExtractor

def main():
    print("🧪 Testing First 100 Products with Two-Pass AI Extraction")
    print("=" * 60)
    
    # Read the first 100 products CSV file
    input_file = "data/results/first_100_products_20250603_110423.csv"
    
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    print(f"📊 Loaded {len(df)} products from {input_file}")
    
    # Take all 100 products
    sample_df = df.copy()
    print(f"🎯 Processing all {len(sample_df)} products...")
    
    # Initialize the AI extractor
    extractor = SteeleAIFitmentExtractor()
    
    # Process the batch
    print("🚀 Starting AI extraction and expansion...")
    results_df = extractor.process_unknown_skus_batch_with_expansion(sample_df)
    
    # Save results with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"data/results/first_100_products_corrected_{timestamp}.csv"
    
    results_df.to_csv(output_file, index=False)
    print(f"✅ Results saved to: {output_file}")
    
    # Show summary
    total_products = len(results_df)
    successful_extractions = len(results_df[results_df['generated_tags'] != '0_Unknown_UNKNOWN'])
    unknown_tags = len(results_df[results_df['generated_tags'] == '0_Unknown_UNKNOWN'])
    
    print(f"\n📈 SUMMARY:")
    print(f"   Total products processed: {total_products}")
    print(f"   ✅ Successful extractions: {successful_extractions}")
    print(f"   ❌ Unknown tags: {unknown_tags}")
    print(f"   Success rate: {(successful_extractions/total_products)*100:.1f}%")
    
    # Show some examples of successful extractions
    successful_df = results_df[results_df['generated_tags'] != '0_Unknown_UNKNOWN']
    if len(successful_df) > 0:
        print(f"\n🎉 SUCCESSFUL EXAMPLES:")
        for i, row in successful_df.head(3).iterrows():
            stock_code = row.get('Variant SKU', 'N/A')
            title = row.get('Title', 'N/A')
            years = row.get('ai_extracted_years', 'N/A')
            make = row.get('ai_extracted_make', 'N/A')
            tags_count = len(row['generated_tags'].split(', ')) if row['generated_tags'] != '0_Unknown_UNKNOWN' else 0
            print(f"   • {stock_code}: {title}")
            print(f"     Years: {years}, Make: {make}")
            print(f"     Generated {tags_count} vehicle tags")
            print()

if __name__ == "__main__":
    main() 