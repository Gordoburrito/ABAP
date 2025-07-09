#!/usr/bin/env python3
"""
Test 20 products with the corrected two-pass AI extraction and save results to CSV.
"""

import pandas as pd
import os
from datetime import datetime
from utils.ai_fitment_extractor import SteeleAIFitmentExtractor

def main():
    print("🧪 Testing 20 Products with Two-Pass AI Extraction")
    print("=" * 60)
    
    # Read the first 100 products CSV file
    input_file = "data/results/first_100_products_20250603_110423.csv"
    
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    print(f"📊 Loaded {len(df)} products from {input_file}")
    
    # Take first 20 products
    sample_df = df.head(20).copy()
    print(f"🎯 Processing first 20 products...")
    
    # Initialize the AI extractor
    extractor = SteeleAIFitmentExtractor()
    
    # Process the sample with expansion
    print("🤖 Running two-pass AI extraction with golden master expansion...")
    results_df = extractor.process_unknown_skus_batch_with_expansion(sample_df)
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"data/results/sample_20_products_{timestamp}.csv"
    
    # Save results to CSV
    results_df.to_csv(output_file, index=False)
    
    print(f"✅ Results saved to: {output_file}")
    print(f"📊 Processed {len(results_df)} products")
    
    # Show summary of results
    print("\n📈 RESULTS SUMMARY:")
    print("=" * 50)
    
    # Count products with generated tags vs Unknown
    unknown_count = len(results_df[results_df['generated_tags'] == '0_Unknown_UNKNOWN'])
    success_count = len(results_df[results_df['generated_tags'] != '0_Unknown_UNKNOWN'])
    
    print(f"✅ Successfully processed: {success_count}")
    print(f"❌ Unknown/Failed: {unknown_count}")
    
    # Show some examples
    print("\n🔍 SAMPLE RESULTS:")
    print("-" * 50)
    
    for idx, row in results_df.head(5).iterrows():
        print(f"\nProduct: {row['Variant SKU']}")
        print(f"  Description: {row['Body HTML'][:80]}...")
        print(f"  AI Years: {row['ai_extracted_years']}")
        print(f"  AI Make: {row['ai_extracted_make']}")
        print(f"  AI Model: {row['ai_extracted_model']}")
        
        # Count tags generated
        tags = row['generated_tags']
        if tags != '0_Unknown_UNKNOWN':
            tag_count = len(tags.split(', ')) if tags else 0
            print(f"  Generated Tags: {tag_count} tags")
        else:
            print(f"  Generated Tags: {tags}")
    
    print(f"\n💾 Full results saved to: {output_file}")

if __name__ == "__main__":
    main() 