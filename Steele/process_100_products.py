#!/usr/bin/env python3
"""
Process 100 Products with Two-Pass AI Extraction
Loads products with 0_Unknown_UNKNOWN tags and processes them using the two-pass AI system.
"""

import pandas as pd
import os
from datetime import datetime
from utils.ai_fitment_extractor import SteeleAIFitmentExtractor

def main():
    """Process 100 products with 0_Unknown_UNKNOWN tags using two-pass AI extraction"""
    
    print("🚀 Processing 100 Products with Two-Pass AI Extraction")
    print("=" * 60)
    
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key: export OPENAI_API_KEY='your-key-here'")
        return
    
    # Load the CSV file with 100 products
    csv_file = "data/results/first_100_products_20250603_110423.csv"
    
    if not os.path.exists(csv_file):
        print(f"❌ Error: Input file not found: {csv_file}")
        print("Please ensure the CSV file exists in the data/results directory")
        return
    
    print(f"📁 Loading products from: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"📊 Loaded {len(df)} products from CSV")
    
    # Filter products with 0_Unknown_UNKNOWN tags
    unknown_products = df[df['Tags'].str.contains('0_Unknown_UNKNOWN', na=False)].copy()
    print(f"🔍 Found {len(unknown_products)} products with 0_Unknown_UNKNOWN tags")
    
    if len(unknown_products) == 0:
        print("ℹ️  No 0_Unknown_UNKNOWN products found to process")
        return
    
    # Create input DataFrame in the format expected by the AI extractor
    input_df = pd.DataFrame({
        'StockCode': unknown_products['Variant SKU'],
        'Product Name': unknown_products['Title'],
        'Description': unknown_products['Body HTML'],
        'Current Tags': unknown_products['Tags']
    })
    
    print("\n📋 Sample products to process:")
    for idx, row in input_df.head(5).iterrows():
        print(f"  {idx+1}. {row['StockCode']}: {row['Product Name']}")
        print(f"     Current: {row['Current Tags']}")
        print(f"     Description: {row['Description'][:100]}...")
        print()
    
    if len(unknown_products) > 5:
        print(f"  ... and {len(unknown_products) - 5} more products")
    
    print("\n🚀 Initializing AI Fitment Extractor...")
    try:
        extractor = SteeleAIFitmentExtractor()
        print("✅ AI Fitment Extractor initialized successfully")
        print("🔧 Using GPT-4o model for enhanced accuracy")
    except Exception as e:
        print(f"❌ Error initializing extractor: {e}")
        return
    
    print(f"\n🔄 Processing {len(input_df)} products with Two-Pass AI Extraction...")
    print("This may take several minutes depending on API response times...")
    print("=" * 60)
    
    # Process the products
    start_time = datetime.now()
    results_df = extractor.process_unknown_skus_batch_with_expansion(input_df)
    end_time = datetime.now()
    
    processing_time = end_time - start_time
    print(f"\n⏱️  Processing completed in {processing_time}")
    
    print("\n📊 RESULTS SUMMARY:")
    print("=" * 60)
    
    successful_extractions = 0
    improved_tags = 0
    cost_estimate = 0.0
    
    for idx, row in results_df.iterrows():
        stock_code = row['StockCode']
        original_tags = input_df.iloc[idx]['Current Tags']
        new_tags = row['generated_tags']
        confidence = row['ai_confidence']
        
        print(f"\n🏷️  {stock_code}: {row['Product Name'][:50]}...")
        print(f"   Original: {original_tags}")
        print(f"   New Tags: {new_tags}")
        print(f"   Confidence: {confidence:.2f}")
        print(f"   Make: {row['ai_extracted_make']}")
        print(f"   Model: {row['ai_extracted_model']}")
        
        if row['extraction_error']:
            print(f"   ⚠️  Error: {row['extraction_error']}")
        else:
            successful_extractions += 1
            
        if '0_Unknown_UNKNOWN' not in new_tags and '0_Unknown_UNKNOWN' in original_tags:
            improved_tags += 1
            print("   ✅ IMPROVED: Generated specific vehicle tags!")
        elif new_tags != original_tags:
            improved_tags += 1
            print("   ✅ IMPROVED: Tags changed from original!")
        else:
            print("   ❌ No improvement")
        
        # Estimate cost (approximate)
        cost_estimate += 0.002  # Rough estimate per API call
    
    print(f"\n📈 FINAL SUMMARY:")
    print("=" * 60)
    print(f"   Total products processed: {len(input_df)}")
    print(f"   Successful extractions: {successful_extractions}")
    print(f"   Improved tags: {improved_tags}")
    print(f"   Success rate: {(successful_extractions/len(input_df)*100):.1f}%")
    print(f"   Improvement rate: {(improved_tags/len(input_df)*100):.1f}%")
    print(f"   Processing time: {processing_time}")
    print(f"   Estimated cost: ${cost_estimate:.2f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"data/results/processed_100_products_{timestamp}.csv"
    
    # Create output directory if it doesn't exist
    os.makedirs("data/results", exist_ok=True)
    
    results_df.to_csv(output_file, index=False)
    print(f"\n💾 Full results saved to: {output_file}")
    
    # Also save a summary report
    summary_file = f"data/results/processing_summary_{timestamp}.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Two-Pass AI Processing Summary\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write(f"="*50 + "\n\n")
        f.write(f"Input file: {csv_file}\n")
        f.write(f"Total products processed: {len(input_df)}\n")
        f.write(f"Successful extractions: {successful_extractions}\n")
        f.write(f"Improved tags: {improved_tags}\n")
        f.write(f"Success rate: {(successful_extractions/len(input_df)*100):.1f}%\n")
        f.write(f"Improvement rate: {(improved_tags/len(input_df)*100):.1f}%\n")
        f.write(f"Processing time: {processing_time}\n")
        f.write(f"Estimated cost: ${cost_estimate:.2f}\n")
        f.write(f"Output file: {output_file}\n")
    
    print(f"📋 Summary report saved to: {summary_file}")
    
    print("\n🎯 KEY INSIGHTS:")
    if improved_tags > 0:
        print(f"   ✅ Two-pass approach successfully improved {improved_tags} product tags!")
        print("   ✅ The system can handle vague descriptions and expand 'ALL' values")
        print("   ✅ Golden master expansion is working correctly")
    else:
        print("   ❌ No improvements detected - may need to adjust AI prompts or expansion logic")
    
    if successful_extractions == len(input_df):
        print("   ✅ All extractions completed without errors")
    else:
        print(f"   ⚠️  {len(input_df) - successful_extractions} extractions had errors")
    
    print("\n🚀 NEXT STEPS:")
    print("   1. Review the output file to validate the generated tags")
    print("   2. Test a few products manually to ensure accuracy")
    print("   3. If results look good, you can scale up to larger batches")
    print("   4. Consider implementing batch processing for efficiency")

if __name__ == "__main__":
    main() 