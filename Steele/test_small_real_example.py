#!/usr/bin/env python3
"""
Small focused test of the complete two-pass AI extraction on real Steele data.
This will test 3 problematic products with real OpenAI API calls.
"""

import pandas as pd
import os
from datetime import datetime
from utils.ai_fitment_extractor import SteeleAIFitmentExtractor

def main():
    print("🧪 Small Real Two-Pass AI Extraction Test")
    print("=" * 50)
    
    # Create a small sample of the most problematic entries
    test_data = {
        'StockCode': [
            '10-0108-45',  # Independent (1920-1929) 
            '10-0127-52',  # Independent (1920-1924)
            '10-0130-108'  # Street Rod/Custom Build
        ],
        'Product Name': [
            'Glass weatherstrip kit',
            'Windshield to Cowl Weatherstrip', 
            'Accelerator Pedal Pad'
        ],
        'Body HTML': [
            'Running Board Step Pad. Made from top quality rubber to ensure durability. This part is compatible with models built by Independent (1920 - 1929) automobile manufacturers.',
            'This Windshield to Cowl Weatherstrip comes in a thirty-six inch (36") strip and seals the lower windshield frame to the cowl of the car. This generic part is known to fit Independent vehicle models produced from 1920 - 1924.',
            'Steele Rubber Products has the best quality Accelerator Pedal Pads for your Street Rod or Custom Build project. These Accelerator Pads are made of top-quality EPDM rubber with a high level of fit and finish for the Builder\'s creative application and use.'
        ],
        'generated_tags': [
            '0_Unknown_UNKNOWN',  # Current problematic tags
            '0_Unknown_UNKNOWN',
            '0_Unknown_UNKNOWN'
        ]
    }
    
    # Create DataFrame
    df = pd.DataFrame(test_data)
    
    print(f"📋 Testing {len(df)} problematic products:")
    for idx, row in df.iterrows():
        print(f"  {idx+1}. {row['StockCode']}: {row['Product Name']}")
        description_preview = row['Body HTML'][:80] + "..." if len(row['Body HTML']) > 80 else row['Body HTML']
        print(f"     Description: {description_preview}")
        print(f"     Current tags: {row['generated_tags']}")
        print()
    
    # Initialize the AI fitment extractor
    print("🤖 Initializing AI Fitment Extractor...")
    extractor = SteeleAIFitmentExtractor()
    
    # Process the batch with our two-pass approach
    print("🚀 Running Two-Pass AI Extraction...")
    print("   Pass 1: AI extracts years, make, model")
    print("   Pass 2: Golden Master expansion and validation")
    print()
    
    try:
        # Use the real two-pass processing method
        results_df = extractor.process_unknown_skus_batch_with_expansion(df)
        
        # Generate timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"data/results/small_real_test_{timestamp}.csv"
        
        # Save results
        results_df.to_csv(output_file, index=False)
        
        print("✅ Two-Pass AI Processing Complete!")
        print("=" * 50)
        
        # Show results summary
        print("📊 RESULTS SUMMARY:")
        for idx, row in results_df.iterrows():
            print(f"\n[{idx+1}] {row['StockCode']}: {row['Product Name']}")
            print(f"    🤖 AI Extracted:")
            print(f"       Years: {row.get('ai_extracted_years', 'N/A')}")
            print(f"       Make: {row.get('ai_extracted_make', 'N/A')}")
            print(f"       Model: {row.get('ai_extracted_model', 'N/A')}")
            print(f"       Confidence: {row.get('ai_confidence', 'N/A')}")
            
            # Count and show vehicle tags
            tags = row.get('generated_tags', '0_Unknown_UNKNOWN')
            if tags and tags != '0_Unknown_UNKNOWN':
                tag_list = [tag.strip() for tag in tags.split(',')]
                print(f"    🏷️  Generated Tags: {len(tag_list)} vehicle tags")
                if len(tag_list) <= 5:
                    print(f"       Tags: {', '.join(tag_list)}")
                else:
                    print(f"       First 5: {', '.join(tag_list[:5])}...")
            else:
                print(f"    🏷️  Generated Tags: {tags}")
        
        print("\n" + "=" * 50)
        print(f"📁 OUTPUT SAVED TO: {output_file}")
        print(f"📍 Full path: {os.path.abspath(output_file)}")
        print("=" * 50)
        
        # Verify the file exists
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"✅ File confirmed: {file_size} bytes")
            
            # Show first few lines of the CSV
            print(f"\n📄 Preview of {output_file}:")
            with open(output_file, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines[:3]):  # Show header + first 2 data rows
                    print(f"   {i+1}: {line.strip()}")
                if len(lines) > 3:
                    print(f"   ... and {len(lines)-3} more lines")
        else:
            print("❌ Error: Output file was not created")
        
        return output_file
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    output_file = main()
    if output_file:
        print(f"\n🎯 SUCCESS! Check your results in: {output_file}")
    else:
        print("\n❌ Test failed. Check the error messages above.") 