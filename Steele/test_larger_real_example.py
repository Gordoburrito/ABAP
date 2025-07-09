#!/usr/bin/env python3
"""
Larger focused test of the complete two-pass AI extraction on real Steele data.
This will test 10 problematic products with real OpenAI API calls to verify the year extraction fix.
"""

import pandas as pd
import os
from datetime import datetime
from utils.ai_fitment_extractor import SteeleAIFitmentExtractor

def main():
    print("🧪 LARGER Real Two-Pass AI Extraction Test - Year Fix Verification")
    print("=" * 70)
    
    # Create a larger sample of problematic entries from the CSV
    test_data = {
        'StockCode': [
            '10-0108-45',  # Independent (1920-1929) - should extract 1920-1929
            '10-0127-52',  # Independent (1920-1924) - should extract 1920-1924  
            '10-0130-108', # Street Rod/Custom Build - no years mentioned
            '10-0128-35',  # Sterns-Knight (1912-1915) - should extract 1912-1915
            '10-0129-35',  # Sterns-Knight (1912-1915) - should extract 1912-1915
            '10-0130-40',  # Sterns-Knight (1928-1929) - should extract 1928-1929
            '10-0138-67',  # Sterns-Knight (1913-1925) + Oldsmobile (1916) - mixed
            '10-0145-70',  # Independent (1920-1932) - should extract 1920-1932
            '10-0145-89',  # Independent (1920-1932) - should extract 1920-1932
            '10-0180-22'   # Minerva (1928-1931) - should extract 1928-1931
        ],
        'Product Name': [
            'Glass weatherstrip kit',
            'Windshield to Cowl Weatherstrip',
            'Accelerator Pedal Pad',
            'Pad, rear axle rebound', 
            'Pad, front axle rebound',
            'Pad, accelerator pedal',
            'Pad, stirrup, fldg top bows',
            'Check straps, loop type BLACK',
            'Check straps, loop type BROWN',
            'Pedal pads, Minerva'
        ],
        'Body HTML': [
            'Running Board Step Pad. Made from top quality rubber to ensure durability. This part is compatible with models built by Independent (1920 - 1929) automobile manufacturers.',
            'This Windshield to Cowl Weatherstrip comes in a thirty-six inch (36") strip and seals the lower windshield frame to the cowl of the car. This generic part is known to fit Independent vehicle models produced from 1920 - 1924.',
            'Steele Rubber Products has the best quality Accelerator Pedal Pads for your Street Rod or Custom Build project. These Accelerator Pads are made of top-quality EPDM rubber with a high level of fit and finish for the Builder\'s creative application and use.',
            'Axle Rebound Pad - Rear. Made from top quality rubber to ensure durability. This part is compatible with (1912 - 1915) Sterns-Knight models.',
            'Axle Rebound Pad - Rear. Made from top quality rubber to ensure durability. This part is compatible with (1912 - 1915) Sterns-Knight models.',
            'Accelerator Pedal Pad. Made from top quality rubber to ensure durability. This part is compatible with (1928 - 1929) Sterns-Knight models.',
            'Stirrup Pad for folding top bows. This part is compatible with certain (1913-1925) Sterns-Knight and (1916) Oldsmobile models.',
            'Door Check Straps - loop type. Made from top quality Black rubber to ensure durability. This part is compatible with models built by Independent (1920 - 1932) automobile manufacturers.',
            'Door Check Straps - loop type. Made from top quality Brown rubber to ensure durability. This part is compatible with models built by Independent (1920 - 1932) automobile manufacturers.',
            'Pads, brake and clutch pedal. Good copy of original with bonded and riveted steel back, to be retained by tabs. For (1928 - 1931) Minerva Motors vehicle models.'
        ],
        'generated_tags': [
            '0_Unknown_UNKNOWN',  # Current problematic tags
            '0_Unknown_UNKNOWN',
            '0_Unknown_UNKNOWN',
            '0_Unknown_UNKNOWN',
            '0_Unknown_UNKNOWN',
            '0_Unknown_UNKNOWN',
            '0_Unknown_UNKNOWN',
            '0_Unknown_UNKNOWN',
            '0_Unknown_UNKNOWN',
            '0_Minerva_UNKNOWN'
        ]
    }
    
    # Create DataFrame
    df = pd.DataFrame(test_data)
    
    print(f"📋 Testing {len(df)} problematic products with YEAR EXTRACTION FOCUS:")
    for idx, row in df.iterrows():
        print(f"  {idx+1}. {row['StockCode']}: {row['Product Name']}")
        description_preview = row['Body HTML'][:100] + "..." if len(row['Body HTML']) > 100 else row['Body HTML']
        print(f"     Description: {description_preview}")
        print(f"     Current tags: {row['generated_tags']}")
        print()
    
    # Initialize the AI fitment extractor
    print("🤖 Initializing AI Fitment Extractor with CORRECTED PROMPT...")
    extractor = SteeleAIFitmentExtractor()
    
    # Process the batch with our two-pass approach
    print("🚀 Running Two-Pass AI Extraction with YEAR FIX...")
    print("   Pass 1: AI extracts years, make, model (FIXED YEAR EXTRACTION)")
    print("   Pass 2: Golden Master expansion and validation")
    print()
    
    try:
        # Use the real two-pass processing method
        results_df = extractor.process_unknown_skus_batch_with_expansion(df)
        
        # Generate timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"data/results/larger_real_test_YEAR_FIX_{timestamp}.csv"
        
        # Save results
        results_df.to_csv(output_file, index=False)
        
        print("✅ Two-Pass AI Processing Complete!")
        print("=" * 70)
        
        # Show detailed results analysis focusing on year extraction
        print("📊 DETAILED YEAR EXTRACTION ANALYSIS:")
        print("=" * 70)
        
        for idx, row in results_df.iterrows():
            print(f"\n[{idx+1}] {row['StockCode']}: {row['Product Name']}")
            
            # Extract expected years from description for comparison
            description = row['Body HTML']
            expected_years = []
            if "(1920 - 1929)" in description or "(1920-1929)" in description:
                expected_years = ["1920", "1921", "1922", "1923", "1924", "1925", "1926", "1927", "1928", "1929"]
            elif "(1920 - 1924)" in description or "(1920-1924)" in description:
                expected_years = ["1920", "1921", "1922", "1923", "1924"]
            elif "(1912 - 1915)" in description or "(1912-1915)" in description:
                expected_years = ["1912", "1913", "1914", "1915"]
            elif "(1928 - 1929)" in description or "(1928-1929)" in description:
                expected_years = ["1928", "1929"]
            elif "(1920 - 1932)" in description or "(1920-1932)" in description:
                expected_years = ["1920", "1921", "1922", "1923", "1924", "1925", "1926", "1927", "1928", "1929", "1930", "1931", "1932"]
            elif "(1928 - 1931)" in description or "(1928-1931)" in description:
                expected_years = ["1928", "1929", "1930", "1931"]
            elif "(1913-1925)" in description:
                expected_years = ["1913", "1914", "1915", "1916", "1917", "1918", "1919", "1920", "1921", "1922", "1923", "1924", "1925"]
            elif "Street Rod or Custom Build" in description:
                expected_years = []  # No specific years expected
            
            # Get AI extracted years
            ai_years_str = row.get('ai_extracted_years', 'N/A')
            ai_years = [year.strip() for year in ai_years_str.split(', ')] if ai_years_str != 'N/A' else []
            
            print(f"    📅 YEAR EXTRACTION CHECK:")
            print(f"       Expected: {expected_years}")
            print(f"       AI Got:   {ai_years}")
            
            # Check if years match
            if expected_years and ai_years:
                if set(expected_years) == set(ai_years):
                    print(f"       ✅ CORRECT - Years match perfectly!")
                else:
                    print(f"       ❌ WRONG - Years don't match!")
                    print(f"       Missing: {set(expected_years) - set(ai_years)}")
                    print(f"       Extra: {set(ai_years) - set(expected_years)}")
            elif not expected_years and not ai_years:
                print(f"       ✅ CORRECT - No years expected or extracted")
            elif not expected_years:
                print(f"       ⚠️  AI extracted years when none expected")
            else:
                print(f"       ❌ WRONG - Expected years but AI didn't extract any")
            
            print(f"    🤖 AI Extracted:")
            print(f"       Make: {row.get('ai_extracted_make', 'N/A')}")
            print(f"       Model: {row.get('ai_extracted_model', 'N/A')}")
            print(f"       Confidence: {row.get('ai_confidence', 'N/A')}")
            
            # Count and show vehicle tags
            tags = row.get('generated_tags', '0_Unknown_UNKNOWN')
            if tags and tags != '0_Unknown_UNKNOWN' and 'Minerva' not in tags:
                tag_list = [tag.strip() for tag in tags.split(',')]
                print(f"    🏷️  Generated Tags: {len(tag_list)} vehicle tags")
                if len(tag_list) <= 3:
                    print(f"       Tags: {', '.join(tag_list)}")
                else:
                    print(f"       First 3: {', '.join(tag_list[:3])}...")
            else:
                print(f"    🏷️  Generated Tags: {tags}")
        
        print("\n" + "=" * 70)
        print(f"📁 OUTPUT SAVED TO: {output_file}")
        print(f"📍 Full path: {os.path.abspath(output_file)}")
        print("=" * 70)
        
        # Verify the file exists
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"✅ File confirmed: {file_size} bytes")
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
        print(f"\n🎯 SUCCESS! Check your YEAR-CORRECTED results in: {output_file}")
    else:
        print("\n❌ Test failed. Check the error messages above.") 