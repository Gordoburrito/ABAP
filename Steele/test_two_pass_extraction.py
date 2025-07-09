#!/usr/bin/env python3
"""
Test script for Steele Two-Pass AI Fitment Extraction

This script demonstrates how the enhanced AI extractor handles vague fitment descriptions
like "models built by Independent (1920-1929) automobile manufacturers" using a two-pass approach:

1. First Pass: Extract basic fitment information, identifying "ALL" values
2. Second Pass: Expand "ALL" values using golden master data to get specific makes/models

This approach is similar to what was implemented in the REM processor.
"""

import sys
import os
import pandas as pd
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.ai_fitment_extractor import SteeleAIFitmentExtractor

def create_test_cases():
    """Create test cases that demonstrate the problem and solution"""
    test_cases = [
        {
            'StockCode': '10-0108-45',
            'Product Name': 'Glass weatherstrip kit',
            'Description': 'Running Board Step Pad. Made from top quality rubber to ensure durability. This part is compatible with models built by Independent (1920 - 1929) automobile manufacturers.',
            'Make': 'Independent',
            'Model': 'UNKNOWN'
        },
        {
            'StockCode': '10-0127-52',
            'Product Name': 'Windshield to Cowl Weatherstrip',
            'Description': 'This Windshield to Cowl Weatherstrip comes in a thirty-six inch (36") strip and seals the lower windshield frame to the cowl of the car. This generic part is known to fit Independent vehicle models produced from 1920 - 1924.',
            'Make': 'Independent',
            'Model': 'UNKNOWN'
        },
        {
            'StockCode': '10-0128-35',
            'Product Name': 'Pad, rear axle rebound',
            'Description': 'Axle Rebound Pad - Rear. Made from top quality rubber to ensure durability. This part is compatible with (1912 - 1915) Sterns-Knight models.',
            'Make': 'Sterns-Knight',
            'Model': 'UNKNOWN'
        },
        {
            'StockCode': '10-0130-108',
            'Product Name': 'Accelerator Pedal Pad',
            'Description': 'Steele Rubber Products has the best quality Accelerator Pedal Pads for your Street Rod or Custom Build project. These Accelerator Pads are made of top-quality EPDM rubber with a high level of fit and finish for the Builder\'s creative application and use.',
            'Make': 'Street Rod',
            'Model': 'UNKNOWN'
        },
        {
            'StockCode': '10-0180-22',
            'Product Name': 'Pedal pads, Minerva',
            'Description': 'Pads, brake and clutch pedal. Good copy of original with bonded and riveted steel back, to be retained by tabs. For (1928 - 1931) Minerva Motors vehicle models.',
            'Make': 'Minerva',
            'Model': 'UNKNOWN'
        }
    ]
    
    return pd.DataFrame(test_cases)

def run_two_pass_extraction_test():
    """Run the two-pass extraction test"""
    print("🚀 STEELE TWO-PASS AI FITMENT EXTRACTION TEST")
    print("=" * 70)
    print("Testing enhanced AI extraction for vague fitment descriptions")
    print("Comparing traditional approach vs. two-pass approach")
    print("=" * 70)
    
    # Create test cases
    test_df = create_test_cases()
    
    print(f"\n📋 TEST CASES ({len(test_df)} items):")
    print("-" * 50)
    for idx, row in test_df.iterrows():
        print(f"{idx+1}. {row['StockCode']}: {row['Product Name']}")
        print(f"   Description: {row['Description'][:80]}...")
        print(f"   Current: {row['Make']} {row['Model']}")
        print()
    
    # Initialize the enhanced extractor
    try:
        extractor = SteeleAIFitmentExtractor()
        print("✅ AI Fitment Extractor initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize extractor: {e}")
        return
    
    # Process test cases with two-pass approach
    print("\n🔄 PROCESSING WITH TWO-PASS APPROACH...")
    print("=" * 70)
    
    try:
        results_df = extractor.process_unknown_skus_batch_with_expansion(test_df)
        
        # Display results
        print("\n📊 RESULTS COMPARISON:")
        print("=" * 70)
        
        for idx, row in results_df.iterrows():
            print(f"\n{idx+1}. {row['StockCode']}: {row['Product Name']}")
            print(f"   Original: {row['Make']} {row['Model']}")
            print(f"   AI First Pass: {row['ai_extracted_make']} {row['ai_extracted_model']}")
            print(f"   Generated Tags: {row['generated_tags']}")
            print(f"   Confidence: {row['ai_confidence']:.2f}")
            
            if row['extraction_error']:
                print(f"   ❌ Error: {row['extraction_error']}")
            else:
                # Show improvement
                original_tag = f"0_{row['Make']}_UNKNOWN"
                new_tag = row['generated_tags']
                if original_tag != new_tag and '0_Unknown_UNKNOWN' not in new_tag:
                    print(f"   ✅ IMPROVED: {original_tag} → {new_tag}")
                else:
                    print(f"   ⚠️  No improvement over: {original_tag}")
        
        # Summary
        print("\n" + "=" * 70)
        print("📈 SUMMARY:")
        
        # Count improvements
        improved_count = 0
        total_processed = len(results_df)
        
        for _, row in results_df.iterrows():
            if (not row['extraction_error'] and 
                '0_Unknown_UNKNOWN' not in row['generated_tags'] and
                row['ai_extracted_make'] != "UNKNOWN"):
                improved_count += 1
        
        print(f"   • Total test cases: {total_processed}")
        print(f"   • Successfully improved: {improved_count}")
        print(f"   • Improvement rate: {improved_count/total_processed*100:.1f}%")
        
        # Save results
        output_file = Path(__file__).parent / "data" / "samples" / "two_pass_test_results.csv"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_file, index=False)
        print(f"   • Results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()

def demonstrate_problem():
    """Demonstrate the current problem with 0_Unknown_UNKNOWN tags"""
    print("\n🔍 CURRENT PROBLEM DEMONSTRATION:")
    print("-" * 50)
    
    current_examples = [
        {
            'description': 'models built by Independent (1920-1929) automobile manufacturers',
            'current_output': '0_Unknown_UNKNOWN',
            'should_be': '1920_Independent_Model_A, 1921_Independent_Model_B, ...'
        },
        {
            'description': 'Street Rod or Custom Build project',
            'current_output': '0_Unknown_UNKNOWN', 
            'should_be': 'ALL_Makes_ALL_Models (universal fitment)'
        },
        {
            'description': '(1912-1915) Sterns-Knight models',
            'current_output': '0_Unknown_UNKNOWN',
            'should_be': '1912_Sterns-Knight_Model_X, 1913_Sterns-Knight_Model_Y, ...'
        }
    ]
    
    for i, example in enumerate(current_examples, 1):
        print(f"{i}. Description: '{example['description']}'")
        print(f"   Current Output: {example['current_output']}")
        print(f"   Should Generate: {example['should_be']}")
        print()

if __name__ == "__main__":
    print("🧪 STEELE TWO-PASS AI FITMENT EXTRACTION")
    print("Testing solution for vague fitment descriptions")
    print()
    
    # First show the problem
    demonstrate_problem()
    
    # Then test the solution
    run_two_pass_extraction_test()
    
    print("\n" + "=" * 70)
    print("✅ Test complete! Check the generated tags above.")
    print("The two-pass approach should handle vague descriptions much better")
    print("than the current 0_Unknown_UNKNOWN fallback.")
    print("=" * 70) 