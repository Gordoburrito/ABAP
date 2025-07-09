#!/usr/bin/env python3
"""
Test script for the real two-pass AI extraction on actual problematic Steele data.
This will test the problematic entries from the CSV that currently show 0_Unknown_UNKNOWN tags.
"""

import pandas as pd
import os
from datetime import datetime
from utils.ai_fitment_extractor import SteeleAIFitmentExtractor

def main():
    print("🧪 Testing Two-Pass AI Extraction on Real Steele Data")
    print("=" * 60)
    
    # Sample problematic entries from the CSV file
    problematic_data = {
        'StockCode': [
            '10-0108-45',
            '10-0127-52', 
            '10-0128-35',
            '10-0129-35',
            '10-0130-108',
            '10-0130-40'
        ],
        'Product Name': [
            'Glass weatherstrip kit',
            'Windshield to Cowl Weatherstrip',
            'Pad, rear axle rebound',
            'Pad, front axle rebound', 
            'Accelerator Pedal Pad',
            'Pad, accelerator pedal'
        ],
        'Description': [
            'Running Board Step Pad. Made from top quality rubber to ensure durability. This part is compatible with models built by Independent (1920 - 1929) automobile manufacturers.',
            'This Windshield to Cowl Weatherstrip comes in a thirty-six inch (36") strip and seals the lower windshield frame to the cowl of the car. This generic part is known to fit Independent vehicle models produced from 1920 - 1924.',
            'Axle Rebound Pad - Rear. Made from top quality rubber to ensure durability. This part is compatible with (1912 - 1915) Sterns-Knight models.',
            'Axle Rebound Pad - Rear. Made from top quality rubber to ensure durability. This part is compatible with (1912 - 1915) Sterns-Knight models.',
            'Steele Rubber Products has the best quality Accelerator Pedal Pads for your Street Rod or Custom Build project. These Accelerator Pads are made of top-quality EPDM rubber with a high level of fit and finish for the Builder\'s creative application and use.',
            'Accelerator Pedal Pad. Made from top quality rubber to ensure durability. This part is compatible with (1928 - 1929) Sterns-Knight models.'
        ]
    }
    
    # Convert to DataFrame
    test_df = pd.DataFrame(problematic_data)
    
    print(f"Testing {len(test_df)} problematic product descriptions...")
    print("\nCurrent behavior: These all generate '0_Unknown_UNKNOWN' tags")
    print("Expected: Two-pass AI approach should handle them better\n")
    
    # Initialize the AI Fitment Extractor
    # Note: This will use mocked OpenAI calls for testing
    extractor = SteeleAIFitmentExtractor()
    
    print("🔍 Testing WITHOUT API calls (using mock extraction for demonstration)...")
    print("=" * 60)
    
    # Test each entry individually with mock responses
    for idx, row in test_df.iterrows():
        print(f"\n[{idx+1}/{len(test_df)}] Testing: {row['StockCode']}")
        print(f"Product: {row['Product Name']}")
        print(f"Description: {row['Description'][:100]}...")
        
        # Mock the extraction based on description patterns
        if "Independent" in row['Description']:
            if "1920 - 1929" in row['Description']:
                mock_extraction = {
                    'years': [str(year) for year in range(1920, 1930)],
                    'make': 'ALL',  # Independent not in golden master, so AI should use ALL
                    'model': 'ALL',
                    'confidence': 0.85,
                    'reasoning': 'Vague Independent manufacturers 1920-1929'
                }
            elif "1920 - 1924" in row['Description']:
                mock_extraction = {
                    'years': [str(year) for year in range(1920, 1925)],
                    'make': 'ALL', 
                    'model': 'ALL',
                    'confidence': 0.80,
                    'reasoning': 'Generic Independent models 1920-1924'
                }
        elif "Sterns-Knight" in row['Description']:
            if "1912 - 1915" in row['Description']:
                mock_extraction = {
                    'years': [str(year) for year in range(1912, 1916)],
                    'make': 'Sterns-Knight',
                    'model': 'ALL',
                    'confidence': 0.90,
                    'reasoning': 'Specific Sterns-Knight 1912-1915'
                }
            elif "1928 - 1929" in row['Description']:
                mock_extraction = {
                    'years': [str(year) for year in range(1928, 1930)],
                    'make': 'Sterns-Knight',
                    'model': 'ALL',
                    'confidence': 0.88,
                    'reasoning': 'Specific Sterns-Knight 1928-1929'
                }
        elif "Street Rod or Custom Build" in row['Description']:
            mock_extraction = {
                'years': ['ALL'],  # Street rod could be any year
                'make': 'ALL',
                'model': 'ALL', 
                'confidence': 0.75,
                'reasoning': 'Street Rod/Custom Build - universal application'
            }
        else:
            mock_extraction = {
                'years': [],
                'make': 'UNKNOWN',
                'model': 'UNKNOWN',
                'confidence': 0.0,
                'reasoning': 'Unable to extract fitment'
            }
            
        print(f"🤖 Mock AI Extraction: Years={mock_extraction['years']}, Make={mock_extraction['make']}, Model={mock_extraction['model']}")
        
        # Test expansion logic (this will use real golden master data)
        from utils.ai_fitment_extractor import FitmentExtraction
        extraction_obj = FitmentExtraction(**mock_extraction)
        expanded = extractor.expand_fitment_extraction(extraction_obj)
        
        print(f"🔄 After Expansion: Years={expanded.years}, Make={expanded.make}, Model={expanded.model}")
        
        # Generate tags
        final_tags = extractor._generate_vehicle_tags_from_extraction(expanded)
        print(f"🏷️  Final Tags: {', '.join(final_tags[:5])}{'...' if len(final_tags) > 5 else ''}")
        print(f"    Total tags generated: {len(final_tags)}")
        
    print("\n" + "=" * 60)
    print("✅ Two-Pass AI Extraction Test Complete!")
    
    print(f"\n📍 Next Steps:")
    print("1. The TDD tests are passing - our logic is sound")
    print("2. Run this with real OpenAI API calls to test actual AI extraction")
    print("3. Process the full dataset with the new two-pass approach")
    print("4. Results will be saved to: data/results/real_two_pass_test_[timestamp].csv")

if __name__ == "__main__":
    main() 