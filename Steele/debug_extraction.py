#!/usr/bin/env python3
"""
Debug script to trace the AI extraction and expansion process step by step
"""

import pandas as pd
from utils.ai_fitment_extractor import SteeleAIFitmentExtractor

def main():
    print("🔍 DEBUG: AI Extraction Process")
    print("=" * 50)
    
    # Initialize extractor
    extractor = SteeleAIFitmentExtractor()
    
    # Test the problematic case
    product_name = "Glass weatherstrip kit"
    description = "Running Board Step Pad. Made from top quality rubber to ensure durability. This part is compatible with models built by Independent (1920 - 1929) automobile manufacturers."
    
    print(f"Product: {product_name}")
    print(f"Description: {description}")
    print()
    
    # STEP 1: First pass AI extraction
    print("STEP 1: First Pass AI Extraction")
    print("-" * 30)
    extraction = extractor.extract_fitment_from_description(
        product_name=product_name,
        description=description,
        stock_code="10-0108-45"
    )
    
    print(f"✅ AI Extracted:")
    print(f"  Years: {extraction.years}")
    print(f"  Make: {extraction.make}")
    print(f"  Model: {extraction.model}")
    print(f"  Confidence: {extraction.confidence}")
    print(f"  Reasoning: {extraction.reasoning}")
    print(f"  Error: {extraction.error}")
    print()
    
    # STEP 2: Second pass expansion
    print("STEP 2: Second Pass Expansion")
    print("-" * 30)
    expanded_extraction = extractor.expand_fitment_extraction(extraction)
    
    print(f"🔄 After Expansion:")
    print(f"  Years: {expanded_extraction.years}")
    print(f"  Make: {expanded_extraction.make}")
    print(f"  Model: {expanded_extraction.model}")
    print(f"  Vehicle Tags Count: {len(expanded_extraction.vehicle_tags) if expanded_extraction.vehicle_tags else 0}")
    print(f"  Error: {expanded_extraction.error}")
    print()
    
    # STEP 3: Show some sample tags
    if expanded_extraction.vehicle_tags:
        print("STEP 3: Sample Generated Tags")
        print("-" * 30)
        sample_tags = expanded_extraction.vehicle_tags[:10]  # Show first 10
        for i, tag in enumerate(sample_tags, 1):
            print(f"  {i}. {tag}")
        
        if len(expanded_extraction.vehicle_tags) > 10:
            print(f"  ... and {len(expanded_extraction.vehicle_tags) - 10} more tags")
    else:
        print("❌ No vehicle tags generated!")
    
    print()
    print("=" * 50)
    print("Debug complete!")

if __name__ == "__main__":
    main() 