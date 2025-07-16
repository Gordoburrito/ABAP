#!/usr/bin/env python3
"""
Debug the golden master model matching enhancement
"""

import os
import pandas as pd
from utils.ai_extraction import TwoPassAIEngine
from openai import OpenAI

def test_model_matching():
    print("🔍 TESTING GOLDEN MASTER MODEL MATCHING")
    print("=" * 50)
    
    # Check API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not found")
        return
    
    # Load golden master
    try:
        golden_df = pd.read_csv('../shared/data/master_ultimate_golden.csv', low_memory=False)
        golden_df.columns = [col.lower().replace(' ', '_') for col in golden_df.columns]
        
        # Initialize AI engine
        client = OpenAI(api_key=api_key)
        two_pass_engine = TwoPassAIEngine(client, golden_df)
        
        print(f"✅ Golden master loaded: {len(golden_df)} records")
        
    except Exception as e:
        print(f"❌ Could not load golden master: {e}")
        return
    
    # Test cases with short model names that should be matched to golden master
    test_cases = [
        {
            "name": "Lincoln K models (short name)",
            "description": "This part fits 1935-1938 Lincoln K models",
            "expected_refinement": "K → Model K"
        },
        {
            "name": "Studebaker Commander/President (should find specific models)", 
            "description": "This part fits 1938-1939 Studebaker Commander and President models",
            "expected_refinement": "Commander → 6-7A Commander, 6-8A Commander; President → State President"
        },
        {
            "name": "Complex Lincoln description",
            "description": "Fits 1931-1939 Lincoln K, KA, KB models",
            "expected_refinement": "K → Model K, KA → Model KA, KB → Model KB"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}: {test_case['name']}")
        print(f"Description: {test_case['description']}")
        print(f"Expected: {test_case['expected_refinement']}")
        print(f"{'='*60}")
        
        product_info = f"""
Product Name: Test Part
Description: {test_case['description']}
Stock Code: TEST-{i}
Price: $25.00
"""
        
        try:
            # Pass 1: Extract with golden master matching
            initial_data = two_pass_engine.extract_initial_vehicle_info(product_info)
            
            all_vehicles = initial_data.get('all_vehicles', [])
            print(f"\n✅ PASS 1 RESULTS ({len(all_vehicles)} vehicles):")
            for j, vehicle in enumerate(all_vehicles):
                print(f"  {j+1}. {vehicle['year_min']}-{vehicle['year_max']} {vehicle['make']} {vehicle['model']}")
            
            # Check if golden master matching worked
            models_found = [v['model'] for v in all_vehicles]
            print(f"\n📊 ANALYSIS:")
            print(f"Models extracted: {models_found}")
            
            # Check against what we know is in golden master
            if any('Model K' in model for model in models_found):
                print(f"✅ SUCCESS: Found full 'Model K' format")
            elif any('K' == model for model in models_found):
                print(f"⚠️  PARTIAL: Still using short 'K' format")
            
            if any('State President' in model for model in models_found):
                print(f"✅ SUCCESS: Found 'State President' from golden master")
            elif any('President' == model for model in models_found):
                print(f"⚠️  PARTIAL: Still using generic 'President'")
                
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")

if __name__ == "__main__":
    test_model_matching()