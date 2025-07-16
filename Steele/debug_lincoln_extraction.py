#!/usr/bin/env python3
"""
Debug the Lincoln Model K, KA, KB, Series K extraction specifically
"""

import os
import pandas as pd
from utils.ai_extraction import TwoPassAIEngine
from openai import OpenAI

def debug_lincoln():
    print("🔍 DEBUGGING LINCOLN MODEL EXTRACTION")
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
    
    # Test the exact Lincoln description
    product_info = """
Product Name: Side Roof Rail Weatherstrip
Description: This Side Roof Rail Weatherstrip fits 1931-1939 Lincoln Model K, KA, KB and Series K models. Fifteen (15) foot strip for convertible side window fits aluminum retainer channel.
Stock Code: 14-0001-80
Price: $45.00
"""
    
    print(f"\n📝 TESTING PRODUCT:")
    print(f"Description: 1931-1939 Lincoln Model K, KA, KB and Series K models")
    print(f"\n🚀 RUNNING PASS 1 EXTRACTION...")
    
    try:
        # Pass 1: Extract initial vehicle info
        initial_data = two_pass_engine.extract_initial_vehicle_info(product_info)
        
        print(f"\n✅ PASS 1 RESULTS:")
        print(f"Primary: {initial_data['year_min']}-{initial_data['year_max']} {initial_data['make']} {initial_data['model']}")
        
        all_vehicles = initial_data.get('all_vehicles', [])
        print(f"\nALL VEHICLES EXTRACTED ({len(all_vehicles)}):")
        for i, vehicle in enumerate(all_vehicles):
            print(f"  {i+1}. {vehicle['year_min']}-{vehicle['year_max']} {vehicle['make']} {vehicle['model']}")
        
        # Expected: 4 vehicles (Model K, Model KA, Model KB, Series K)
        expected_models = ['Model K', 'Model KA', 'Model KB', 'Series K']
        found_models = [v['model'] for v in all_vehicles]
        
        print(f"\n📊 ANALYSIS:")
        print(f"Expected models: {expected_models}")
        print(f"Found models: {found_models}")
        
        missing = set(expected_models) - set(found_models)
        if missing:
            print(f"❌ MISSING MODELS: {missing}")
        else:
            print(f"✅ ALL MODELS FOUND!")
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")

if __name__ == "__main__":
    debug_lincoln()