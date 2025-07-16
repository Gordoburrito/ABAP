#!/usr/bin/env python3
"""
Debug the full Lincoln tag generation end-to-end
"""

import os
import pandas as pd
from utils.ai_extraction import TwoPassAIEngine
from utils.golden_master_tag_generator import GoldenMasterTagGenerator
from utils.multi_vehicle_tags import MultiVehicleTagGenerator
from openai import OpenAI

def debug_lincoln_tags():
    print("🔍 DEBUGGING LINCOLN TAG GENERATION END-TO-END")
    print("=" * 55)
    
    # Check API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not found")
        return
    
    # Load golden master
    try:
        golden_df = pd.read_csv('../shared/data/master_ultimate_golden.csv', low_memory=False)
        golden_df.columns = [col.lower().replace(' ', '_') for col in golden_df.columns]
        tag_generator = GoldenMasterTagGenerator(golden_df)
        multi_tag_generator = MultiVehicleTagGenerator(tag_generator)
        
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
    
    try:
        # Pass 1: Extract initial vehicle info
        print(f"\n🚀 RUNNING PASS 1 EXTRACTION...")
        initial_data = two_pass_engine.extract_initial_vehicle_info(product_info)
        
        all_vehicles = initial_data.get('all_vehicles', [])
        print(f"✅ Pass 1 - Found {len(all_vehicles)} vehicles:")
        for i, vehicle in enumerate(all_vehicles):
            print(f"  {i+1}. {vehicle['year_min']}-{vehicle['year_max']} {vehicle['make']} {vehicle['model']}")
        
        # Pass 2: Refine with golden master
        print(f"\n🚀 RUNNING PASS 2 REFINEMENT...")
        refined_data = two_pass_engine.refine_with_golden_master(initial_data, product_info)
        print(f"✅ Pass 2 - Primary: {refined_data['year_min']}-{refined_data['year_max']} {refined_data['make']} {refined_data['model']}")
        
        # Generate tags for ALL vehicles
        print(f"\n🚀 RUNNING MULTI-VEHICLE TAG GENERATION...")
        multi_result = multi_tag_generator.generate_all_vehicle_tags(initial_data)
        
        print(f"\n✅ TAG GENERATION RESULTS:")
        print(f"Total tags: {multi_result['total_tag_count']}")
        print(f"Tags: {multi_result['combined_tags_string']}")
        
        print(f"\n📊 VEHICLE BREAKDOWN:")
        for vehicle_info in multi_result['vehicle_breakdown']:
            print(f"  • {vehicle_info['make']} {vehicle_info['model']} ({vehicle_info['year_range']}): {vehicle_info['tag_count']} tags")
            if vehicle_info['tags']:
                print(f"    Tags: {', '.join(vehicle_info['tags'])}")
        
        # Check if we got tags for all models
        models_with_tags = [v for v in multi_result['vehicle_breakdown'] if v['tag_count'] > 0]
        print(f"\n🎯 SUMMARY:")
        print(f"Models with tags: {len(models_with_tags)}/4")
        
        if len(models_with_tags) == 4:
            print(f"✅ SUCCESS: All 4 Lincoln models generated tags!")
        else:
            missing = [v['model'] for v in multi_result['vehicle_breakdown'] if v['tag_count'] == 0]
            print(f"❌ MISSING TAGS FOR: {missing}")
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")

if __name__ == "__main__":
    debug_lincoln_tags()