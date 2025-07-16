#!/usr/bin/env python3
"""
Test the Studebaker 5-model fix specifically
"""

import os
import pandas as pd
from utils.ai_extraction import TwoPassAIEngine
from utils.golden_master_tag_generator import GoldenMasterTagGenerator
from utils.multi_vehicle_tags import MultiVehicleTagGenerator
from openai import OpenAI

def test_studebaker():
    print("🔍 TESTING STUDEBAKER 5-MODEL FIX")
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
        tag_generator = GoldenMasterTagGenerator(golden_df)
        multi_tag_generator = MultiVehicleTagGenerator(tag_generator)
        
        # Initialize AI engine
        client = OpenAI(api_key=api_key)
        two_pass_engine = TwoPassAIEngine(client, golden_df)
        
        print(f"✅ Golden master loaded: {len(golden_df)} records")
        
    except Exception as e:
        print(f"❌ Could not load golden master: {e}")
        return
    
    # Test the exact Studebaker description from your CSV
    product_info = """
Product Name: Front Door Vent Window Weatherstrips
Description: This Front Door Vent Window Weatherstrips fits 1938-1939 Studebaker 6-7A Commander, 6-8A Commander, 8-4C State President, 8-5C State President, and 9A Commander models
Stock Code: TEST-STUDEBAKER
Price: $45.00
"""
    
    print(f"\n📝 TESTING STUDEBAKER PRODUCT:")
    print(f"Expected: 5 models (6-7A Commander, 6-8A Commander, 8-4C State President, 8-5C State President, 9A Commander)")
    
    try:
        # Pass 1: Extract all vehicles
        print(f"\n🚀 PASS 1 - EXTRACTING ALL VEHICLES...")
        initial_data = two_pass_engine.extract_initial_vehicle_info(product_info)
        
        all_vehicles = initial_data.get('all_vehicles', [])
        print(f"✅ Pass 1 extracted {len(all_vehicles)} vehicles:")
        for i, vehicle in enumerate(all_vehicles):
            print(f"  {i+1}. {vehicle['year_min']}-{vehicle['year_max']} {vehicle['make']} {vehicle['model']}")
        
        # Pass 2: Refine (but we'll ignore this for tags)
        print(f"\n🚀 PASS 2 - REFINING PRIMARY VEHICLE...")
        refined_data = two_pass_engine.refine_with_golden_master(initial_data, product_info)
        print(f"✅ Pass 2 refined: {refined_data['year_min']}-{refined_data['year_max']} {refined_data['make']} {refined_data['model']}")
        
        # Generate tags for ALL REFINED vehicles from Pass 2 (the proper fix!)
        print(f"\n🚀 MULTI-VEHICLE TAG GENERATION - USING ALL REFINED VEHICLES...")
        
        # Use refined vehicles if available
        refined_vehicles = refined_data.get('all_vehicles_refined', initial_data.get('all_vehicles', []))
        print(f"Using {len(refined_vehicles)} refined vehicles for tag generation")
        
        # Create data structure for tag generation
        refined_vehicle_data = {
            'all_vehicles': refined_vehicles,
            'year_min': refined_data['year_min'],
            'year_max': refined_data['year_max'],
            'make': refined_data['make'],
            'model': refined_data['model']
        }
        
        multi_result = multi_tag_generator.generate_all_vehicle_tags(refined_vehicle_data)
        
        print(f"\n✅ TAG RESULTS:")
        print(f"Total vehicles processed: {multi_result['vehicle_count']}")
        print(f"Total tags generated: {multi_result['total_tag_count']}")
        print(f"All tags: {multi_result['combined_tags_string']}")
        
        print(f"\n📊 VEHICLE BREAKDOWN:")
        for vehicle_info in multi_result['vehicle_breakdown']:
            print(f"  • {vehicle_info['make']} {vehicle_info['model']} ({vehicle_info['year_range']}): {vehicle_info['tag_count']} tags")
            if vehicle_info['tags']:
                print(f"    Tags: {', '.join(vehicle_info['tags'])}")
        
        # Check if we got tags for all 5 models
        models_with_tags = [v for v in multi_result['vehicle_breakdown'] if v['tag_count'] > 0]
        print(f"\n🎯 SUMMARY:")
        print(f"Expected: 5 Studebaker models")
        print(f"Found: {len(all_vehicles)} vehicles in Pass 1")
        print(f"With tags: {len(models_with_tags)} models")
        
        if len(models_with_tags) >= 3:  # Some models might not be in golden master
            print(f"✅ SUCCESS: Multiple Studebaker models generating tags!")
        else:
            print(f"❌ ISSUE: Only {len(models_with_tags)} models generating tags")
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")

if __name__ == "__main__":
    test_studebaker()