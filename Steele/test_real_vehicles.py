#!/usr/bin/env python3
"""
Test with realistic vehicle data that should generate tags
"""

import os
import pandas as pd
from utils.ai_extraction import TwoPassAIEngine
from utils.golden_master_tag_generator import GoldenMasterTagGenerator
from openai import OpenAI

def test_real_vehicles():
    """Test with products that should generate vehicle tags"""
    
    print("🔍 TESTING WITH REAL VEHICLE DATA")
    print("=" * 50)
    
    # Check API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not found")
        return
    
    # Load test data with real vehicles
    try:
        df = pd.read_csv('test_with_real_vehicles.csv')
        print(f"📋 Loaded {len(df)} test products")
        for _, row in df.iterrows():
            print(f"   - {row['Product Name']}")
    except Exception as e:
        print(f"❌ Could not load test data: {e}")
        return
    
    # Load golden master
    try:
        golden_df = pd.read_csv('../shared/data/master_ultimate_golden.csv', low_memory=False)
        golden_df.columns = [col.lower().replace(' ', '_') for col in golden_df.columns]
        tag_generator = GoldenMasterTagGenerator(golden_df)
        
        # Initialize AI engine
        client = OpenAI(api_key=api_key)
        two_pass_engine = TwoPassAIEngine(client, golden_df)
        
        print(f"✅ Golden master loaded: {len(golden_df)} records")
        
    except Exception as e:
        print(f"❌ Could not load golden master: {e}")
        return
    
    # Process each product
    results = []
    
    for idx, row in df.iterrows():
        print(f"\n🔍 Processing {idx+1}/{len(df)}: {row['Product Name']}")
        
        # Create product info
        product_info = f"""
Product Name: {row['Product Name']}
Description: {row['Description']}
Stock Code: {row['StockCode']}
Price: ${row['MAP']}
"""
        
        try:
            # Pass 1: Extract initial vehicle info
            initial_data = two_pass_engine.extract_initial_vehicle_info(product_info)
            print(f"   Pass 1: {initial_data['year_min']}-{initial_data['year_max']} {initial_data['make']} {initial_data['model']}")
            
            # Pass 2: Refine with golden master
            refined_data = two_pass_engine.refine_with_golden_master(initial_data, product_info)
            print(f"   Pass 2: {refined_data['make']} {refined_data['model']}")
            
            # Generate tags - parse models more intelligently
            models = []
            if isinstance(refined_data['model'], str):
                # Handle "810 and 812", "810/812", etc.
                model_str = refined_data['model'].replace(' and ', ', ').replace('/', ', ').replace('|', ', ')
                models = [m.strip() for m in model_str.split(',') if m.strip()]
                print(f"   Models parsed: {models}")
            
            tags = ""
            if refined_data['make'].lower() not in ['universal', 'unknown'] and models:
                vehicle_tags = tag_generator.generate_vehicle_tags_from_car_ids(
                    refined_data['year_min'], refined_data['year_max'], 
                    refined_data['make'], models
                )
                if vehicle_tags:
                    tags = ', '.join(vehicle_tags)
                    print(f"   ✅ Tags: {tags}")
                else:
                    print(f"   ❌ No tags generated for {refined_data['make']} {models}")
            else:
                print(f"   🔄 Universal product - no tags")
            
            # Store result
            results.append({
                'SKU': row['StockCode'],
                'Product_Name': row['Product Name'],
                'Pass1_Make': initial_data['make'],
                'Pass1_Model': initial_data['model'],
                'Pass2_Make': refined_data['make'],
                'Pass2_Model': refined_data['model'],
                'Final_Tags': tags,
                'Tags_Count': len(vehicle_tags) if 'vehicle_tags' in locals() and vehicle_tags else 0
            })
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            results.append({
                'SKU': row['StockCode'],
                'Product_Name': row['Product Name'],
                'Pass1_Make': 'ERROR',
                'Pass1_Model': 'ERROR',
                'Pass2_Make': 'ERROR',
                'Pass2_Model': 'ERROR',
                'Final_Tags': '',
                'Tags_Count': 0
            })
    
    # Create DataFrame and save
    results_df = pd.DataFrame(results)
    
    # Create output file
    output_path = 'data/results/real_vehicles_test_results.csv'
    os.makedirs('data/results', exist_ok=True)
    results_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Results saved to: {output_path}")
    
    # Show summary
    print(f"\n📋 REAL VEHICLES TEST SUMMARY:")
    print(f"   Products processed: {len(results_df)}")
    
    successful_tags = results_df[results_df['Tags_Count'] > 0]
    print(f"   Products with tags: {len(successful_tags)}/{len(results_df)}")
    
    if len(successful_tags) > 0:
        print(f"   ✅ SUCCESSFUL TAG GENERATIONS:")
        for _, row in successful_tags.iterrows():
            print(f"      {row['SKU']}: {row['Final_Tags']}")
    
    # Show specific vehicle results
    print(f"\n📋 DETAILED RESULTS:")
    for _, row in results_df.iterrows():
        print(f"   {row['SKU']}: {row['Pass1_Make']} → {row['Pass2_Make']}")
        if row['Final_Tags']:
            print(f"      Tags: {row['Final_Tags']}")
        else:
            print(f"      Tags: (none)")


if __name__ == "__main__":
    test_real_vehicles()