#!/usr/bin/env python3
"""
Test multi-vehicle extraction and tag generation
"""

import os
import pandas as pd
from datetime import datetime
from utils.ai_extraction import TwoPassAIEngine
from utils.golden_master_tag_generator import GoldenMasterTagGenerator
from utils.multi_vehicle_tags import MultiVehicleTagGenerator
from openai import OpenAI

def test_multi_vehicle_extraction():
    """Test the enhanced multi-vehicle extraction"""
    
    print("🔍 TESTING MULTI-VEHICLE EXTRACTION")
    print("=" * 50)
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
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
        
        print(f"✅ System initialized with {len(golden_df)} vehicle records")
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return
    
    # Test products with multiple vehicle compatibility
    test_products = [
        {
            'name': 'Multi-Vehicle Rear Window Gasket Set',
            'description': 'This Rear Window Gasket Set fits 1936-1937 Cord: 810, 812; 1940-1941 Graham Hollywood and Hupmobile Skylark 4 door Sedan models. Molded as left and right with properly shaped corners for a precise, factory fit.',
            'expected_vehicles': ['Cord 810/812', 'Graham Hollywood', 'Hupmobile Skylark']
        },
        {
            'name': 'Universal Weatherstrip for Lincoln Models',
            'description': 'This Side Roof Rail Weatherstrip fits 1931-1939 Lincoln Model K, KA, KB and Series K models. Fifteen (15) foot strip for convertible side window fits aluminum retainer channel.',
            'expected_vehicles': ['Lincoln Model K/KA/KB/Series K']
        },
        {
            'name': 'Ford Multi-Year Door Handle',
            'description': 'Door handle assembly for 1965-1970 Ford Mustang and 1968-1972 Ford Torino models. Chrome finish with black button.',
            'expected_vehicles': ['Ford Mustang', 'Ford Torino']
        }
    ]
    
    print(f"\n🚗 TESTING MULTI-VEHICLE EXTRACTION:")
    results = []
    
    for i, product in enumerate(test_products, 1):
        print(f"\n   {i}. {product['name']}")
        print(f"      📝 Description: {product['description'][:100]}...")
        print(f"      🎯 Expected: {', '.join(product['expected_vehicles'])}")
        
        # Create product info
        product_info = f"""
Product Name: {product['name']}
Description: {product['description']}
"""
        
        try:
            # Extract all vehicle information
            extraction_result = two_pass_engine.extract_initial_vehicle_info(product_info)
            
            print(f"      🔍 AI Extracted Vehicles:")
            all_vehicles = extraction_result.get('all_vehicles', [])
            for idx, vehicle in enumerate(all_vehicles):
                print(f"         {idx+1}. {vehicle['year_min']}-{vehicle['year_max']} {vehicle['make']} {vehicle['model']}")
            
            # Generate tags for all vehicles
            multi_result = multi_tag_generator.generate_all_vehicle_tags(extraction_result)
            
            print(f"      ✅ Generated Tags:")
            print(f"         Total Tags: {multi_result['total_tag_count']}")
            print(f"         Vehicle Count: {multi_result['vehicle_count']}")
            
            # Show breakdown by vehicle
            for vehicle_info in multi_result['vehicle_breakdown']:
                if vehicle_info['tag_count'] > 0:
                    print(f"         • {vehicle_info['make']} {vehicle_info['model']}: {vehicle_info['tag_count']} tags")
                    print(f"           {', '.join(vehicle_info['tags'][:3])}{'...' if len(vehicle_info['tags']) > 3 else ''}")
                else:
                    print(f"         • {vehicle_info['make']} {vehicle_info['model']}: no tags")
            
            # Store result
            results.append({
                'Product_Name': product['name'],
                'Description': product['description'],
                'Vehicles_Found': len(all_vehicles),
                'Total_Tags': multi_result['total_tag_count'],
                'Vehicle_Breakdown': multi_result['vehicle_breakdown'],
                'All_Tags': multi_result['combined_tags_string'],
                'Vehicle_Summary': multi_tag_generator.format_vehicle_summary(multi_result)
            })
            
        except Exception as e:
            print(f"      ❌ Error: {str(e)}")
            results.append({
                'Product_Name': product['name'],
                'Description': product['description'],
                'Vehicles_Found': 0,
                'Total_Tags': 0,
                'Vehicle_Breakdown': [],
                'All_Tags': '',
                'Vehicle_Summary': f'Error: {str(e)}'
            })
    
    # Save results
    results_df = pd.DataFrame(results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f'data/results/multi_vehicle_test_{timestamp}.csv'
    
    os.makedirs('data/results', exist_ok=True)
    results_df.to_csv(output_path, index=False)
    
    print(f"\n✅ RESULTS SAVED: {output_path}")
    
    # Summary
    print(f"\n📊 MULTI-VEHICLE TEST SUMMARY:")
    total_vehicles = sum(r['Vehicles_Found'] for r in results if isinstance(r['Vehicles_Found'], int))
    total_tags = sum(r['Total_Tags'] for r in results if isinstance(r['Total_Tags'], int))
    
    print(f"   🚗 Total vehicles extracted: {total_vehicles}")
    print(f"   🏷️  Total tags generated: {total_tags}")
    print(f"   📊 Average tags per product: {total_tags/len(results):.1f}")
    
    # Show successful examples
    print(f"\n✅ SUCCESSFUL MULTI-VEHICLE EXAMPLES:")
    for result in results:
        if result['Total_Tags'] > 0:
            print(f"   • {result['Product_Name'][:40]:40} → {result['Total_Tags']} tags")
            print(f"     {result['Vehicle_Summary']}")
            if result['All_Tags']:
                tags_preview = result['All_Tags'][:80] + "..." if len(result['All_Tags']) > 80 else result['All_Tags']
                print(f"     Tags: {tags_preview}")
    
    print(f"\n🎉 MULTI-VEHICLE TEST COMPLETED!")
    return output_path


if __name__ == "__main__":
    test_multi_vehicle_extraction()