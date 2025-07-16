#!/usr/bin/env python3
"""
Quick test with 5 products to verify the two-pass AI fix
"""

import os
import pandas as pd
import time
from datetime import datetime
from utils.ai_extraction import TwoPassAIEngine
from utils.golden_master_tag_generator import GoldenMasterTagGenerator
from utils.multi_vehicle_tags import MultiVehicleTagGenerator
from openai import OpenAI

def test_5_products():
    """Test the two-pass AI approach with 5 products quickly"""
    
    print("🔍 TESTING 5 PRODUCTS WITH TWO-PASS AI FIX")
    print("=" * 50)
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not found")
        return
    
    # Load test data - take first 5 from the 100-product dataset
    try:
        df = pd.read_csv('data/results/test_100_products.csv')
        df = df.head(5)  # Take first 5
        print(f"✅ Loaded {len(df)} test products")
    except Exception as e:
        print(f"❌ Could not load test data: {e}")
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
    
    # Process each product
    results = []
    start_time = time.time()
    
    print(f"\n🚀 PROCESSING 5 PRODUCTS:")
    
    for idx, row in df.iterrows():
        print(f"\n   {idx+1}/5: {row['Product Name'][:50]}...")
        
        # Create product info
        product_info = f"""
Product Name: {row['Product Name']}
Description: {row['Description']}
Stock Code: {row['StockCode']}
Price: ${row.get('MAP', 0)}
"""
        
        try:
            # Pass 1: Extract initial vehicle info
            initial_data = two_pass_engine.extract_initial_vehicle_info(product_info)
            all_vehicles = initial_data.get('all_vehicles', [])
            print(f"       Pass 1: {len(all_vehicles)} vehicles extracted")
            
            # Pass 2: Refine with golden master
            refined_data = two_pass_engine.refine_with_golden_master(initial_data, product_info)
            
            # Get refined vehicles
            refined_vehicles = refined_data.get('all_vehicles_refined', initial_data.get('all_vehicles', []))
            print(f"       Pass 2: {len(refined_vehicles)} vehicles refined")
            
            # Generate tags for ALL REFINED vehicles
            refined_vehicle_data = {
                'all_vehicles': refined_vehicles,
                'year_min': refined_data['year_min'],
                'year_max': refined_data['year_max'],
                'make': refined_data['make'],
                'model': refined_data['model']
            }
            
            multi_result = multi_tag_generator.generate_all_vehicle_tags(refined_vehicle_data)
            tags = multi_result['combined_tags_string']
            tag_count = multi_result['total_tag_count']
            
            if tag_count > 0:
                print(f"       ✅ Tags: {tag_count} ({tags[:60]}{'...' if len(tags) > 60 else ''})")
            else:
                print(f"       ❌ No tags generated")
            
            # Store result
            results.append({
                'ID': idx + 1,
                'Product_Name': row['Product Name'],
                'Pass1_Vehicles': len(all_vehicles),
                'Pass2_Vehicles': len(refined_vehicles),
                'Final_Tags_Count': tag_count,
                'Status': 'SUCCESS' if tag_count > 0 else 'NO_TAGS'
            })
            
        except Exception as e:
            print(f"       ❌ ERROR: {str(e)}")
            results.append({
                'ID': idx + 1,
                'Product_Name': row['Product Name'],
                'Pass1_Vehicles': 0,
                'Pass2_Vehicles': 0,
                'Final_Tags_Count': 0,
                'Status': 'ERROR'
            })
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    # Create DataFrame and save
    results_df = pd.DataFrame(results)
    
    print(f"\n✅ RESULTS SUMMARY:")
    print(f"   🕐 Total time: {total_duration:.1f} seconds")
    print(f"   ⚡ Average time per product: {total_duration/5:.1f} seconds")
    
    # Success rates
    successful = results_df[results_df['Status'] == 'SUCCESS']
    print(f"   ✅ Products with tags: {len(successful)}/5 ({len(successful)/5*100:.1f}%)")
    
    if len(successful) > 0:
        total_tags = successful['Final_Tags_Count'].sum()
        print(f"   🏷️  Total tags generated: {total_tags}")
        print(f"   📊 Average tags per successful product: {total_tags/len(successful):.1f}")
    
    print(f"\n📋 DETAILED RESULTS:")
    for _, row in results_df.iterrows():
        status_icon = "✅" if row['Status'] == 'SUCCESS' else "❌" if row['Status'] == 'ERROR' else "🔄"
        print(f"   {row['ID']}. {status_icon} {row['Product_Name'][:40]:40} → P1:{row['Pass1_Vehicles']} P2:{row['Pass2_Vehicles']} Tags:{row['Final_Tags_Count']}")
    
    print(f"\n🎉 5-PRODUCT TEST COMPLETED!")

if __name__ == "__main__":
    test_5_products()