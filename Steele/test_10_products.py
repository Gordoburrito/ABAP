#!/usr/bin/env python3
"""
Test 10 vehicle-specific products with the two-pass AI approach
"""

import os
import pandas as pd
import time
from datetime import datetime
from utils.ai_extraction import TwoPassAIEngine
from utils.golden_master_tag_generator import GoldenMasterTagGenerator
from utils.multi_vehicle_tags import MultiVehicleTagGenerator
from openai import OpenAI

def test_10_products():
    """Test the two-pass AI approach with 10 vehicle-specific products"""
    
    print("🔍 TESTING 10 VEHICLE-SPECIFIC PRODUCTS")
    print("=" * 50)
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not found")
        return
    
    # Load test data - take first 10 from the 100-product dataset
    try:
        df = pd.read_csv('data/results/test_100_products.csv')
        df = df.head(10)  # Take first 10
        print(f"✅ Loaded {len(df)} test products")
    except Exception as e:
        print(f"❌ Could not load test data. Run create_100_product_test.py first: {e}")
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
    
    print(f"\n🚀 PROCESSING 10 PRODUCTS:")
    
    for idx, row in df.iterrows():
        print(f"\n   {idx+1:2d}/10: {row['Product Name'][:60]}...")
        
        # Create product info
        product_info = f"""
Product Name: {row['Product Name']}
Description: {row['Description']}
Stock Code: {row['StockCode']}
Price: ${row.get('MAP', 0)}
"""
        
        try:
            # Pass 1: Extract initial vehicle info (now extracts ALL vehicles)
            initial_data = two_pass_engine.extract_initial_vehicle_info(product_info)
            print(f"       Pass 1 Primary: {initial_data['year_min']}-{initial_data['year_max']} {initial_data['make']} {initial_data['model']}")
            
            # Show all vehicles found
            all_vehicles = initial_data.get('all_vehicles', [])
            if len(all_vehicles) > 1:
                print(f"       Pass 1 All Vehicles ({len(all_vehicles)}):")
                for i, vehicle in enumerate(all_vehicles):
                    print(f"          {i+1}. {vehicle['year_min']}-{vehicle['year_max']} {vehicle['make']} {vehicle['model']}")
            
            # Pass 2: Refine ALL vehicles with golden master
            refined_data = two_pass_engine.refine_with_golden_master(initial_data, product_info)
            
            # Get ALL refined vehicles from Pass 2
            refined_vehicles = refined_data.get('all_vehicles_refined', initial_data.get('all_vehicles', []))
            print(f"       Pass 2: Refined {len(refined_vehicles)} vehicles")
            for i, vehicle in enumerate(refined_vehicles):
                print(f"          {i+1}. {vehicle['year_min']}-{vehicle['year_max']} {vehicle['make']} {vehicle['model']}")
            
            # Generate tags for ALL REFINED vehicles from Pass 2
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
                print(f"       ✅ Multi-Vehicle Tags ({tag_count}): {tags[:80]}{'...' if len(tags) > 80 else ''}")
                # Show breakdown
                for vehicle_info in multi_result['vehicle_breakdown']:
                    if vehicle_info['tag_count'] > 0:
                        print(f"          • {vehicle_info['make']} {vehicle_info['model']}: {vehicle_info['tag_count']} tags")
            else:
                print(f"       🔄 No vehicle-specific tags generated")
            
            # Format all vehicles for debugging
            pass1_vehicles_debug = []
            for v in all_vehicles:
                pass1_vehicles_debug.append(f"{v['year_min']}-{v['year_max']} {v['make']} {v['model']}")
            pass1_all_vehicles_str = " | ".join(pass1_vehicles_debug)
            
            pass2_vehicles_debug = []
            for v in refined_vehicles:
                pass2_vehicles_debug.append(f"{v['year_min']}-{v['year_max']} {v['make']} {v['model']}")
            pass2_all_vehicles_str = " | ".join(pass2_vehicles_debug)
            
            # Store result with debugging data
            results.append({
                'ID': idx + 1,
                'SKU': row['StockCode'],
                'Product_Name': row['Product Name'],
                'Original_Description': row['Description'],
                'Pass1_All_Vehicles': pass1_all_vehicles_str,
                'Pass1_Vehicle_Count': len(all_vehicles),
                'Pass2_All_Vehicles': pass2_all_vehicles_str,
                'Pass2_Vehicle_Count': len(refined_vehicles),
                'Pass1_Make': initial_data['make'],
                'Pass1_Model': initial_data['model'],
                'Pass2_Make': refined_data['make'],
                'Pass2_Model': refined_data['model'],
                'Year_Range': f"{refined_data['year_min']}-{refined_data['year_max']}",
                'Final_Tags': tags,
                'Tags_Count': tag_count,
                'Vehicle_Summary': multi_tag_generator.format_vehicle_summary(multi_result)
            })
            
        except Exception as e:
            print(f"       ❌ ERROR: {str(e)}")
            results.append({
                'ID': idx + 1,
                'SKU': row['StockCode'],
                'Product_Name': row['Product Name'],
                'Original_Description': row['Description'],
                'Pass1_All_Vehicles': 'ERROR',
                'Pass1_Vehicle_Count': 0,
                'Pass2_All_Vehicles': 'ERROR',
                'Pass2_Vehicle_Count': 0,
                'Pass1_Make': 'ERROR',
                'Pass1_Model': 'ERROR',
                'Pass2_Make': 'ERROR',
                'Pass2_Model': 'ERROR',
                'Year_Range': 'ERROR',
                'Final_Tags': '',
                'Tags_Count': 0,
                'Vehicle_Summary': f'Error: {str(e)}'
            })
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    # Create DataFrame and save
    results_df = pd.DataFrame(results)
    
    # Create output file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f'data/results/test_10_products_{timestamp}.csv'
    
    os.makedirs('data/results', exist_ok=True)
    results_df.to_csv(output_path, index=False)
    
    print(f"\n✅ RESULTS SAVED: {output_path}")
    
    # Generate comprehensive statistics
    print(f"\n📊 TEST RESULTS SUMMARY:")
    print(f"   🕐 Total time: {total_duration:.1f} seconds")
    print(f"   ⚡ Average time per product: {total_duration/10:.1f} seconds")
    
    # Success rates
    successful = results_df[results_df['Pass2_Make'] != 'ERROR']
    print(f"   ✅ Successful processing: {len(successful)}/10 ({len(successful)/10*100:.1f}%)")
    
    # Tag generation statistics
    with_tags = results_df[results_df['Tags_Count'] > 0]
    print(f"   🏷️  Products with tags: {len(with_tags)}/10 ({len(with_tags)/10*100:.1f}%)")
    
    if len(with_tags) > 0:
        total_tags = with_tags['Tags_Count'].sum()
        avg_tags = with_tags['Tags_Count'].mean()
        max_tags = with_tags['Tags_Count'].max()
        print(f"   📊 Total tags generated: {total_tags}")
        print(f"   📊 Average tags per product: {avg_tags:.1f}")
        print(f"   📊 Maximum tags: {max_tags}")
    
    # Show all results
    print(f"\n📋 DETAILED RESULTS:")
    for _, row in results_df.iterrows():
        status = "✅" if row['Tags_Count'] > 0 else "❌" if row['Pass2_Make'] == 'ERROR' else "🔄"
        print(f"   {row['ID']:2d}. {status} {row['Product_Name'][:45]:45} → {row['Pass2_Make']} ({row['Tags_Count']} tags)")
        if row['Final_Tags']:
            print(f"       Tags: {row['Final_Tags'][:100]}{'...' if len(row['Final_Tags']) > 100 else ''}")
    
    # Performance metrics
    print(f"\n⚡ PERFORMANCE METRICS:")
    if len(with_tags) > 0:
        tags_per_minute = (with_tags['Tags_Count'].sum() * 60) / total_duration
        print(f"   🏷️  Tags generated per minute: {tags_per_minute:.1f}")
    
    products_per_minute = (10 * 60) / total_duration
    print(f"   📦 Products processed per minute: {products_per_minute:.1f}")
    
    # Extrapolate to 100 products
    print(f"\n📈 EXTRAPOLATION TO 100 PRODUCTS:")
    estimated_100_time = (total_duration * 10) / 60  # minutes
    print(f"   🕐 Estimated time for 100 products: {estimated_100_time:.1f} minutes")
    
    if len(with_tags) > 0:
        success_rate = len(with_tags) / 10
        estimated_tags = success_rate * 100 * avg_tags
        print(f"   🏷️  Estimated tags for 100 products: {estimated_tags:.0f}")
    
    print(f"\n🎉 10-PRODUCT TEST COMPLETED!")
    return output_path


if __name__ == "__main__":
    test_10_products()