#!/usr/bin/env python3
"""
Test 100 vehicle-specific products with the two-pass AI approach
"""

import os
import pandas as pd
import time
from datetime import datetime
from utils.ai_extraction import TwoPassAIEngine
from utils.golden_master_tag_generator import GoldenMasterTagGenerator
from utils.multi_vehicle_tags import MultiVehicleTagGenerator
from openai import OpenAI

def test_100_products():
    """Test the two-pass AI approach with 100 vehicle-specific products"""
    
    print("🔍 TESTING 100 VEHICLE-SPECIFIC PRODUCTS")
    print("=" * 55)
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not found")
        return
    
    # Load test data
    try:
        df = pd.read_csv('data/results/test_100_products.csv')
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
    
    print(f"\n🚀 PROCESSING 100 PRODUCTS:")
    
    for idx, row in df.iterrows():
        print(f"   {idx+1:3d}/100: {row['Product Name'][:50]}...", end="")
        
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
            
            # Pass 2: Refine with golden master
            refined_data = two_pass_engine.refine_with_golden_master(initial_data, product_info)
            
            # Generate tags for ALL REFINED vehicles from Pass 2
            # Use refined vehicles if available, otherwise fall back to Pass 1
            refined_vehicles = refined_data.get('all_vehicles_refined', initial_data.get('all_vehicles', []))
            
            # Create a data structure for multi-vehicle tag generation using refined vehicles
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
                print(f" ✅ ({tag_count} tags)")
            else:
                print(f" ❌ (no tags)")
            
            # Format all vehicles from Pass 1 for debugging
            all_vehicles = initial_data.get('all_vehicles', [])
            pass1_vehicles_debug = []
            for v in all_vehicles:
                pass1_vehicles_debug.append(f"{v['year_min']}-{v['year_max']} {v['make']} {v['model']}")
            pass1_all_vehicles_str = " | ".join(pass1_vehicles_debug)
            
            # Store result with debugging data
            results.append({
                'ID': idx + 1,
                'SKU': row['StockCode'],
                'Product_Name': row['Product Name'],
                'Original_Description': row['Description'],
                'Pass1_Primary_Vehicle': f"{initial_data['year_min']}-{initial_data['year_max']} {initial_data['make']} {initial_data['model']}",
                'Pass1_All_Vehicles': pass1_all_vehicles_str,
                'Pass1_Vehicle_Count': len(all_vehicles),
                'Pass2_Refined_Vehicle': f"{refined_data['year_min']}-{refined_data['year_max']} {refined_data['make']} {refined_data['model']}",
                'Pass1_Year_Min': initial_data['year_min'],
                'Pass1_Year_Max': initial_data['year_max'],
                'Pass1_Make': initial_data['make'],
                'Pass1_Model': initial_data['model'],
                'Pass2_Year_Min': refined_data['year_min'],
                'Pass2_Year_Max': refined_data['year_max'],
                'Pass2_Make': refined_data['make'],
                'Pass2_Model': refined_data['model'],
                'Final_Tags': tags,
                'Tags_Count': tag_count,
                'Title': refined_data['title'],
                'Body_HTML': refined_data['body_html']
            })
            
        except Exception as e:
            print(f" ❌ ERROR: {str(e)[:30]}...")
            results.append({
                'ID': idx + 1,
                'SKU': row['StockCode'],
                'Product_Name': row['Product Name'],
                'Original_Description': row['Description'],
                'Pass1_Primary_Vehicle': 'ERROR',
                'Pass1_All_Vehicles': 'ERROR',
                'Pass1_Vehicle_Count': 0,
                'Pass2_Refined_Vehicle': 'ERROR',
                'Pass1_Year_Min': 'ERROR',
                'Pass1_Year_Max': 'ERROR',
                'Pass1_Make': 'ERROR',
                'Pass1_Model': 'ERROR',
                'Pass2_Year_Min': 'ERROR',
                'Pass2_Year_Max': 'ERROR',
                'Pass2_Make': 'ERROR',
                'Pass2_Model': 'ERROR',
                'Final_Tags': '',
                'Tags_Count': 0,
                'Title': f'Error: {str(e)}',
                'Body_HTML': f'<p>Error: {str(e)}</p>'
            })
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    # Create DataFrame and save
    results_df = pd.DataFrame(results)
    
    # Create output files
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    detailed_output = f'data/results/test_100_detailed_{timestamp}.csv'
    summary_output = f'data/results/test_100_summary_{timestamp}.csv'
    
    os.makedirs('data/results', exist_ok=True)
    results_df.to_csv(detailed_output, index=False)
    
    # Create summary with just essential columns
    summary_df = results_df[['ID', 'SKU', 'Product_Name', 'Original_Description', 'Pass2_Make', 'Pass2_Model', 'Final_Tags', 'Tags_Count']].copy()
    summary_df.to_csv(summary_output, index=False)
    
    print(f"\n✅ RESULTS SAVED:")
    print(f"   📋 Detailed: {detailed_output}")
    print(f"   📊 Summary: {summary_output}")
    
    # Generate comprehensive statistics
    print(f"\n📊 TEST RESULTS SUMMARY:")
    print(f"   🕐 Total time: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
    print(f"   ⚡ Average time per product: {total_duration/100:.1f} seconds")
    print(f"   📦 Products processed: {len(results_df)}")
    
    # Success rates
    successful = results_df[results_df['Pass2_Make'] != 'ERROR']
    print(f"   ✅ Successful processing: {len(successful)}/{len(results_df)} ({len(successful)/len(results_df)*100:.1f}%)")
    
    # Tag generation statistics
    with_tags = results_df[results_df['Tags_Count'] > 0]
    print(f"   🏷️  Products with tags: {len(with_tags)}/{len(results_df)} ({len(with_tags)/len(results_df)*100:.1f}%)")
    
    if len(with_tags) > 0:
        avg_tags = with_tags['Tags_Count'].mean()
        max_tags = with_tags['Tags_Count'].max()
        print(f"   📊 Average tags per product: {avg_tags:.1f}")
        print(f"   📊 Maximum tags: {max_tags}")
    
    # Make analysis
    make_analysis = successful['Pass2_Make'].value_counts()
    print(f"\n📋 TOP MAKES IDENTIFIED:")
    for make, count in make_analysis.head(10).items():
        make_with_tags = len(with_tags[with_tags['Pass2_Make'] == make])
        print(f"   {make}: {count} products ({make_with_tags} with tags)")
    
    # Show successful tag examples
    print(f"\n✅ SUCCESSFUL TAG EXAMPLES:")
    tag_examples = with_tags.head(10)
    for _, row in tag_examples.iterrows():
        tags_preview = row['Final_Tags'][:80] + "..." if len(row['Final_Tags']) > 80 else row['Final_Tags']
        print(f"   {row['ID']:3d}. {row['Product_Name'][:40]:40} → {tags_preview}")
    
    # Performance metrics
    print(f"\n⚡ PERFORMANCE METRICS:")
    if len(with_tags) > 0:
        tags_per_minute = (with_tags['Tags_Count'].sum() * 60) / total_duration
        print(f"   🏷️  Tags generated per minute: {tags_per_minute:.1f}")
    
    products_per_minute = (len(results_df) * 60) / total_duration
    print(f"   📦 Products processed per minute: {products_per_minute:.1f}")
    
    # Cost estimation (rough)
    if len(successful) > 0:
        estimated_tokens_per_product = 2000  # Conservative estimate
        total_tokens = len(successful) * estimated_tokens_per_product
        estimated_cost = (total_tokens / 1000) * 0.00015  # gpt-4.1-mini pricing
        print(f"   💰 Estimated API cost: ${estimated_cost:.2f}")
    
    print(f"\n🎉 100-PRODUCT TEST COMPLETED!")
    return detailed_output, summary_output


if __name__ == "__main__":
    test_100_products()