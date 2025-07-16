#!/usr/bin/env python3
"""
Test batch processing with configurable product count
"""

import os
import pandas as pd
import time
from datetime import datetime
from utils.ai_extraction import TwoPassAIEngine
from utils.golden_master_tag_generator import GoldenMasterTagGenerator
from openai import OpenAI
import sys

def test_batch_products(batch_size=25):
    """Test batch processing with specified number of products"""
    
    print(f"🔍 TESTING {batch_size} VEHICLE-SPECIFIC PRODUCTS")
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
        df = df.head(batch_size)
        print(f"✅ Loaded {len(df)} test products")
    except Exception as e:
        print(f"❌ Could not load test data. Run create_100_product_test.py first: {e}")
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
    start_time = time.time()
    
    print(f"\n🚀 PROCESSING {batch_size} PRODUCTS:")
    print("   Progress: ", end="", flush=True)
    
    for idx, row in df.iterrows():
        # Progress indicator
        if (idx + 1) % 5 == 0:
            print(f"{idx+1}", end="", flush=True)
        else:
            print(".", end="", flush=True)
        
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
            
            # Generate tags
            tags = ""
            tag_count = 0
            
            if refined_data['make'].lower() not in ['universal', 'unknown']:
                # Parse models intelligently
                model_str = refined_data['model'].replace(' and ', ', ').replace('/', ', ').replace('|', ', ')
                models = [m.strip() for m in model_str.split(',') if m.strip()]
                
                vehicle_tags = tag_generator.generate_vehicle_tags_from_car_ids(
                    refined_data['year_min'], refined_data['year_max'], 
                    refined_data['make'], models
                )
                
                if vehicle_tags:
                    tags = ', '.join(vehicle_tags)
                    tag_count = len(vehicle_tags)
            
            # Store result
            results.append({
                'ID': idx + 1,
                'SKU': row['StockCode'],
                'Product_Name': row['Product Name'],
                'Original_Description': row['Description'],
                'Pass2_Make': refined_data['make'],
                'Pass2_Model': refined_data['model'],
                'Year_Range': f"{refined_data['year_min']}-{refined_data['year_max']}",
                'Final_Tags': tags,
                'Tags_Count': tag_count,
                'Status': 'SUCCESS'
            })
            
        except Exception as e:
            results.append({
                'ID': idx + 1,
                'SKU': row['StockCode'],
                'Product_Name': row['Product Name'],
                'Original_Description': row['Description'],
                'Pass2_Make': 'ERROR',
                'Pass2_Model': 'ERROR',
                'Year_Range': 'ERROR',
                'Final_Tags': '',
                'Tags_Count': 0,
                'Status': f'ERROR: {str(e)[:50]}'
            })
    
    print(f" ✅")  # Complete progress line
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    # Create DataFrame and save
    results_df = pd.DataFrame(results)
    
    # Create output file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f'data/results/test_batch_{batch_size}_products_{timestamp}.csv'
    
    os.makedirs('data/results', exist_ok=True)
    results_df.to_csv(output_path, index=False)
    
    print(f"\n✅ RESULTS SAVED: {output_path}")
    
    # Generate comprehensive statistics
    print(f"\n📊 TEST RESULTS SUMMARY:")
    print(f"   🕐 Total time: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
    print(f"   ⚡ Average time per product: {total_duration/batch_size:.1f} seconds")
    
    # Success rates
    successful = results_df[results_df['Status'] == 'SUCCESS']
    print(f"   ✅ Successful processing: {len(successful)}/{batch_size} ({len(successful)/batch_size*100:.1f}%)")
    
    # Tag generation statistics
    with_tags = results_df[results_df['Tags_Count'] > 0]
    print(f"   🏷️  Products with tags: {len(with_tags)}/{batch_size} ({len(with_tags)/batch_size*100:.1f}%)")
    
    if len(with_tags) > 0:
        total_tags = with_tags['Tags_Count'].sum()
        avg_tags = with_tags['Tags_Count'].mean()
        max_tags = with_tags['Tags_Count'].max()
        print(f"   📊 Total tags generated: {total_tags}")
        print(f"   📊 Average tags per tagged product: {avg_tags:.1f}")
        print(f"   📊 Maximum tags for one product: {max_tags}")
    
    # Make analysis
    make_counts = successful['Pass2_Make'].value_counts()
    print(f"\n📋 TOP MAKES IDENTIFIED:")
    for make, count in make_counts.head(8).items():
        make_with_tags = len(with_tags[with_tags['Pass2_Make'] == make])
        print(f"   {make:15}: {count:2d} products ({make_with_tags:2d} with tags)")
    
    # Show successful examples
    print(f"\n✅ SAMPLE SUCCESSFUL TAG GENERATIONS:")
    tag_examples = with_tags.head(8)
    for _, row in tag_examples.iterrows():
        tags_preview = row['Final_Tags'][:60] + "..." if len(row['Final_Tags']) > 60 else row['Final_Tags']
        print(f"   {row['ID']:2d}. {row['Pass2_Make']:12} → {tags_preview}")
    
    # Performance metrics
    print(f"\n⚡ PERFORMANCE METRICS:")
    if len(with_tags) > 0:
        tags_per_minute = (with_tags['Tags_Count'].sum() * 60) / total_duration
        print(f"   🏷️  Tags generated per minute: {tags_per_minute:.1f}")
    
    products_per_minute = (batch_size * 60) / total_duration
    print(f"   📦 Products processed per minute: {products_per_minute:.1f}")
    
    # Extrapolate to 100 products
    print(f"\n📈 EXTRAPOLATION TO 100 PRODUCTS:")
    estimated_100_time = (total_duration * 100 / batch_size) / 60  # minutes
    print(f"   🕐 Estimated time for 100 products: {estimated_100_time:.1f} minutes")
    
    if len(with_tags) > 0:
        success_rate = len(with_tags) / batch_size
        estimated_tags = success_rate * 100 * avg_tags
        print(f"   🏷️  Estimated tags for 100 products: {estimated_tags:.0f}")
    
    print(f"\n🎉 {batch_size}-PRODUCT BATCH TEST COMPLETED!")
    return output_path


if __name__ == "__main__":
    # Allow command line argument for batch size
    batch_size = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    test_batch_products(batch_size)