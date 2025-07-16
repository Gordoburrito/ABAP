#!/usr/bin/env python3
"""
Test script to create simplified output with debugging data
"""

import os
import pandas as pd
import logging
from utils.ai_extraction import TwoPassAIEngine
from utils.golden_master_tag_generator import GoldenMasterTagGenerator
from openai import OpenAI

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def create_simplified_debug_output():
    """Create simplified CSV with essential columns plus debug data"""
    
    print("🔧 CREATING SIMPLIFIED DEBUG OUTPUT")
    print("=" * 50)
    
    # Load sample data
    try:
        df = pd.read_excel('data/raw/steele.xlsx')
        sample_df = df.head(10)  # 10 products
        
        print(f"📋 Processing {len(sample_df)} products")
        
    except Exception as e:
        print(f"❌ Could not load steele.xlsx: {e}")
        return
    
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not found")
        return
    
    # Load golden master
    try:
        golden_df = pd.read_csv('../shared/data/master_ultimate_golden.csv')
        # Normalize column names
        golden_df.columns = [col.lower().replace(' ', '_') for col in golden_df.columns]
        tag_generator = GoldenMasterTagGenerator(golden_df)
        
        # Initialize AI engine
        client = OpenAI(api_key=api_key)
        two_pass_engine = TwoPassAIEngine(client, golden_df)
        
    except Exception as e:
        print(f"❌ Could not load golden master: {e}")
        return
    
    # Process each product
    results = []
    
    for idx, row in sample_df.iterrows():
        print(f"\n🔍 Processing {idx+1}/{len(sample_df)}: {row['Product Name'][:50]}...")
        
        # Create product info
        product_info = f"""
Product Name: {row['Product Name']}
Description: {row.get('Description', '')}
Stock Code: {row['StockCode']}
Price: ${row.get('MAP', 0)}
"""
        
        try:
            # Pass 1: Extract initial vehicle info
            initial_data = two_pass_engine.extract_initial_vehicle_info(product_info)
            print(f"   Pass 1: {initial_data['year_min']}-{initial_data['year_max']} {initial_data['make']} {initial_data['model']}")
            
            # Pass 2: Refine with golden master
            refined_data = two_pass_engine.refine_with_golden_master(initial_data, product_info)
            print(f"   Pass 2: {refined_data['make']} {refined_data['model']}")
            
            # Generate tags
            models = []
            if isinstance(refined_data['model'], str):
                models = [m.strip() for m in refined_data['model'].replace(',', '|').split('|') if m.strip()]
            
            tags = ""
            if refined_data['make'].lower() not in ['universal', 'unknown'] and models:
                vehicle_tags = tag_generator.generate_vehicle_tags_from_car_ids(
                    refined_data['year_min'], refined_data['year_max'], 
                    refined_data['make'], models
                )
                if vehicle_tags:
                    tags = ', '.join(vehicle_tags)
                    print(f"   Tags: {tags}")
                else:
                    print(f"   Tags: (none - make '{refined_data['make']}' not found in golden master)")
            else:
                print(f"   Tags: (none - universal product)")
            
            # Store result
            results.append({
                'ID': '',
                'Command': 'MERGE',
                'Title': refined_data['title'],
                'Body HTML': refined_data['body_html'],
                'Vendor': refined_data['make'] if refined_data['make'] != 'Universal' else 'Universal',
                'Tags': tags,
                'Pass1_Year_Min': initial_data['year_min'],
                'Pass1_Year_Max': initial_data['year_max'],
                'Pass1_Make': initial_data['make'],
                'Pass1_Model': initial_data['model'],
                'Pass2_Make': refined_data['make'],
                'Pass2_Model': refined_data['model'],
                'SKU': row['StockCode']
            })
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            # Add error result
            results.append({
                'ID': '',
                'Command': 'MERGE',
                'Title': f"Error processing {row['Product Name']}",
                'Body HTML': f"<p>Error: {str(e)}</p>",
                'Vendor': 'Unknown',
                'Tags': '',
                'Pass1_Year_Min': '',
                'Pass1_Year_Max': '',
                'Pass1_Make': '',
                'Pass1_Model': '',
                'Pass2_Make': '',
                'Pass2_Model': '',
                'SKU': row['StockCode']
            })
    
    # Create DataFrame and save
    results_df = pd.DataFrame(results)
    
    # Create output file
    output_path = 'data/results/simplified_debug_output.csv'
    os.makedirs('data/results', exist_ok=True)
    results_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Simplified output saved to: {output_path}")
    print(f"📊 Columns: {list(results_df.columns)}")
    print(f"📋 Rows: {len(results_df)}")
    
    # Show sample
    print(f"\n📋 Sample Results:")
    for i, row in results_df.head(3).iterrows():
        print(f"   {i+1}. {row['Title'][:50]}...")
        print(f"      Vendor: {row['Vendor']}")
        print(f"      Tags: {row['Tags'] if row['Tags'] else '(empty)'}")
        print(f"      Pass1: {row['Pass1_Year_Min']}-{row['Pass1_Year_Max']} {row['Pass1_Make']} {row['Pass1_Model']}")
        print(f"      Pass2: {row['Pass2_Make']} {row['Pass2_Model']}")


if __name__ == "__main__":
    create_simplified_debug_output()