#!/usr/bin/env python3
"""
Debug Pass 1: Initial AI extraction
Tests the first pass of the two-pass AI approach
"""

import os
import pandas as pd
import json
from openai import OpenAI
from utils.ai_extraction import TwoPassAIEngine

def test_pass1_extraction():
    """Test Pass 1 AI extraction independently"""
    
    print("🔍 DEBUGGING PASS 1 - INITIAL AI EXTRACTION")
    print("=" * 50)
    
    # Check API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not found")
        return
    
    # Load test data
    try:
        df = pd.read_csv('data/results/debug_test_input.csv')
        print(f"📋 Loaded {len(df)} test products")
    except Exception as e:
        print(f"❌ Could not load test data: {e}")
        return
    
    # Load golden master to pass to AI engine
    try:
        golden_df = pd.read_csv('../shared/data/master_ultimate_golden.csv', low_memory=False)
        print(f"✅ Golden master loaded: {len(golden_df)} records")
        print(f"   Original columns: {list(golden_df.columns)}")
        
        # Initialize AI engine
        client = OpenAI(api_key=api_key)
        engine = TwoPassAIEngine(client, golden_df)
        print("✅ AI engine initialized")
    except Exception as e:
        print(f"❌ Could not initialize AI engine: {e}")
        return
    
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
            # Extract initial vehicle info
            initial_data = engine.extract_initial_vehicle_info(product_info)
            
            print(f"   ✅ Pass 1 Results:")
            print(f"      Year Range: {initial_data['year_min']}-{initial_data['year_max']}")
            print(f"      Make: {initial_data['make']}")
            print(f"      Model: {initial_data['model']}")
            print(f"      Title: {initial_data['title'][:60]}...")
            
            # Store result
            results.append({
                'SKU': row['StockCode'],
                'Product_Name': row['Product Name'],
                'Pass1_Year_Min': initial_data['year_min'],
                'Pass1_Year_Max': initial_data['year_max'],
                'Pass1_Make': initial_data['make'],
                'Pass1_Model': initial_data['model'],
                'Pass1_Title': initial_data['title'],
                'Pass1_Body_HTML': initial_data['body_html']
            })
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            import traceback
            print(f"   ❌ Traceback: {traceback.format_exc()}")
            results.append({
                'SKU': row['StockCode'],
                'Product_Name': row['Product Name'],
                'Pass1_Year_Min': 'ERROR',
                'Pass1_Year_Max': 'ERROR',
                'Pass1_Make': 'ERROR',
                'Pass1_Model': 'ERROR',
                'Pass1_Title': f'Error: {str(e)}',
                'Pass1_Body_HTML': f'<p>Error: {str(e)}</p>'
            })
    
    # Save results
    results_df = pd.DataFrame(results)
    output_path = 'data/results/debug_pass1_results.csv'
    os.makedirs('data/results', exist_ok=True)
    results_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Pass 1 results saved to: {output_path}")
    print(f"📊 Processed {len(results)} products")
    
    # Show summary
    print(f"\n📋 PASS 1 SUMMARY:")
    makes = results_df['Pass1_Make'].value_counts()
    print(f"   Makes found: {dict(makes)}")
    
    year_ranges = results_df.apply(lambda x: f"{x['Pass1_Year_Min']}-{x['Pass1_Year_Max']}", axis=1).value_counts()
    print(f"   Year ranges: {dict(year_ranges.head())}")


if __name__ == "__main__":
    test_pass1_extraction()