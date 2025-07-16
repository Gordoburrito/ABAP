#!/usr/bin/env python3
"""
Debug Pass 2: Golden master refinement
Tests the second pass of the two-pass AI approach
"""

import os
import pandas as pd
import json
from openai import OpenAI
from utils.ai_extraction import TwoPassAIEngine

def test_pass2_refinement():
    """Test Pass 2 refinement independently"""
    
    print("🔍 DEBUGGING PASS 2 - GOLDEN MASTER REFINEMENT")
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
        print(f"✅ Golden master loaded: {len(golden_df)} records")
        print(f"   Columns: {list(golden_df.columns)}")
    except Exception as e:
        print(f"❌ Could not load golden master: {e}")
        return
    
    # Load Pass 1 results
    try:
        pass1_df = pd.read_csv('data/results/debug_pass1_results.csv')
        print(f"✅ Pass 1 results loaded: {len(pass1_df)} products")
    except Exception as e:
        print(f"❌ Could not load Pass 1 results. Run debug_pass1.py first: {e}")
        return
    
    # Load original test data
    try:
        original_df = pd.read_csv('data/results/debug_test_input.csv')
        print(f"✅ Original test data loaded: {len(original_df)} products")
    except Exception as e:
        print(f"❌ Could not load original test data: {e}")
        return
    
    # Initialize AI engine
    try:
        client = OpenAI(api_key=api_key)
        engine = TwoPassAIEngine(client, golden_df)
        print("✅ AI engine initialized")
    except Exception as e:
        print(f"❌ Could not initialize AI engine: {e}")
        return
    
    results = []
    
    for idx, row in pass1_df.iterrows():
        print(f"\n🔍 Processing {idx+1}/{len(pass1_df)}: {row['Product_Name']}")
        
        # Skip if Pass 1 failed
        if row['Pass1_Make'] == 'ERROR':
            print(f"   ⚠️  Skipping - Pass 1 failed")
            continue
        
        # Create Pass 1 data structure
        pass1_data = {
            'year_min': row['Pass1_Year_Min'],
            'year_max': row['Pass1_Year_Max'],
            'make': row['Pass1_Make'],
            'model': row['Pass1_Model'],
            'title': row['Pass1_Title'],
            'body_html': row['Pass1_Body_HTML']
        }
        
        # Get original product info
        original_row = original_df[original_df['StockCode'] == row['SKU']].iloc[0]
        product_info = f"""
Product Name: {original_row['Product Name']}
Description: {original_row['Description']}
Stock Code: {original_row['StockCode']}
Price: ${original_row['MAP']}
"""
        
        try:
            # Refine with golden master
            refined_data = engine.refine_with_golden_master(pass1_data, product_info)
            
            print(f"   ✅ Pass 2 Results:")
            print(f"      Year Range: {refined_data['year_min']}-{refined_data['year_max']}")
            print(f"      Make: {refined_data['make']}")
            print(f"      Model: {refined_data['model']}")
            print(f"      Title: {refined_data['title'][:60]}...")
            
            # Check if make exists in golden master
            make_exists = golden_df['make'].str.lower().str.contains(refined_data['make'].lower(), na=False).any()
            print(f"      Make in Golden Master: {'✅' if make_exists else '❌'}")
            
            # Store result
            results.append({
                'SKU': row['SKU'],
                'Product_Name': row['Product_Name'],
                'Pass1_Make': row['Pass1_Make'],
                'Pass1_Model': row['Pass1_Model'],
                'Pass2_Year_Min': refined_data['year_min'],
                'Pass2_Year_Max': refined_data['year_max'],
                'Pass2_Make': refined_data['make'],
                'Pass2_Model': refined_data['model'],
                'Pass2_Title': refined_data['title'],
                'Pass2_Body_HTML': refined_data['body_html'],
                'Make_In_Golden_Master': make_exists
            })
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            results.append({
                'SKU': row['SKU'],
                'Product_Name': row['Product_Name'],
                'Pass1_Make': row['Pass1_Make'],
                'Pass1_Model': row['Pass1_Model'],
                'Pass2_Year_Min': 'ERROR',
                'Pass2_Year_Max': 'ERROR',
                'Pass2_Make': 'ERROR',
                'Pass2_Model': 'ERROR',
                'Pass2_Title': f'Error: {str(e)}',
                'Pass2_Body_HTML': f'<p>Error: {str(e)}</p>',
                'Make_In_Golden_Master': False
            })
    
    # Save results
    results_df = pd.DataFrame(results)
    output_path = 'data/results/debug_pass2_results.csv'
    os.makedirs('data/results', exist_ok=True)
    results_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Pass 2 results saved to: {output_path}")
    print(f"📊 Processed {len(results)} products")
    
    # Show summary
    print(f"\n📋 PASS 2 SUMMARY:")
    if len(results_df) > 0:
        makes = results_df['Pass2_Make'].value_counts()
        print(f"   Makes found: {dict(makes)}")
        
        valid_makes = results_df['Make_In_Golden_Master'].sum()
        print(f"   Valid makes in golden master: {valid_makes}/{len(results)}")
        
        # Show changes from Pass 1 to Pass 2
        changes = results_df[results_df['Pass1_Make'] != results_df['Pass2_Make']]
        if len(changes) > 0:
            print(f"\n🔄 CHANGES FROM PASS 1 TO PASS 2:")
            for _, row in changes.iterrows():
                print(f"   {row['SKU']}: {row['Pass1_Make']} → {row['Pass2_Make']}")
    else:
        print(f"   No valid results processed")


if __name__ == "__main__":
    test_pass2_refinement()