#!/usr/bin/env python3
"""
Debug Golden Master: Lookup and validation
Tests golden master data access and validation
"""

import os
import pandas as pd
from utils.golden_master_tag_generator import GoldenMasterTagGenerator

def test_golden_master_lookup():
    """Test golden master lookup independently"""
    
    print("🔍 DEBUGGING GOLDEN MASTER - LOOKUP & VALIDATION")
    print("=" * 50)
    
    # Load golden master
    try:
        golden_df = pd.read_csv('../shared/data/master_ultimate_golden.csv', low_memory=False)
        print(f"✅ Golden master loaded: {len(golden_df)} records")
        print(f"   Original columns: {list(golden_df.columns)}")
        
        # Normalize column names
        golden_df.columns = [col.lower().replace(' ', '_') for col in golden_df.columns]
        print(f"   Normalized columns: {list(golden_df.columns)}")
        
    except Exception as e:
        print(f"❌ Could not load golden master: {e}")
        return
    
    # Initialize tag generator
    try:
        tag_generator = GoldenMasterTagGenerator(golden_df)
        print("✅ Tag generator initialized")
    except Exception as e:
        print(f"❌ Could not initialize tag generator: {e}")
        return
    
    # Show sample data
    print(f"\n📋 GOLDEN MASTER SAMPLE DATA:")
    sample = golden_df.head(10)[['year', 'make', 'model', 'car_id']]
    for _, row in sample.iterrows():
        print(f"   {row['year']} {row['make']} {row['model']} -> {row['car_id']}")
    
    # Test specific makes
    test_makes = ['Cord', 'Hupmobile', 'AC', 'Ford', 'Chevrolet', 'Universal', 'Unknown']
    
    print(f"\n🔍 TESTING MAKE LOOKUPS:")
    for make in test_makes:
        matches = golden_df[golden_df['make'].str.lower() == make.lower()]
        print(f"   {make}: {len(matches)} matches")
        if len(matches) > 0:
            models = matches['model'].unique()[:5]  # Show first 5 models
            print(f"      Sample models: {list(models)}")
    
    # Test tag generation with Pass 2 results
    try:
        pass2_df = pd.read_csv('data/results/debug_pass2_results.csv')
        print(f"\n✅ Pass 2 results loaded: {len(pass2_df)} products")
    except Exception as e:
        print(f"\n❌ Could not load Pass 2 results. Run debug_pass2.py first: {e}")
        return
    
    print(f"\n🔍 TESTING TAG GENERATION:")
    results = []
    
    for idx, row in pass2_df.iterrows():
        print(f"\n   Product {idx+1}: {row['Product_Name'][:50]}...")
        
        # Skip if Pass 2 failed
        if row['Pass2_Make'] == 'ERROR':
            print(f"      ⚠️  Skipping - Pass 2 failed")
            continue
        
        try:
            # Parse models
            models = []
            if isinstance(row['Pass2_Model'], str):
                # Handle "810 and 812", "810/812", etc.
                model_str = row['Pass2_Model'].replace(' and ', ', ').replace('/', ', ').replace('|', ', ')
                models = [m.strip() for m in model_str.split(',') if m.strip()]
            
            print(f"      Year Range: {row['Pass2_Year_Min']}-{row['Pass2_Year_Max']}")
            print(f"      Make: {row['Pass2_Make']}")
            print(f"      Models: {models}")
            
            # Generate tags
            tags = tag_generator.generate_vehicle_tags_from_car_ids(
                row['Pass2_Year_Min'], row['Pass2_Year_Max'], 
                row['Pass2_Make'], models
            )
            
            if tags:
                print(f"      ✅ Generated tags: {tags}")
            else:
                print(f"      ❌ No tags generated")
                
                # Debug why no tags
                make_matches = golden_df[golden_df['make'].str.lower() == row['Pass2_Make'].lower()]
                print(f"         Make '{row['Pass2_Make']}' matches: {len(make_matches)}")
                
                if len(make_matches) > 0 and models:
                    for model in models:
                        model_matches = make_matches[make_matches['model'].str.lower() == model.lower()]
                        print(f"         Model '{model}' matches: {len(model_matches)}")
                        
                        if len(model_matches) > 0:
                            year_matches = model_matches[
                                (model_matches['year'] >= row['Pass2_Year_Min']) & 
                                (model_matches['year'] <= row['Pass2_Year_Max'])
                            ]
                            print(f"         Year range matches: {len(year_matches)}")
            
            # Store result
            results.append({
                'SKU': row['SKU'],
                'Product_Name': row['Product_Name'],
                'Make': row['Pass2_Make'],
                'Model': row['Pass2_Model'],
                'Year_Min': row['Pass2_Year_Min'],
                'Year_Max': row['Pass2_Year_Max'],
                'Tags': ', '.join(tags) if tags else '',
                'Tags_Count': len(tags) if tags else 0,
                'Make_In_Golden_Master': row['Make_In_Golden_Master']
            })
            
        except Exception as e:
            print(f"      ❌ Error: {str(e)}")
            results.append({
                'SKU': row['SKU'],
                'Product_Name': row['Product_Name'],
                'Make': row['Pass2_Make'],
                'Model': row['Pass2_Model'],
                'Year_Min': row['Pass2_Year_Min'],
                'Year_Max': row['Pass2_Year_Max'],
                'Tags': f'Error: {str(e)}',
                'Tags_Count': 0,
                'Make_In_Golden_Master': False
            })
    
    # Save results
    results_df = pd.DataFrame(results)
    output_path = 'data/results/debug_golden_master_results.csv'
    os.makedirs('data/results', exist_ok=True)
    results_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Golden master results saved to: {output_path}")
    print(f"📊 Processed {len(results)} products")
    
    # Show summary
    print(f"\n📋 GOLDEN MASTER SUMMARY:")
    print(f"   Products with tags: {len(results_df[results_df['Tags_Count'] > 0])}/{len(results_df)}")
    print(f"   Products with valid makes: {results_df['Make_In_Golden_Master'].sum()}/{len(results_df)}")
    
    tag_counts = results_df['Tags_Count'].value_counts().sort_index()
    print(f"   Tag count distribution: {dict(tag_counts)}")
    
    # Show successful tag generations
    successful = results_df[results_df['Tags_Count'] > 0]
    if len(successful) > 0:
        print(f"\n✅ SUCCESSFUL TAG GENERATIONS:")
        for _, row in successful.iterrows():
            print(f"   {row['SKU']}: {row['Tags']}")


if __name__ == "__main__":
    test_golden_master_lookup()