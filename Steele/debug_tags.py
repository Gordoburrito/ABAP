#!/usr/bin/env python3
"""
Debug Tags: Final tag generation and formatting
Tests the final tag generation process
"""

import os
import pandas as pd
from utils.golden_master_tag_generator import GoldenMasterTagGenerator

def test_tag_generation():
    """Test tag generation independently"""
    
    print("🔍 DEBUGGING TAGS - FINAL TAG GENERATION")
    print("=" * 50)
    
    # Load golden master
    try:
        golden_df = pd.read_csv('../shared/data/master_ultimate_golden.csv', low_memory=False)
        golden_df.columns = [col.lower().replace(' ', '_') for col in golden_df.columns]
        tag_generator = GoldenMasterTagGenerator(golden_df)
        print(f"✅ Golden master loaded: {len(golden_df)} records")
    except Exception as e:
        print(f"❌ Could not load golden master: {e}")
        return
    
    # Load golden master results
    try:
        golden_results_df = pd.read_csv('data/results/debug_golden_master_results.csv')
        print(f"✅ Golden master results loaded: {len(golden_results_df)} products")
    except Exception as e:
        print(f"❌ Could not load golden master results. Run debug_golden_master.py first: {e}")
        return
    
    print(f"\n🔍 TESTING TAG GENERATION SCENARIOS:")
    
    # Test specific vehicle combinations
    test_scenarios = [
        {
            'name': 'Cord 810 (1936-1937)',
            'year_min': 1936,
            'year_max': 1937,
            'make': 'Cord',
            'models': ['810', '812']
        },
        {
            'name': 'Hupmobile Skylark (1939-1941)',
            'year_min': 1939,
            'year_max': 1941,
            'make': 'Hupmobile',
            'models': ['Skylark']
        },
        {
            'name': 'AC (Invalid make)',
            'year_min': 1900,
            'year_max': 2024,
            'make': 'AC',
            'models': ['100 Series']
        },
        {
            'name': 'Universal (No specific vehicle)',
            'year_min': 1900,
            'year_max': 2024,
            'make': 'Universal',
            'models': ['Universal']
        },
        {
            'name': 'Ford Mustang (Should work)',
            'year_min': 1964,
            'year_max': 1970,
            'make': 'Ford',
            'models': ['Mustang']
        }
    ]
    
    results = []
    
    for scenario in test_scenarios:
        print(f"\n   Testing: {scenario['name']}")
        print(f"      Year Range: {scenario['year_min']}-{scenario['year_max']}")
        print(f"      Make: {scenario['make']}")
        print(f"      Models: {scenario['models']}")
        
        try:
            # Generate tags
            tags = tag_generator.generate_vehicle_tags_from_car_ids(
                scenario['year_min'], scenario['year_max'], 
                scenario['make'], scenario['models']
            )
            
            if tags:
                print(f"      ✅ Generated tags: {tags}")
                print(f"      📊 Tag count: {len(tags)}")
            else:
                print(f"      ❌ No tags generated")
                
                # Debug why no tags
                make_matches = golden_df[golden_df['make'].str.lower() == scenario['make'].lower()]
                print(f"         Make '{scenario['make']}' matches: {len(make_matches)}")
                
                if len(make_matches) > 0:
                    for model in scenario['models']:
                        model_matches = make_matches[make_matches['model'].str.lower() == model.lower()]
                        print(f"         Model '{model}' matches: {len(model_matches)}")
                        
                        if len(model_matches) > 0:
                            year_matches = model_matches[
                                (model_matches['year'] >= scenario['year_min']) & 
                                (model_matches['year'] <= scenario['year_max'])
                            ]
                            print(f"         Year range matches: {len(year_matches)}")
                            if len(year_matches) > 0:
                                print(f"         Sample car_ids: {list(year_matches['car_id'].head())}")
            
            # Store result
            results.append({
                'Scenario': scenario['name'],
                'Year_Min': scenario['year_min'],
                'Year_Max': scenario['year_max'],
                'Make': scenario['make'],
                'Models': ', '.join(scenario['models']),
                'Tags': ', '.join(tags) if tags else '',
                'Tags_Count': len(tags) if tags else 0,
                'Success': len(tags) > 0 if tags else False
            })
            
        except Exception as e:
            print(f"      ❌ Error: {str(e)}")
            results.append({
                'Scenario': scenario['name'],
                'Year_Min': scenario['year_min'],
                'Year_Max': scenario['year_max'],
                'Make': scenario['make'],
                'Models': ', '.join(scenario['models']),
                'Tags': f'Error: {str(e)}',
                'Tags_Count': 0,
                'Success': False
            })
    
    # Test with actual product data
    print(f"\n🔍 TESTING WITH ACTUAL PRODUCT DATA:")
    
    for idx, row in golden_results_df.iterrows():
        if row['Tags_Count'] > 0:  # Only show successful ones
            print(f"\n   Product {idx+1}: {row['Product_Name'][:50]}...")
            print(f"      Make: {row['Make']}")
            print(f"      Model: {row['Model']}")
            print(f"      Year Range: {row['Year_Min']}-{row['Year_Max']}")
            print(f"      Generated Tags: {row['Tags']}")
            
            # Add to results
            results.append({
                'Scenario': f"Product: {row['Product_Name'][:30]}...",
                'Year_Min': row['Year_Min'],
                'Year_Max': row['Year_Max'],
                'Make': row['Make'],
                'Models': row['Model'],
                'Tags': row['Tags'],
                'Tags_Count': row['Tags_Count'],
                'Success': True
            })
    
    # Save results
    results_df = pd.DataFrame(results)
    output_path = 'data/results/debug_tags_results.csv'
    os.makedirs('data/results', exist_ok=True)
    results_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Tag generation results saved to: {output_path}")
    print(f"📊 Tested {len(results)} scenarios")
    
    # Show summary
    print(f"\n📋 TAG GENERATION SUMMARY:")
    successful = results_df[results_df['Success'] == True]
    print(f"   Successful tag generations: {len(successful)}/{len(results_df)}")
    
    if len(successful) > 0:
        avg_tags = successful['Tags_Count'].mean()
        print(f"   Average tags per successful generation: {avg_tags:.1f}")
        
        makes_with_tags = successful['Make'].value_counts()
        print(f"   Makes with successful tags: {dict(makes_with_tags)}")
    
    # Show tag format examples
    print(f"\n📋 TAG FORMAT EXAMPLES:")
    for _, row in successful.head(5).iterrows():
        if row['Tags']:
            sample_tags = row['Tags'].split(', ')[:3]  # Show first 3 tags
            print(f"   {row['Make']}: {', '.join(sample_tags)}")


if __name__ == "__main__":
    test_tag_generation()