#!/usr/bin/env python3
"""
Debug what Lincoln models are in the golden master
"""

import pandas as pd

def debug_golden_lincoln():
    print("🔍 DEBUGGING GOLDEN MASTER LINCOLN MODELS")
    print("=" * 50)
    
    # Load golden master
    try:
        golden_df = pd.read_csv('../shared/data/master_ultimate_golden.csv', low_memory=False)
        golden_df.columns = [col.lower().replace(' ', '_') for col in golden_df.columns]
        print(f"✅ Golden master loaded: {len(golden_df)} records")
        
    except Exception as e:
        print(f"❌ Could not load golden master: {e}")
        return
    
    # Filter for Lincoln 1931-1939 
    lincoln_df = golden_df[
        (golden_df['make'] == 'Lincoln') &
        (pd.to_numeric(golden_df['year'], errors='coerce') >= 1931) &
        (pd.to_numeric(golden_df['year'], errors='coerce') <= 1939)
    ].dropna(subset=['year'])
    
    print(f"\n📊 LINCOLN 1931-1939 MODELS IN GOLDEN MASTER:")
    print(f"Total records: {len(lincoln_df)}")
    
    # Get unique models
    unique_models = sorted(lincoln_df['model'].dropna().unique())
    print(f"\nUnique models ({len(unique_models)}):")
    for model in unique_models:
        count = len(lincoln_df[lincoln_df['model'] == model])
        years = sorted(pd.to_numeric(lincoln_df[lincoln_df['model'] == model]['year'], errors='coerce').dropna().astype(int).unique())
        year_range = f"{min(years)}-{max(years)}" if years else "No years"
        print(f"  - {model}: {count} records ({year_range})")
    
    # Check specifically for our models
    target_models = ['Model K', 'KA', 'KB', 'Series K']
    print(f"\n🎯 CHECKING TARGET MODELS:")
    for model in target_models:
        matches = lincoln_df[lincoln_df['model'] == model]
        if len(matches) > 0:
            years = sorted(pd.to_numeric(matches['year'], errors='coerce').dropna().astype(int).unique())
            print(f"✅ {model}: {len(matches)} records ({min(years)}-{max(years)})")
        else:
            print(f"❌ {model}: NOT FOUND")
    
    # Check for partial matches
    print(f"\n🔍 CHECKING FOR PARTIAL MATCHES:")
    for model in target_models:
        if model not in unique_models:
            # Look for partial matches
            partial_matches = [m for m in unique_models if model.lower() in m.lower() or m.lower() in model.lower()]
            if partial_matches:
                print(f"🔍 {model} -> Possible matches: {partial_matches}")

if __name__ == "__main__":
    debug_golden_lincoln()