#!/usr/bin/env python3
"""
Debug what Studebaker models are in the golden master
"""

import pandas as pd

def debug_studebaker():
    print("🔍 DEBUGGING GOLDEN MASTER STUDEBAKER MODELS")
    print("=" * 55)
    
    # Load golden master
    try:
        golden_df = pd.read_csv('../shared/data/master_ultimate_golden.csv', low_memory=False)
        golden_df.columns = [col.lower().replace(' ', '_') for col in golden_df.columns]
        print(f"✅ Golden master loaded: {len(golden_df)} records")
        
    except Exception as e:
        print(f"❌ Could not load golden master: {e}")
        return
    
    # Filter for Studebaker 1938-1939 
    studebaker_df = golden_df[
        (golden_df['make'] == 'Studebaker') &
        (pd.to_numeric(golden_df['year'], errors='coerce') >= 1938) &
        (pd.to_numeric(golden_df['year'], errors='coerce') <= 1939)
    ].dropna(subset=['year'])
    
    print(f"\n📊 STUDEBAKER 1938-1939 MODELS IN GOLDEN MASTER:")
    print(f"Total records: {len(studebaker_df)}")
    
    # Get unique models
    unique_models = sorted(studebaker_df['model'].dropna().unique())
    print(f"\nUnique models ({len(unique_models)}):")
    for model in unique_models:
        count = len(studebaker_df[studebaker_df['model'] == model])
        years = sorted(pd.to_numeric(studebaker_df[studebaker_df['model'] == model]['year'], errors='coerce').dropna().astype(int).unique())
        year_range = f"{min(years)}-{max(years)}" if years else "No years"
        print(f"  - {model}: {count} records ({year_range})")
    
    # Check specifically for our extracted models
    target_models = ['6-7A Commander', '6-8A Commander', '8-4C State President', '8-5C State President', '9A Commander']
    print(f"\n🎯 CHECKING EXTRACTED MODELS:")
    for model in target_models:
        matches = studebaker_df[studebaker_df['model'] == model]
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
            partial_matches = []
            for golden_model in unique_models:
                if any(word in golden_model.lower() for word in model.lower().split() if len(word) > 2):
                    partial_matches.append(golden_model)
            if partial_matches:
                print(f"🔍 {model} -> Possible matches: {partial_matches}")
            else:
                print(f"❌ {model} -> No partial matches found")

if __name__ == "__main__":
    debug_studebaker()