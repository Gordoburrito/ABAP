#!/usr/bin/env python3
"""
Test script to fix AI hallucinations by comparing against master_ultimate_golden.csv

This script:
1. Loads the master golden file to get correct model names
2. Tests fixing logic on a small sample file
3. Shows before/after comparisons for validation
"""

import pandas as pd
import re
from datetime import datetime

def load_golden_models():
    """Load correct model names from the master golden file."""
    print("Loading master golden file...")
    
    # Load the master golden file (first few thousand rows for testing)
    master_file = '/Users/gordonlewis/ABAP/shared/data/master_ultimate_golden.csv'
    
    # Load just enough to get the model patterns - using nrows to limit for testing
    df = pd.read_csv(master_file, nrows=10000, low_memory=False)
    
    # Create a set of correct model names
    correct_models = set()
    
    for _, row in df.iterrows():
        if pd.notna(row.get('Model', '')):
            correct_models.add(row['Model'])
    
    print(f"Loaded {len(correct_models)} unique model names from golden file")
    
    # Show some examples of models with spaces
    space_models = [m for m in correct_models if ' ' in m][:10]
    print(f"Examples of models with spaces: {space_models}")
    
    return correct_models

def create_hallucination_fixes(correct_models):
    """Create a mapping of hallucinated tags to correct ones."""
    fixes = {}
    
    # Check if we have "Galaxie 500" in the golden file
    if "Galaxie 500" in correct_models:
        print("✅ Found 'Galaxie 500' as correct model name in golden file")
    else:
        print("❌ 'Galaxie 500' not found in golden file sample")
        # Let's see what Galaxie models we do have
        galaxie_models = [m for m in correct_models if 'Galaxie' in m]
        print(f"Galaxie models found: {galaxie_models}")
    
    # Create fix patterns based on golden file
    for model in correct_models:
        if ' ' in model:
            # Create underscore version that might be hallucinated
            underscore_version = model.replace(' ', '_')
            fixes[underscore_version] = model
    
    print(f"Created {len(fixes)} hallucination fix patterns")
    return fixes

def fix_vehicle_tags(tags_string, hallucination_fixes):
    """Fix vehicle tags using the hallucination mapping."""
    if pd.isna(tags_string) or tags_string == "":
        return tags_string
    
    fixed_tags = tags_string
    
    # Apply fixes for each hallucination pattern
    for bad_pattern, correct_pattern in hallucination_fixes.items():
        # Fix for Ford, Mercury, Edsel, etc.
        for make in ['Ford', 'Mercury', 'Edsel']:
            bad_tag_pattern = f'_{make}_{bad_pattern}'
            correct_tag_pattern = f'_{make}_{correct_pattern}'
            
            if bad_tag_pattern in fixed_tags:
                fixed_tags = fixed_tags.replace(bad_tag_pattern, correct_tag_pattern)
                print(f"  Fixed: {bad_tag_pattern} -> {correct_tag_pattern}")
    
    return fixed_tags

def main():
    """Test the hallucination fixing on a small sample."""
    print("=== Testing AI Hallucination Fixes ===")
    
    # Load the golden models
    correct_models = load_golden_models()
    
    # Create fix mappings
    hallucination_fixes = create_hallucination_fixes(correct_models)
    
    # Load the test sample
    test_file = 'data/results/galaxie_test_sample.csv'
    print(f"\nLoading test file: {test_file}")
    
    df = pd.read_csv(test_file)
    print(f"Loaded {len(df)} test products")
    
    # Show examples before fixing
    print("\n=== BEFORE Fixing ===")
    for i, tags in enumerate(df['Tags'][:3]):
        if 'Galaxie_500' in str(tags):
            print(f"Product {i}: ...{str(tags)[1000:1200]}...")
    
    # Apply fixes
    print("\n=== Applying Fixes ===")
    df['Tags_Fixed'] = df['Tags'].apply(lambda x: fix_vehicle_tags(x, hallucination_fixes))
    
    # Show examples after fixing
    print("\n=== AFTER Fixing ===")
    for i, tags in enumerate(df['Tags_Fixed'][:3]):
        if 'Galaxie 500' in str(tags):
            print(f"Product {i}: ...{str(tags)[1000:1200]}...")
    
    # Count changes
    changes_made = 0
    for i in range(len(df)):
        if str(df['Tags'].iloc[i]) != str(df['Tags_Fixed'].iloc[i]):
            changes_made += 1
    
    print(f"\n=== Results ===")
    print(f"Products with changes: {changes_made}/{len(df)}")
    print(f"'Galaxie_500' tags before: {sum(1 for tags in df['Tags'] if 'Galaxie_500' in str(tags))}")
    print(f"'Galaxie 500' tags after: {sum(1 for tags in df['Tags_Fixed'] if 'Galaxie 500' in str(tags))}")
    
    # Save test results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'data/results/test_fixed_{timestamp}.csv'
    df.to_csv(output_file, index=False)
    print(f"Test results saved to: {output_file}")
    
    return output_file

if __name__ == "__main__":
    main() 