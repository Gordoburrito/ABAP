#!/usr/bin/env python3
"""
Test the comprehensive model hallucination fix on a 1000-product sample first.
"""

import pandas as pd
import re
from datetime import datetime

def load_golden_models_full():
    """Load all correct model names from the master golden file."""
    print("Loading master golden file...")
    
    master_file = '/Users/gordonlewis/ABAP/shared/data/master_ultimate_golden.csv'
    
    # Load the full file - this might take a moment
    df = pd.read_csv(master_file, low_memory=False)
    
    # Create a set of correct model names
    correct_models = set()
    
    for _, row in df.iterrows():
        if pd.notna(row.get('Model', '')):
            correct_models.add(row['Model'])
    
    print(f"Loaded {len(correct_models)} unique model names from golden file")
    
    # Find models with spaces that might be hallucinated with underscores
    space_models = [m for m in correct_models if ' ' in m]
    print(f"Found {len(space_models)} models with spaces (potential hallucination targets)")
    
    return space_models

def create_comprehensive_fixes(space_models):
    """Create comprehensive fix mappings for all space-containing models."""
    fixes = {}
    
    for model in space_models:
        # Create underscore version that might be hallucinated
        underscore_version = model.replace(' ', '_')
        fixes[underscore_version] = model
    
    # Show some examples
    print(f"\nCreated {len(fixes)} fix patterns. Examples:")
    for i, (bad, good) in enumerate(list(fixes.items())[:5]):
        print(f"  {bad} -> {good}")
    
    return fixes

def fix_model_hallucinations(tags_string, model_fixes):
    """Apply all model hallucination fixes to a tags string."""
    if pd.isna(tags_string) or tags_string == "":
        return tags_string, []
    
    fixed_tags = tags_string
    changes_made = []
    
    # Apply each fix pattern
    for bad_model, correct_model in model_fixes.items():
        # Create pattern to match: YEAR_MAKE_BAD_MODEL -> YEAR_MAKE_CORRECT_MODEL
        pattern = r'(\d{4}_\w+_)' + re.escape(bad_model)
        
        # Apply the fix using a lambda to avoid regex group issues
        def replace_func(match):
            return match.group(1) + correct_model
        
        # Apply the fix
        before_count = fixed_tags.count(bad_model)
        fixed_tags = re.sub(pattern, replace_func, fixed_tags)
        after_count = fixed_tags.count(bad_model)
        
        if before_count > after_count:
            changes_made.append(f"{bad_model} -> {correct_model}")
    
    return fixed_tags, changes_made

def main():
    """Test the comprehensive fix on 1000 products."""
    print("=== Testing Comprehensive Model Hallucination Fix ===")
    
    # Load correct models from golden file
    space_models = load_golden_models_full()
    
    # Create fix mappings
    model_fixes = create_comprehensive_fixes(space_models)
    
    # Load the test dataset
    input_file = 'data/results/test_1000_products.csv'
    print(f"\nLoading test dataset: {input_file}")
    
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} test products")
    
    # Count issues before fixing
    print("\n=== Before fixing ===")
    galaxie_500_bad = sum(1 for tags in df['Tags'] if 'Galaxie_500' in str(tags))
    country_squire_bad = sum(1 for tags in df['Tags'] if 'Country_Squire' in str(tags))
    ranch_wagon_bad = sum(1 for tags in df['Tags'] if 'Ranch_Wagon' in str(tags))
    
    print(f"'Galaxie_500' (bad) tags: {galaxie_500_bad}")
    print(f"'Country_Squire' (bad) tags: {country_squire_bad}")
    print(f"'Ranch_Wagon' (bad) tags: {ranch_wagon_bad}")
    
    # Apply fixes
    print("\n=== Processing fixes ===")
    total_changes = 0
    
    for i in range(len(df)):
        original_tags = df.iloc[i]['Tags']
        fixed_tags, changes = fix_model_hallucinations(original_tags, model_fixes)
        
        if changes:
            df.iloc[i, df.columns.get_loc('Tags')] = fixed_tags
            total_changes += len(changes)
            
            if i < 5:  # Show first few fixes
                print(f"  Product {i}: {len(changes)} fixes")
    
    print(f"Total fixes applied: {total_changes}")
    
    # Count after fixing
    print("\n=== After fixing ===")
    galaxie_500_good = sum(1 for tags in df['Tags'] if 'Galaxie 500' in str(tags))
    galaxie_500_bad = sum(1 for tags in df['Tags'] if 'Galaxie_500' in str(tags))
    country_squire_good = sum(1 for tags in df['Tags'] if 'Country Squire' in str(tags))
    country_squire_bad = sum(1 for tags in df['Tags'] if 'Country_Squire' in str(tags))
    ranch_wagon_good = sum(1 for tags in df['Tags'] if 'Ranch Wagon' in str(tags))
    ranch_wagon_bad = sum(1 for tags in df['Tags'] if 'Ranch_Wagon' in str(tags))
    
    print(f"'Galaxie 500' (good) tags: {galaxie_500_good}")
    print(f"'Galaxie_500' (bad) tags remaining: {galaxie_500_bad}")
    print(f"'Country Squire' (good) tags: {country_squire_good}")
    print(f"'Country_Squire' (bad) tags remaining: {country_squire_bad}")
    print(f"'Ranch Wagon' (good) tags: {ranch_wagon_good}")
    print(f"'Ranch_Wagon' (bad) tags remaining: {ranch_wagon_bad}")
    
    # Save test results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'data/results/test_comprehensive_fixed_{timestamp}.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\n✅ Test results saved to: {output_file}")
    
    # Show some example fixes
    print("\n=== Example fixes ===")
    for i, row in df.head(3).iterrows():
        tags = str(row['Tags'])
        if 'Galaxie 500' in tags or 'Country Squire' in tags or 'Ranch Wagon' in tags:
            # Find and show fixed parts
            matches = re.findall(r'\d{4}_\w+_(Galaxie 500|Country Squire|Ranch Wagon)', tags)
            if matches:
                print(f"Product {i}: {matches[0]}")
    
    return output_file

if __name__ == "__main__":
    main() 