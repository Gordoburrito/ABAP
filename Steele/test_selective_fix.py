#!/usr/bin/env python3
"""
Selective test to fix only specific model hallucinations, not manufacturer names.

This script ONLY fixes model names that should have spaces, like:
- Galaxie_500 -> Galaxie 500
- Country_Squire -> Country Squire  
- Ranch_Wagon -> Ranch Wagon

It DOES NOT fix manufacturer names or other patterns.
"""

import pandas as pd
import re
from datetime import datetime

def load_specific_model_fixes():
    """Load only specific model names that we know need fixing from the golden file."""
    print("Loading specific model fixes from master golden file...")
    
    master_file = '/Users/gordonlewis/ABAP/shared/data/master_ultimate_golden.csv'
    
    # Load the full file
    df = pd.read_csv(master_file, low_memory=False)
    
    # Create a set of correct model names
    correct_models = set()
    
    for _, row in df.iterrows():
        if pd.notna(row.get('Model', '')):
            correct_models.add(row['Model'])
    
    print(f"Loaded {len(correct_models)} unique model names from golden file")
    
    # Only select models that have spaces and are commonly hallucinated
    # Focus on automotive models, not manufacturer names
    target_models = []
    
    for model in correct_models:
        if ' ' in model and any(keyword in model for keyword in [
            'Galaxie', 'Country', 'Ranch', 'Custom', 'Station', 'Falcon', 
            'Club', 'Colony', 'Del Rio', 'Courier'
        ]):
            target_models.append(model)
    
    print(f"Selected {len(target_models)} automotive models for fixing")
    
    # Show examples
    print("Examples of models to fix:")
    for model in sorted(target_models)[:10]:
        underscore_version = model.replace(' ', '_')
        print(f"  {underscore_version} -> {model}")
    
    return target_models

def create_selective_fixes(target_models):
    """Create fix mappings only for specific automotive models."""
    fixes = {}
    
    for model in target_models:
        # Create underscore version that might be hallucinated
        underscore_version = model.replace(' ', '_')
        fixes[underscore_version] = model
    
    print(f"\nCreated {len(fixes)} selective fix patterns")
    return fixes

def fix_selective_hallucinations(tags_string, model_fixes):
    """Apply only selective model fixes to vehicle tags."""
    if pd.isna(tags_string) or tags_string == "":
        return tags_string, []
    
    fixed_tags = tags_string
    changes_made = []
    
    # Apply each fix pattern
    for bad_model, correct_model in model_fixes.items():
        # Create pattern to match: YEAR_MAKE_BAD_MODEL -> YEAR_MAKE_CORRECT_MODEL
        # This preserves the make (Ford, Mercury, etc.) and only fixes the model
        pattern = r'(\d{4}_(?:Ford|Mercury|Edsel)_)' + re.escape(bad_model)
        
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
    """Test selective model fixes on 1000 products."""
    print("=== Testing Selective Model Hallucination Fix ===")
    
    # Load specific automotive models from golden file
    target_models = load_specific_model_fixes()
    
    # Create fix mappings
    model_fixes = create_selective_fixes(target_models)
    
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
    print("\n=== Processing selective fixes ===")
    total_changes = 0
    
    for i in range(len(df)):
        original_tags = df.iloc[i]['Tags']
        fixed_tags, changes = fix_selective_hallucinations(original_tags, model_fixes)
        
        if changes:
            df.iloc[i, df.columns.get_loc('Tags')] = fixed_tags
            total_changes += len(changes)
            
            if i < 5:  # Show first few fixes
                print(f"  Product {i}: {len(changes)} fixes - {changes[:2]}")
    
    print(f"Total selective fixes applied: {total_changes}")
    
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
    output_file = f'data/results/test_selective_fixed_{timestamp}.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\n✅ Selective test results saved to: {output_file}")
    
    # Show some example fixes
    print("\n=== Example selective fixes ===")
    for i, row in df.head(3).iterrows():
        tags = str(row['Tags'])
        # Look for our specific fixes
        for pattern in ['Galaxie 500', 'Country Squire', 'Ranch Wagon']:
            if pattern in tags:
                matches = re.findall(r'\d{4}_(?:Ford|Mercury|Edsel)_' + pattern.replace(' ', r'\s'), tags)
                if matches:
                    print(f"Product {i}: {matches[0]}")
                    break
    
    return output_file

if __name__ == "__main__":
    main() 