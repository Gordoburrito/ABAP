#!/usr/bin/env python3
"""
Fix specific Galaxie_500 hallucination to 'Galaxie 500' based on master golden file validation.

The master golden file shows that 'Galaxie 500' (with space) is the correct model name.
All instances of 'Galaxie_500' (with underscore) should be fixed to 'Galaxie 500'.
"""

import pandas as pd
import re
from datetime import datetime

def fix_galaxie_500_tags(tags_string):
    """Fix Galaxie_500 tags to Galaxie 500 (space instead of underscore)."""
    if pd.isna(tags_string) or tags_string == "":
        return tags_string
    
    # Fix the specific pattern: X_Y_Galaxie_500 -> X_Y_Galaxie 500
    # This preserves the format like 1962_Ford_Galaxie 500
    fixed_tags = re.sub(r'(\d{4}_\w+_)Galaxie_500', r'\1Galaxie 500', tags_string)
    
    return fixed_tags

def main():
    """Fix the Galaxie_500 hallucination in the test file."""
    print("=== Fixing Galaxie_500 -> Galaxie 500 ===")
    
    # Load the test file
    input_file = 'data/results/galaxie_test_sample.csv'
    print(f"Loading: {input_file}")
    
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} products")
    
    # Count before fixing
    before_count = sum(1 for tags in df['Tags'] if 'Galaxie_500' in str(tags))
    print(f"Products with 'Galaxie_500' tags: {before_count}")
    
    # Apply the fix
    df['Tags'] = df['Tags'].apply(fix_galaxie_500_tags)
    
    # Count after fixing
    after_count = sum(1 for tags in df['Tags'] if 'Galaxie 500' in str(tags))
    remaining_bad = sum(1 for tags in df['Tags'] if 'Galaxie_500' in str(tags))
    
    print(f"Products with 'Galaxie 500' tags after fix: {after_count}")
    print(f"Remaining 'Galaxie_500' tags: {remaining_bad}")
    
    # Save the fixed file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'data/results/galaxie_fixed_{timestamp}.csv'
    df.to_csv(output_file, index=False)
    
    print(f"Fixed file saved to: {output_file}")
    
    # Show some example fixes
    print("\n=== Example fixes ===")
    for i, row in df.head(3).iterrows():
        tags = str(row['Tags'])
        if 'Galaxie 500' in tags:
            # Find and show the fixed part
            matches = re.findall(r'\d{4}_\w+_Galaxie 500', tags)
            if matches:
                print(f"Product {i}: {matches[0]}")
    
    return output_file

if __name__ == "__main__":
    main() 