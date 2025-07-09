#!/usr/bin/env python3
"""
Fix ONLY the specific Galaxie_500 hallucination to 'Galaxie 500'.

This script:
- Only fixes Galaxie_500 -> Galaxie 500 
- Does NOT touch manufacturer names
- Very fast and targeted approach
"""

import pandas as pd
import re
from datetime import datetime

def fix_galaxie_only(tags_string):
    """Fix only Galaxie_500 tags to Galaxie 500 (space instead of underscore)."""
    if pd.isna(tags_string) or tags_string == "":
        return tags_string, 0
    
    # Count before
    before_count = tags_string.count('Galaxie_500')
    
    # Fix the specific pattern: YEAR_MAKE_Galaxie_500 -> YEAR_MAKE_Galaxie 500
    # This preserves everything else and only fixes the model name
    fixed_tags = re.sub(r'(\d{4}_(?:Ford|Mercury|Edsel)_)Galaxie_500', r'\1Galaxie 500', tags_string)
    
    # Count after
    after_count = fixed_tags.count('Galaxie_500')
    
    changes_made = before_count - after_count
    
    return fixed_tags, changes_made

def main():
    """Fix only Galaxie_500 hallucinations in the test file."""
    print("=== Fixing ONLY Galaxie_500 -> Galaxie 500 ===")
    
    # Load the test file first
    test_file = 'data/results/test_1000_products.csv'
    print(f"Loading test dataset: {test_file}")
    
    df = pd.read_csv(test_file)
    print(f"Loaded {len(df)} test products")
    
    # Count before fixing
    before_count = sum(1 for tags in df['Tags'] if 'Galaxie_500' in str(tags))
    print(f"Products with 'Galaxie_500' tags: {before_count}")
    
    # Apply the fix
    print("Applying Galaxie_500 -> Galaxie 500 fix...")
    total_fixes = 0
    
    for i in range(len(df)):
        original_tags = df.iloc[i]['Tags']
        fixed_tags, changes = fix_galaxie_only(original_tags)
        
        if changes > 0:
            df.iloc[i, df.columns.get_loc('Tags')] = fixed_tags
            total_fixes += changes
    
    # Count after fixing
    after_count_good = sum(1 for tags in df['Tags'] if 'Galaxie 500' in str(tags))
    after_count_bad = sum(1 for tags in df['Tags'] if 'Galaxie_500' in str(tags))
    
    print(f"\n=== Results ===")
    print(f"Total Galaxie_500 instances fixed: {total_fixes}")
    print(f"Products with 'Galaxie 500' (correct): {after_count_good}")
    print(f"Remaining 'Galaxie_500' (incorrect): {after_count_bad}")
    
    # Save the fixed file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'data/results/galaxie_only_fixed_{timestamp}.csv'
    df.to_csv(output_file, index=False)
    
    print(f"Fixed file saved to: {output_file}")
    
    # Show example fixes
    print("\n=== Example fixes ===")
    for i, row in df.head(5).iterrows():
        tags = str(row['Tags'])
        if 'Galaxie 500' in tags:
            # Find and show the fixed parts
            matches = re.findall(r'\d{4}_(?:Ford|Mercury|Edsel)_Galaxie 500', tags)
            if matches:
                print(f"Product {i}: {matches[0]}")
    
    return output_file

if __name__ == "__main__":
    main() 