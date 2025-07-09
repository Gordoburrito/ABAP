#!/usr/bin/env python3
"""
Two-step hallucination fix:
1. Fix underscores in known model names (Galaxie_500 -> Galaxie 500)
2. Remove any remaining hallucinated tags by checking against golden master

Based on remove_hallucinated_car_tags.py approach.
"""

import pandas as pd
import re
from datetime import datetime

def load_golden_vehicle_tags():
    """Load all valid vehicle tags from the master golden file."""
    print("Loading golden vehicle tags from master file...")
    
    master_file = '/Users/gordonlewis/ABAP/shared/data/master_ultimate_golden.csv'
    df = pd.read_csv(master_file, low_memory=False)
    
    # Create golden tags in the format: YEAR_MAKE_MODEL
    golden_tags = set()
    
    for _, row in df.iterrows():
        if pd.notna(row.get('Year', '')) and pd.notna(row.get('Make', '')) and pd.notna(row.get('Model', '')):
            # Create tag in same format as our data
            tag = f"{int(row['Year'])}_{row['Make']}_{row['Model']}"
            golden_tags.add(tag)
    
    print(f"Loaded {len(golden_tags)} valid vehicle tags from golden file")
    
    # Show some examples
    examples = list(golden_tags)[:5]
    print(f"Examples: {examples}")
    
    return golden_tags

def fix_underscore_models(tags_string):
    """Step 1: Fix known underscore issues in model names."""
    if pd.isna(tags_string) or tags_string == "":
        return tags_string, []
    
    fixed_tags = tags_string
    fixes_made = []
    
    # Common model fixes based on golden file
    model_fixes = {
        'Galaxie_500': 'Galaxie 500',
        'Country_Squire': 'Country Squire',
        'Ranch_Wagon': 'Ranch Wagon',
        'Country_Sedan': 'Country Sedan',
        'Club_Wagon': 'Club Wagon',
        'Colony_Park': 'Colony Park',
        'Custom_300': 'Custom 300',
        'Custom_500': 'Custom 500',
        'Del_Rio_Wagon': 'Del Rio Wagon',
        'Courier_Sedan_Delivery': 'Courier Sedan Delivery',
        'Falcon_Sedan_Delivery': 'Falcon Sedan Delivery',
        'Turnpike_Cruiser': 'Turnpike Cruiser'
    }
    
    for bad_model, correct_model in model_fixes.items():
        # Fix pattern: YEAR_MAKE_BAD_MODEL -> YEAR_MAKE_CORRECT_MODEL
        pattern = r'(\d{4}_(?:Ford|Mercury|Edsel)_)' + re.escape(bad_model)
        
        def replace_func(match):
            return match.group(1) + correct_model
        
        before_count = fixed_tags.count(bad_model)
        fixed_tags = re.sub(pattern, replace_func, fixed_tags)
        after_count = fixed_tags.count(bad_model)
        
        if before_count > after_count:
            fixes_made.append(f"{bad_model} -> {correct_model}")
    
    return fixed_tags, fixes_made

def remove_hallucinated_tags(tags_string, golden_tags):
    """Step 2: Remove tags that don't exist in the golden master."""
    if pd.isna(tags_string) or tags_string == "":
        return tags_string, [], []
    
    # Split tags
    test_tags = [tag.strip() for tag in tags_string.split(',') if tag.strip()]
    
    # Find real vs hallucinated tags
    real_tags = []
    hallucinated_tags = []
    
    for tag in test_tags:
        if tag in golden_tags:
            real_tags.append(tag)
        else:
            hallucinated_tags.append(tag)
    
    # Return cleaned tags
    cleaned_tags = ', '.join(real_tags)
    
    return cleaned_tags, real_tags, hallucinated_tags

def process_small_test():
    """Process a small test file with both steps."""
    print("=== Two-Step Hallucination Fix ===")
    
    # Load golden tags
    golden_tags = load_golden_vehicle_tags()
    
    # Create small test file (20 products)
    print("\nCreating small test file...")
    input_file = 'data/results/corrected_shopify_format_20250705_170607.csv'
    test_file = 'data/results/small_test_20_products.csv'
    
    # Create 20-product sample
    full_df = pd.read_csv(input_file, nrows=21)  # +1 for header
    full_df.to_csv(test_file, index=False)
    print(f"Created test file with {len(full_df)} products")
    
    # Load test file
    df = pd.read_csv(test_file)
    
    # Stats before processing
    print(f"\n=== Before Processing ===")
    original_tags_count = sum(len([t.strip() for t in str(row['Tags']).split(',') if t.strip()]) for _, row in df.iterrows())
    print(f"Total tags before: {original_tags_count}")
    
    # Step 1: Fix underscores
    print(f"\n=== Step 1: Fixing Underscores ===")
    total_underscore_fixes = 0
    
    for i in range(len(df)):
        original_tags = df.iloc[i]['Tags']
        fixed_tags, fixes = fix_underscore_models(original_tags)
        
        if fixes:
            df.iloc[i, df.columns.get_loc('Tags')] = fixed_tags
            total_underscore_fixes += len(fixes)
            print(f"  Product {i}: {fixes}")
    
    print(f"Total underscore fixes: {total_underscore_fixes}")
    
    # Step 2: Remove hallucinations
    print(f"\n=== Step 2: Removing Hallucinations ===")
    total_hallucinations = 0
    total_real_tags = 0
    
    for i in range(len(df)):
        original_tags = df.iloc[i]['Tags']
        cleaned_tags, real_tags, hallucinated_tags = remove_hallucinated_tags(original_tags, golden_tags)
        
        if hallucinated_tags:
            df.iloc[i, df.columns.get_loc('Tags')] = cleaned_tags
            total_hallucinations += len(hallucinated_tags)
            total_real_tags += len(real_tags)
            
            if len(hallucinated_tags) > 0:
                print(f"  Product {i}: {len(hallucinated_tags)} hallucinations, {len(real_tags)} real tags")
                if len(hallucinated_tags) <= 5:  # Show examples if not too many
                    print(f"    Hallucinated: {hallucinated_tags[:5]}")
    
    # Final stats
    print(f"\n=== Final Results ===")
    final_tags_count = sum(len([t.strip() for t in str(row['Tags']).split(',') if t.strip()]) for _, row in df.iterrows())
    print(f"Original tags: {original_tags_count}")
    print(f"Final tags: {final_tags_count}")
    print(f"Tags removed: {original_tags_count - final_tags_count}")
    print(f"Underscore fixes: {total_underscore_fixes}")
    print(f"Hallucinations removed: {total_hallucinations}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'data/results/cleaned_test_{timestamp}.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\n✅ Cleaned test file saved to: {output_file}")
    
    # Show examples
    print(f"\n=== Example Results ===")
    for i, row in df.head(3).iterrows():
        tags = str(row['Tags'])
        tag_count = len([t.strip() for t in tags.split(',') if t.strip()])
        print(f"Product {i}: {tag_count} clean tags")
        if 'Galaxie 500' in tags:
            print(f"  ✅ Contains fixed: Galaxie 500")
    
    return output_file

if __name__ == "__main__":
    process_small_test() 