#!/usr/bin/env python3
"""
Fix all model name hallucinations in the full Shopify dataset by referencing master_ultimate_golden.csv

This script:
1. Loads correct model names from the master golden file
2. Identifies common hallucination patterns (underscores instead of spaces)
3. Applies fixes to the full dataset
4. Validates results and reports statistics
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
        return tags_string
    
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
    """Fix all model hallucinations in the full dataset."""
    print("=== Comprehensive Model Hallucination Fix ===")
    
    # Load correct models from golden file
    space_models = load_golden_models_full()
    
    # Create fix mappings
    model_fixes = create_comprehensive_fixes(space_models)
    
    # Load the full Shopify dataset
    input_file = 'data/results/corrected_shopify_format_20250705_170607.csv'
    print(f"\nLoading full dataset: {input_file}")
    
    # Load in chunks to handle large file
    chunk_size = 1000
    chunks = []
    total_rows = 0
    
    for chunk in pd.read_csv(input_file, chunksize=chunk_size):
        total_rows += len(chunk)
        chunks.append(chunk)
        if total_rows % 5000 == 0:
            print(f"  Loaded {total_rows} rows...")
    
    print(f"Loaded {total_rows} total products")
    
    # Process chunks
    print("\n=== Processing chunks ===")
    fixed_chunks = []
    total_changes = 0
    
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}...")
        
        # Apply fixes to this chunk
        chunk_changes = 0
        for j in range(len(chunk)):
            original_tags = chunk.iloc[j]['Tags']
            fixed_tags, changes = fix_model_hallucinations(original_tags, model_fixes)
            
            if changes:
                chunk.iloc[j, chunk.columns.get_loc('Tags')] = fixed_tags
                chunk_changes += len(changes)
        
        fixed_chunks.append(chunk)
        total_changes += chunk_changes
        
        if chunk_changes > 0:
            print(f"  Made {chunk_changes} fixes in chunk {i+1}")
    
    # Combine fixed chunks
    print(f"\n=== Results ===")
    print(f"Total model name fixes applied: {total_changes}")
    
    fixed_df = pd.concat(fixed_chunks, ignore_index=True)
    
    # Save the fixed dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'data/results/fixed_all_hallucinations_{timestamp}.csv'
    
    print(f"Saving fixed dataset to: {output_file}")
    fixed_df.to_csv(output_file, index=False)
    
    # Validation
    print(f"\n=== Validation ===")
    
    # Check for specific patterns
    galaxie_500_count = sum(1 for tags in fixed_df['Tags'] if 'Galaxie 500' in str(tags))
    galaxie_500_bad = sum(1 for tags in fixed_df['Tags'] if 'Galaxie_500' in str(tags))
    
    print(f"'Galaxie 500' (correct) tags: {galaxie_500_count}")
    print(f"'Galaxie_500' (incorrect) tags remaining: {galaxie_500_bad}")
    
    # Show file size
    import os
    if os.path.exists(output_file):
        size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"Output file size: {size_mb:.1f} MB")
    
    print(f"\n✅ Fixed dataset saved to: {output_file}")
    return output_file

if __name__ == "__main__":
    main() 