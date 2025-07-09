#!/usr/bin/env python3
"""
Extract first 100 unique stock codes from processed product data.
"""

import pandas as pd
import re
import os
from pathlib import Path

def extract_stock_codes_from_text(text):
    """Extract SKU and MPN codes from product description text."""
    if pd.isna(text) or not isinstance(text, str):
        return []
    
    codes = []
    
    # Pattern to match SKU: followed by code (more flexible)
    sku_pattern = r'SKU:\s*([A-Za-z0-9\-_\.]+)'
    sku_matches = re.findall(sku_pattern, text)
    codes.extend(sku_matches)
    
    # Pattern to match MPN: followed by code (more flexible)
    mpn_pattern = r'MPN:\s*([A-Za-z0-9\-_\.]+)'
    mpn_matches = re.findall(mpn_pattern, text)
    codes.extend(mpn_matches)
    
    # Also look for patterns like "SKU LV8-xxx" without colon
    sku_pattern2 = r'SKU\s+([A-Za-z0-9\-_\.]+)'
    sku_matches2 = re.findall(sku_pattern2, text)
    codes.extend(sku_matches2)
    
    # Also look for patterns like "MPN xxx" without colon
    mpn_pattern2 = r'MPN\s+([A-Za-z0-9\-_\.]+)'
    mpn_matches2 = re.findall(mpn_pattern2, text)
    codes.extend(mpn_matches2)
    
    return codes

def process_csv_file(file_path):
    """Process a CSV file and extract stock codes."""
    print(f"Processing: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        print(f"Columns available: {list(df.columns)}")
        
        # Look for product description column (prioritize exact match)
        target_column = None
        if 'product_description' in df.columns:
            target_column = 'product_description'
        else:
            # Look for any column with 'description' in the name
            desc_columns = [col for col in df.columns if 'description' in col.lower()]
            if desc_columns:
                target_column = desc_columns[0]
        
        if not target_column:
            print(f"No description column found in {file_path}")
            return []
        
        print(f"Using column: {target_column}")
        
        all_codes = []
        for idx, text in enumerate(df[target_column]):
            codes = extract_stock_codes_from_text(text)
            all_codes.extend(codes)
            
            if idx < 5 and codes:  # Show first few examples
                print(f"Row {idx}: Found codes {codes}")
        
        print(f"Total codes found in this file: {len(all_codes)}")
        return all_codes
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []

def main():
    # Define paths to processed data directories
    processed_dirs = [
        "REM/data/processed",
        "Steele/data/processed", 
        "Ford/data/processed",
        "project/src/ABAP/meta_info/data/processed"
    ]
    
    all_stock_codes = []
    
    # Process each directory
    for dir_path in processed_dirs:
        if os.path.exists(dir_path):
            print(f"\n=== Processing directory: {dir_path} ===")
            
            # Get all CSV files in directory
            csv_files = list(Path(dir_path).glob("*.csv"))
            
            for csv_file in csv_files:
                # Skip very large files (>10MB) to avoid memory issues
                file_size = csv_file.stat().st_size / (1024 * 1024)  # MB
                if file_size > 10:
                    print(f"Skipping large file: {csv_file} ({file_size:.1f}MB)")
                    continue
                
                codes = process_csv_file(csv_file)
                all_stock_codes.extend(codes)
                
                # Stop if we have enough codes
                unique_codes = list(dict.fromkeys(all_stock_codes))  # Preserve order, remove duplicates
                if len(unique_codes) >= 100:
                    print(f"Found {len(unique_codes)} unique codes, stopping early")
                    break
            
            # Stop if we have enough codes
            unique_codes = list(dict.fromkeys(all_stock_codes))
            if len(unique_codes) >= 100:
                break
    
    # Get first 100 unique codes
    unique_codes = list(dict.fromkeys(all_stock_codes))[:100]
    
    print(f"\n=== RESULTS ===")
    print(f"Total codes found: {len(all_stock_codes)}")
    print(f"Unique codes found: {len(set(all_stock_codes))}")
    print(f"First 100 unique codes: {len(unique_codes)}")
    
    # Create sample file
    sample_data = {
        'stock_code': unique_codes,
        'code_type': ['SKU/MPN'] * len(unique_codes),
        'source': ['processed_data'] * len(unique_codes)
    }
    
    sample_df = pd.DataFrame(sample_data)
    
    # Save to samples directory
    output_file = "shared/data/first_100_stock_codes_sample.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    sample_df.to_csv(output_file, index=False)
    
    print(f"\nSample saved to: {output_file}")
    print(f"First 10 codes: {unique_codes[:10]}")
    
    return unique_codes

if __name__ == "__main__":
    codes = main() 