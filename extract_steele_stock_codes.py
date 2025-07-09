#!/usr/bin/env python3
"""
Extract first 100 unique stock codes from Steele processed data.
"""

import pandas as pd
import re
import os
from pathlib import Path

def extract_stock_codes_from_text(text):
    """Extract SKU and MPN codes from text."""
    if pd.isna(text) or not isinstance(text, str):
        return []
    
    codes = []
    
    # Pattern to match SKU: followed by code
    sku_pattern = r'SKU:\s*([A-Za-z0-9\-_\.]+)'
    sku_matches = re.findall(sku_pattern, text)
    codes.extend(sku_matches)
    
    # Pattern to match MPN: followed by code
    mpn_pattern = r'MPN:\s*([A-Za-z0-9\-_\.]+)'
    mpn_matches = re.findall(mpn_pattern, text)
    codes.extend(mpn_matches)
    
    return codes

def process_steele_csv_file(file_path, max_codes=100):
    """Process Steele CSV file and extract stock codes efficiently."""
    print(f"Processing Steele file: {file_path}")
    
    all_codes = []
    
    try:
        # Read file in chunks to handle large size
        chunk_size = 1000
        chunk_iter = pd.read_csv(file_path, chunksize=chunk_size)
        
        for chunk_num, chunk in enumerate(chunk_iter):
            print(f"Processing chunk {chunk_num + 1} (rows {chunk_num * chunk_size + 1}-{(chunk_num + 1) * chunk_size})")
            
            if chunk_num == 0:
                print(f"Columns available: {list(chunk.columns)}")
            
            # Look for direct SKU and MPN columns first
            sku_codes = []
            mpn_codes = []
            
            if 'SKU' in chunk.columns:
                sku_values = chunk['SKU'].dropna().astype(str).tolist()
                sku_codes.extend([code for code in sku_values if code and code != 'nan'])
                print(f"Found {len(sku_codes)} SKU codes in this chunk")
            
            if 'MPN' in chunk.columns:
                mpn_values = chunk['MPN'].dropna().astype(str).tolist()
                mpn_codes.extend([code for code in mpn_values if code and code != 'nan'])
                print(f"Found {len(mpn_codes)} MPN codes in this chunk")
            
            # Also search in description columns
            desc_codes = []
            desc_columns = [col for col in chunk.columns if 'description' in col.lower()]
            
            for desc_col in desc_columns:
                for text in chunk[desc_col]:
                    codes = extract_stock_codes_from_text(text)
                    desc_codes.extend(codes)
            
            if desc_codes:
                print(f"Found {len(desc_codes)} codes from description columns in this chunk")
            
            # Combine all codes from this chunk
            chunk_codes = sku_codes + mpn_codes + desc_codes
            all_codes.extend(chunk_codes)
            
            # Get unique codes so far
            unique_codes = list(dict.fromkeys(all_codes))
            print(f"Total unique codes so far: {len(unique_codes)}")
            
            # Stop if we have enough codes
            if len(unique_codes) >= max_codes:
                print(f"Reached target of {max_codes} unique codes, stopping")
                break
                
            # Show sample codes from first few chunks
            if chunk_num < 3 and len(unique_codes) > 0:
                print(f"Sample codes: {unique_codes[:10]}")
        
        return all_codes
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []

def main():
    steele_file = "Steele/data/processed/steele_processed_complete.csv"
    
    if not os.path.exists(steele_file):
        print(f"Steele file not found: {steele_file}")
        return []
    
    print(f"=== Processing Steele data ===")
    
    # Extract stock codes from Steele data
    all_stock_codes = process_steele_csv_file(steele_file, max_codes=100)
    
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
        'source': ['steele_processed_data'] * len(unique_codes)
    }
    
    sample_df = pd.DataFrame(sample_data)
    
    # Save to Steele samples directory
    output_file = "Steele/data/samples/first_100_stock_codes_sample.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    sample_df.to_csv(output_file, index=False)
    
    print(f"\nSample saved to: {output_file}")
    print(f"First 10 codes: {unique_codes[:10]}")
    
    return unique_codes

if __name__ == "__main__":
    codes = main() 