#!/usr/bin/env python3
"""
Extract first 100 unique stock codes from Steele processed data.
"""

import pandas as pd
import os

def main():
    steele_file = "Steele/data/processed/steele_processed_complete.csv"
    
    if not os.path.exists(steele_file):
        print(f"Steele file not found: {steele_file}")
        return []
    
    print(f"=== Processing Steele data ===")
    
    all_codes = []
    
    try:
        # Read file in chunks to handle large size
        chunk_size = 5000
        chunk_iter = pd.read_csv(steele_file, chunksize=chunk_size)
        
        for chunk_num, chunk in enumerate(chunk_iter):
            print(f"Processing chunk {chunk_num + 1} (rows {chunk_num * chunk_size + 1}-{(chunk_num + 1) * chunk_size})")
            
            if chunk_num == 0:
                print(f"Columns available: {list(chunk.columns)}")
            
            # Extract stock codes from StockCode column
            if 'StockCode' in chunk.columns:
                stock_codes = chunk['StockCode'].dropna().astype(str).tolist()
                stock_codes = [code.strip() for code in stock_codes if code and code != 'nan' and code.strip()]
                all_codes.extend(stock_codes)
                if chunk_num == 0:
                    print(f"Sample StockCodes: {stock_codes[:5]}")
            
            # Extract part numbers from PartNumber column
            if 'PartNumber' in chunk.columns:
                part_numbers = chunk['PartNumber'].dropna().astype(str).tolist()
                part_numbers = [code.strip() for code in part_numbers if code and code != 'nan' and code.strip()]
                all_codes.extend(part_numbers)
                if chunk_num == 0:
                    print(f"Sample PartNumbers: {part_numbers[:5]}")
            
            # Get unique codes so far
            unique_codes = list(dict.fromkeys(all_codes))
            print(f"Total unique codes so far: {len(unique_codes)}")
            
            # Stop if we have enough codes
            if len(unique_codes) >= 100:
                print(f"Reached target of 100 unique codes, stopping")
                break
        
        # Get first 100 unique codes
        unique_codes = list(dict.fromkeys(all_codes))[:100]
        
        print(f"\n=== RESULTS ===")
        print(f"Total codes found: {len(all_codes)}")
        print(f"Unique codes found: {len(set(all_codes))}")
        print(f"First 100 unique codes: {len(unique_codes)}")
        
        # Create sample file
        sample_data = {
            'stock_code': unique_codes,
            'code_type': ['StockCode/PartNumber'] * len(unique_codes),
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
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return []

if __name__ == "__main__":
    codes = main() 