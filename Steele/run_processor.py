#!/usr/bin/env python3
"""
Script to run the Steele processor on the first_100_stock_codes_sample.csv file.
"""

import sys
import os
import time
from pathlib import Path

# Add the utils directory to the path
sys.path.append('utils')

from steele_data_transformer import SteeleDataTransformer

def main():
    """Run the processor on the first_100_stock_codes_sample.csv file."""
    
    print("🚀 STEELE PROCESSOR - FIRST 100 STOCK CODES SAMPLE")
    print("=" * 60)
    
    try:
        # Initialize transformer with AI enabled
        print("Initializing Steele Data Transformer...")
        transformer = SteeleDataTransformer(use_ai=True)
        
        # Process the first_100_stock_codes_sample.csv file
        print("Processing first_100_stock_codes_sample.csv...")
        final_df = transformer.process_complete_pipeline_no_ai('data/samples/first_100_stock_codes_sample.csv')
        
        # Create output directory
        output_dir = Path('data/results')
        output_dir.mkdir(exist_ok=True)
        
        # Save results with timestamp
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f'first_100_products_{timestamp}.csv'
        
        final_df.to_csv(output_file, index=False)
        
        print("\n" + "=" * 60)
        print("✅ PROCESSING COMPLETE")
        print(f"📁 Results saved to: {output_file}")
        print(f"📊 Processed {len(final_df)} unique products from the sample")
        print(f"📈 Input records: 1563 (from sample file)")
        print(f"📉 Output products: {len(final_df)} (consolidated)")
        print("=" * 60)
        
        return str(output_file)
        
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 