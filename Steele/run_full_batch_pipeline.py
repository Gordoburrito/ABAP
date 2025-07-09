#!/usr/bin/env python3

"""
Full Batch Pipeline for Steele Data Processing
Processes ALL unique products from the Steele dataset in manageable batches
"""

import pandas as pd
import os
import sys
from datetime import datetime
from pathlib import Path
import argparse
import time

# Add utils to path
sys.path.append(str(Path(__file__).parent))

from utils.ai_fitment_extractor import SteeleAIFitmentExtractor

class SteeleFullBatchProcessor:
    """Process all Steele products in manageable batches."""
    
    def __init__(self, batch_size=100):
        self.steele_root = Path(__file__).parent
        self.data_dir = self.steele_root / "data"
        self.processed_dir = self.data_dir / "processed"
        self.results_dir = self.data_dir / "results"
        self.batch_size = batch_size
        self.extractor = None
        
        # Ensure directories exist
        self.results_dir.mkdir(exist_ok=True)
        
    def initialize_ai_extractor(self):
        """Initialize the AI fitment extractor."""
        try:
            print("🤖 Initializing AI Fitment Extractor...")
            self.extractor = SteeleAIFitmentExtractor()
            print("✅ AI Extractor initialized successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize AI extractor: {e}")
            return False
    
    def load_unique_products(self):
        """Load unique products from the complete dataset."""
        steele_file = self.processed_dir / "steele_processed_complete.csv"
        
        if not steele_file.exists():
            print(f"❌ File not found: {steele_file}")
            return None
            
        print(f"📖 Loading complete Steele dataset...")
        print(f"📁 File: {steele_file}")
        
        # Load with low_memory=False to handle mixed types
        df = pd.read_csv(steele_file, low_memory=False)
        print(f"📊 Total rows loaded: {len(df):,}")
        
        # Get unique products (one per StockCode)
        print("🔍 Extracting unique products...")
        unique_products = df.groupby('StockCode').first().reset_index()
        print(f"📊 Unique products found: {len(unique_products):,}")
        
        # Prepare for AI processing
        input_df = pd.DataFrame({
            'StockCode': unique_products['StockCode'],
            'ProductName': unique_products['Product Name'],
            'Description': unique_products['Description']
        })
        
        # Remove any rows with missing data
        initial_count = len(input_df)
        input_df = input_df.dropna(subset=['StockCode', 'ProductName', 'Description'])
        final_count = len(input_df)
        
        if initial_count != final_count:
            print(f"⚠️  Removed {initial_count - final_count} products with missing data")
        
        print(f"✅ Ready to process {final_count:,} unique products")
        return input_df
    
    def process_in_batches(self, input_df, start_batch=0):
        """Process all products in batches."""
        if not self.initialize_ai_extractor():
            return None
            
        total_products = len(input_df)
        total_batches = (total_products + self.batch_size - 1) // self.batch_size
        
        print(f"🚀 FULL BATCH PROCESSING STARTED")
        print("=" * 60)
        print(f"📊 Total products: {total_products:,}")
        print(f"📦 Batch size: {self.batch_size}")
        print(f"📋 Total batches: {total_batches}")
        print(f"🎯 Starting from batch: {start_batch + 1}")
        print()
        
        # Create master results file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        master_output = self.results_dir / f"steele_full_batch_results_{timestamp}.csv"
        
        all_results = []
        successful_batches = 0
        failed_batches = 0
        
        start_time = datetime.now()
        
        for batch_num in range(start_batch, total_batches):
            batch_start_idx = batch_num * self.batch_size
            batch_end_idx = min((batch_num + 1) * self.batch_size, total_products)
            
            batch_df = input_df.iloc[batch_start_idx:batch_end_idx].copy()
            
            print(f"🔄 Processing Batch {batch_num + 1}/{total_batches}")
            print(f"   📊 Products {batch_start_idx + 1}-{batch_end_idx} of {total_products:,}")
            print(f"   ⏰ Started: {datetime.now().strftime('%H:%M:%S')}")
            
            try:
                batch_start_time = datetime.now()
                
                # Process this batch
                batch_results = self.extractor.process_unknown_skus_batch_with_expansion(batch_df)
                
                batch_end_time = datetime.now()
                batch_duration = batch_end_time - batch_start_time
                
                # Add batch info to results
                batch_results['batch_number'] = batch_num + 1
                batch_results['processing_timestamp'] = batch_end_time.isoformat()
                
                all_results.append(batch_results)
                successful_batches += 1
                
                # Save individual batch result
                batch_output = self.results_dir / f"batch_{batch_num + 1:04d}_{timestamp}.csv"
                batch_results.to_csv(batch_output, index=False)
                
                print(f"   ✅ Completed in {batch_duration}")
                print(f"   💾 Saved: {batch_output.name}")
                
                # Save cumulative results every 10 batches
                if (batch_num + 1) % 10 == 0 or batch_num == total_batches - 1:
                    cumulative_df = pd.concat(all_results, ignore_index=True)
                    cumulative_df.to_csv(master_output, index=False)
                    print(f"   💾 Updated master file: {master_output.name}")
                
                # Show progress
                elapsed_time = datetime.now() - start_time
                batches_remaining = total_batches - (batch_num + 1)
                
                if successful_batches > 0:
                    avg_time_per_batch = elapsed_time / (batch_num - start_batch + 1)
                    estimated_remaining = avg_time_per_batch * batches_remaining
                    
                    print(f"   📈 Progress: {((batch_num + 1) / total_batches * 100):.1f}%")
                    print(f"   ⏱️  Elapsed: {elapsed_time}")
                    print(f"   🔮 ETA: {estimated_remaining}")
                
                print()
                
            except Exception as e:
                failed_batches += 1
                print(f"   ❌ Batch failed: {e}")
                print(f"   ⚠️  Continuing with next batch...")
                print()
                
                # Save error info
                error_info = pd.DataFrame([{
                    'batch_number': batch_num + 1,
                    'error': str(e),
                    'products_in_batch': len(batch_df),
                    'timestamp': datetime.now().isoformat()
                }])
                
                error_file = self.results_dir / f"batch_errors_{timestamp}.csv"
                if error_file.exists():
                    existing_errors = pd.read_csv(error_file)
                    error_info = pd.concat([existing_errors, error_info], ignore_index=True)
                
                error_info.to_csv(error_file, index=False)
                
                continue
        
        # Final summary
        total_time = datetime.now() - start_time
        
        print("🎉 FULL BATCH PROCESSING COMPLETED!")
        print("=" * 60)
        print(f"📊 Total batches processed: {successful_batches + failed_batches}")
        print(f"✅ Successful batches: {successful_batches}")
        print(f"❌ Failed batches: {failed_batches}")
        print(f"⏱️  Total processing time: {total_time}")
        print(f"💾 Master results file: {master_output}")
        
        if all_results:
            # Create final consolidated results
            final_df = pd.concat(all_results, ignore_index=True)
            final_df.to_csv(master_output, index=False)
            
            # Print final statistics
            total_processed = len(final_df)
            successful_extractions = len(final_df[final_df['extraction_error'].isna() | (final_df['extraction_error'] == '')])
            
            print(f"📈 Products processed: {total_processed:,}")
            print(f"✅ Successful extractions: {successful_extractions:,}")
            print(f"📊 Success rate: {(successful_extractions/total_processed*100):.1f}%")
            
            return str(master_output)
        
        return None

def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description='Steele Full Batch Processing Pipeline')
    parser.add_argument('--batch-size', type=int, default=100, 
                       help='Batch size for processing (default: 100)')
    parser.add_argument('--start-batch', type=int, default=0,
                       help='Starting batch number (0-based, for resuming)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be processed without actually running')
    
    args = parser.parse_args()
    
    processor = SteeleFullBatchProcessor(batch_size=args.batch_size)
    
    print(f"🎯 Steele Full Batch Processing Pipeline")
    print(f"📅 Started at: {datetime.now()}")
    print(f"📦 Batch size: {args.batch_size}")
    print()
    
    # Load data
    input_df = processor.load_unique_products()
    if input_df is None:
        print("❌ Failed to load data")
        sys.exit(1)
    
    if args.dry_run:
        total_batches = (len(input_df) + args.batch_size - 1) // args.batch_size
        print(f"🔍 DRY RUN - Would process:")
        print(f"   📊 Products: {len(input_df):,}")
        print(f"   📦 Batch size: {args.batch_size}")
        print(f"   📋 Total batches: {total_batches}")
        print(f"   🎯 Starting batch: {args.start_batch + 1}")
        return
    
    try:
        result = processor.process_in_batches(input_df, start_batch=args.start_batch)
        
        if result:
            print(f"\n🎉 Processing completed successfully!")
            print(f"📁 Final output: {result}")
        else:
            print(f"\n❌ Processing failed or returned no results")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n⚠️  Processing interrupted by user")
        print(f"💡 You can resume with: --start-batch {args.start_batch}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 