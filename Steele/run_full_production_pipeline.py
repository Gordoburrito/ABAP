#!/usr/bin/env python3
"""
Production script to run the complete Steele transformation pipeline 
on the full processed dataset with AI vehicle tag generation.

This script processes the complete steele_processed_complete.csv file
and generates the final Shopify-ready output.
"""

import pandas as pd
import sys
import time
import os
from pathlib import Path
from datetime import datetime

# Add project paths
project_root = Path(__file__).parent.parent
steele_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.steele_data_transformer import SteeleDataTransformer

class ProductionPipelineRunner:
    """Production pipeline runner with optimizations and monitoring."""
    
    def __init__(self, use_ai: bool = True, batch_size: int = 1000):
        """
        Initialize production pipeline runner.
        
        Args:
            use_ai: Whether to use AI for vehicle tag generation
            batch_size: Number of records to process in each batch
        """
        self.use_ai = use_ai
        self.batch_size = batch_size
        self.steele_root = steele_root
        self.transformer = SteeleDataTransformer(use_ai=use_ai)
        
        # Setup logging
        self.log_file = steele_root / "data" / "results" / f"production_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.log_file.parent.mkdir(exist_ok=True)
    
    def log(self, message: str):
        """Log message to both console and file."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')
    
    def run_full_pipeline(self, input_file: str = "data/processed/steele_processed_complete.csv",
                         output_file: str = "data/results/steele_shopify_complete.csv") -> str:
        """
        Run the complete production pipeline.
        
        Args:
            input_file: Path to processed input data
            output_file: Path for final output
            
        Returns:
            Path to final output file
        """
        start_time = time.time()
        
        self.log("🚀 STARTING FULL PRODUCTION PIPELINE")
        self.log("=" * 60)
        self.log(f"Input file: {input_file}")
        self.log(f"Output file: {output_file}")
        self.log(f"AI enabled: {self.use_ai}")
        self.log(f"Batch size: {self.batch_size}")
        
        try:
            # Step 1: Load and validate input
            self.log("📖 Step 1: Loading processed data...")
            input_path = steele_root / input_file
            
            if not input_path.exists():
                raise FileNotFoundError(f"Input file not found: {input_path}")
            
            # Get file size for progress tracking
            file_size_mb = input_path.stat().st_size / (1024 * 1024)
            self.log(f"   File size: {file_size_mb:.1f} MB")
            
            # Load data in chunks to estimate total rows
            chunk_iter = pd.read_csv(input_path, chunksize=10000)
            first_chunk = next(chunk_iter)
            self.log(f"   Detected columns: {list(first_chunk.columns)}")
            
            # Estimate total rows (rough estimate)
            estimated_rows = int(file_size_mb * 5000)  # Rough estimate
            self.log(f"   Estimated rows: {estimated_rows:,}")
            
            # Step 2: Process in batches
            self.log("🔄 Step 2: Processing in batches...")
            
            output_path = steele_root / output_file
            output_path.parent.mkdir(exist_ok=True)
            
            # Initialize counters
            total_processed = 0
            total_output = 0
            batch_num = 0
            
            # Process first chunk
            batch_results = []
            chunk_iter = pd.read_csv(input_path, chunksize=self.batch_size)
            
            for chunk_df in chunk_iter:
                batch_num += 1
                batch_start = time.time()
                
                self.log(f"   Processing batch {batch_num}: {len(chunk_df)} records")
                
                try:
                    # Run pipeline on this batch
                    batch_result = self._process_batch(chunk_df, batch_num)
                    batch_results.append(batch_result)
                    
                    total_processed += len(chunk_df)
                    total_output += len(batch_result)
                    
                    batch_time = time.time() - batch_start
                    rate = len(chunk_df) / batch_time
                    
                    self.log(f"   ✅ Batch {batch_num}: {len(chunk_df)} → {len(batch_result)} products ({rate:.1f} rec/sec)")
                    
                    # Progress estimate
                    if batch_num % 10 == 0:
                        elapsed = time.time() - start_time
                        estimated_total_time = (elapsed / total_processed) * estimated_rows
                        remaining_time = estimated_total_time - elapsed
                        self.log(f"   📊 Progress: {total_processed:,} processed, ~{remaining_time/60:.1f} min remaining")
                
                except Exception as e:
                    self.log(f"   ❌ Batch {batch_num} failed: {str(e)}")
                    # Continue with next batch
                    continue
            
            # Step 3: Combine all batches
            self.log("🔗 Step 3: Combining batch results...")
            
            if not batch_results:
                raise ValueError("No batches were processed successfully")
            
            # Combine all batch results
            final_df = pd.concat(batch_results, ignore_index=True)
            self.log(f"   Combined {len(batch_results)} batches into {len(final_df)} total records")
            
            # Step 4: Final consolidation across batches
            self.log("📦 Step 4: Final cross-batch consolidation...")
            final_consolidated = self.transformer.consolidate_products_by_unique_id(final_df)
            self.log(f"   Final consolidation: {len(final_df)} → {len(final_consolidated)} unique products")
            
            # Step 5: Save final output
            self.log("💾 Step 5: Saving final output...")
            final_consolidated.to_csv(output_path, index=False)
            
            output_size_mb = output_path.stat().st_size / (1024 * 1024)
            self.log(f"   Saved to: {output_path}")
            self.log(f"   Output size: {output_size_mb:.1f} MB")
            
            # Step 6: Generate final summary
            total_time = time.time() - start_time
            self._generate_final_summary(total_processed, len(final_consolidated), total_time)
            
            self.log("✅ PIPELINE COMPLETED SUCCESSFULLY!")
            return str(output_path)
            
        except Exception as e:
            self.log(f"❌ PIPELINE FAILED: {str(e)}")
            raise
    
    def _process_batch(self, batch_df: pd.DataFrame, batch_num: int) -> pd.DataFrame:
        """
        Process a single batch through the pipeline.
        
        Args:
            batch_df: Batch of data to process
            batch_num: Batch number for logging
            
        Returns:
            Processed DataFrame for this batch
        """
        try:
            # Golden dataset validation
            validation_df = self.transformer.validate_against_golden_dataset(batch_df)
            
            # Transform to standard format
            standard_products = self.transformer.transform_to_standard_format(batch_df, validation_df)
            
            # Template enhancement
            enhanced_products = self.transformer.enhance_with_templates(standard_products)
            
            # Convert to Shopify format
            shopify_df = self.transformer.transform_to_formatted_shopify_import(enhanced_products)
            
            # Batch-level consolidation
            consolidated_df = self.transformer.consolidate_products_by_unique_id(shopify_df)
            
            return consolidated_df
            
        except Exception as e:
            self.log(f"   ❌ Error in batch {batch_num}: {str(e)}")
            raise
    
    def _generate_final_summary(self, total_input: int, total_output: int, total_time: float):
        """Generate final processing summary."""
        
        self.log("\n📊 FINAL PROCESSING SUMMARY")
        self.log("=" * 60)
        self.log(f"Total input records: {total_input:,}")
        self.log(f"Total output products: {total_output:,}")
        self.log(f"Consolidation ratio: {total_input/total_output:.1f}:1")
        self.log(f"Total processing time: {total_time/60:.1f} minutes")
        self.log(f"Processing rate: {total_input/total_time:.1f} records/second")
        
        if self.use_ai:
            self.log("🤖 AI vehicle tag generation: ENABLED")
            self.log("   ✅ Accurate master_ultimate_golden mapping")
            self.log("   ✅ Multi-tag support for generic models")
        else:
            self.log("⚡ AI vehicle tag generation: DISABLED")
            self.log("   ⚡ Fast template-based processing")
        
        self.log(f"📋 Log file: {self.log_file}")

def run_production_pipeline(use_ai: bool = True, batch_size: int = 1000):
    """
    Run the production pipeline with specified settings.
    
    Args:
        use_ai: Whether to use AI for vehicle tag generation
        batch_size: Batch size for processing
    """
    # Change to Steele directory
    os.chdir(steele_root)
    
    runner = ProductionPipelineRunner(use_ai=use_ai, batch_size=batch_size)
    
    try:
        output_file = runner.run_full_pipeline()
        
        print(f"\n🎉 SUCCESS! Final output saved to:")
        print(f"   {output_file}")
        print(f"\n📋 Full log available at:")
        print(f"   {runner.log_file}")
        
        return output_file
        
    except Exception as e:
        print(f"\n💥 PIPELINE FAILED: {str(e)}")
        print(f"\n📋 Check log file for details:")
        print(f"   {runner.log_file}")
        raise

def main():
    """Main function with options for different pipeline configurations."""
    
    print("🚀 STEELE PRODUCTION PIPELINE")
    print("=" * 50)
    
    # Configuration options
    configurations = {
        "1": {"name": "Full AI Pipeline (Recommended)", "use_ai": True, "batch_size": 1000},
        "2": {"name": "Fast Template Pipeline", "use_ai": False, "batch_size": 2000},
        "3": {"name": "Small Batch AI Pipeline", "use_ai": True, "batch_size": 500},
    }
    
    print("Select pipeline configuration:")
    for key, config in configurations.items():
        print(f"  {key}. {config['name']}")
        print(f"     AI: {'Yes' if config['use_ai'] else 'No'}, Batch: {config['batch_size']}")
    
    print(f"  4. Custom configuration")
    print()
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice in configurations:
        config = configurations[choice]
        print(f"\n✅ Selected: {config['name']}")
        run_production_pipeline(use_ai=config['use_ai'], batch_size=config['batch_size'])
    
    elif choice == "4":
        print("\n🛠️  Custom Configuration:")
        use_ai_input = input("Use AI for vehicle tags? (y/n): ").strip().lower()
        use_ai = use_ai_input in ['y', 'yes', '1', 'true']
        
        batch_size_input = input("Batch size (default 1000): ").strip()
        batch_size = int(batch_size_input) if batch_size_input.isdigit() else 1000
        
        print(f"\n✅ Custom: AI={'Yes' if use_ai else 'No'}, Batch={batch_size}")
        run_production_pipeline(use_ai=use_ai, batch_size=batch_size)
    
    else:
        print("❌ Invalid choice. Using default configuration.")
        run_production_pipeline()

if __name__ == "__main__":
    main() 