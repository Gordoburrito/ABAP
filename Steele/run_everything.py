#!/usr/bin/env python3
"""
Master Script for Steele Data Processing Pipeline
Provides options to run different processing modes:
1. Process Unknown SKUs (products with 0_Unknown_UNKNOWN tags)
2. Process specific batch sizes (100, 500, 1000, etc.)
3. Run full production pipeline
4. Test with small samples
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

class SteeleMasterProcessor:
    """Master processor for all Steele data processing operations."""
    
    def __init__(self):
        self.steele_root = Path(__file__).parent
        self.data_dir = self.steele_root / "data"
        self.results_dir = self.data_dir / "results"
        self.processed_dir = self.data_dir / "processed"
        self.samples_dir = self.data_dir / "samples"
        
        # Ensure directories exist
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize AI extractor
        self.extractor = None
        
    def check_environment(self):
        """Check if environment is properly set up."""
        print("🔍 Checking environment...")
        
        # Check OpenAI API key
        if not os.getenv('OPENAI_API_KEY'):
            print("❌ Error: OPENAI_API_KEY environment variable not set")
            print("Please set your OpenAI API key: export OPENAI_API_KEY='your-key-here'")
            return False
            
        # Check for required data files
        steele_processed = self.processed_dir / "steele_processed_complete.csv"
        if not steele_processed.exists():
            print(f"❌ Error: Main data file not found: {steele_processed}")
            return False
            
        print("✅ Environment check passed")
        return True
    
    def initialize_ai_extractor(self):
        """Initialize the AI fitment extractor."""
        if self.extractor is None:
            print("🤖 Initializing AI Fitment Extractor...")
            try:
                self.extractor = SteeleAIFitmentExtractor()
                print("✅ AI Fitment Extractor initialized successfully")
            except Exception as e:
                print(f"❌ Error initializing AI extractor: {e}")
                return False
        return True
    
    def load_unknown_skus(self, limit=None):
        """Load products with 0_Unknown_UNKNOWN tags from processed data."""
        print(f"📖 Loading products with 0_Unknown_UNKNOWN tags...")
        
        # Try to load from existing results first
        results_files = list(self.results_dir.glob("first_100_products_*.csv"))
        if results_files:
            latest_results = max(results_files, key=lambda x: x.stat().st_mtime)
            print(f"📁 Loading from existing results: {latest_results.name}")
            df = pd.read_csv(latest_results)
            
            # Filter for unknown products
            unknown_mask = df['Tags'].str.contains('0_Unknown_UNKNOWN', na=False)
            unknown_products = df[unknown_mask].copy()
            
            if len(unknown_products) > 0:
                print(f"🔍 Found {len(unknown_products)} products with 0_Unknown_UNKNOWN tags")
                
                # Create input format for AI extractor
                input_df = pd.DataFrame({
                    'StockCode': unknown_products['Variant SKU'],
                    'ProductName': unknown_products['Title'],
                    'Description': unknown_products['Body HTML']
                })
                
                if limit:
                    input_df = input_df.head(limit)
                    print(f"📊 Limited to first {len(input_df)} products")
                    
                return input_df
        
        # If no results file, load from processed data
        steele_processed = self.processed_dir / "steele_processed_complete.csv"
        if steele_processed.exists():
            print(f"📁 Loading from processed data: {steele_processed.name}")
            # Load first chunk to check structure
            df_sample = pd.read_csv(steele_processed, nrows=1000)
            print(f"📊 Sample columns: {list(df_sample.columns)}")
            
            # Create sample unknown SKUs
            input_df = pd.DataFrame({
                'StockCode': df_sample['StockCode'].head(limit or 100),
                'ProductName': df_sample['Product Name'].head(limit or 100),
                'Description': df_sample['Description'].head(limit or 100)
            })
            
            print(f"📊 Created sample of {len(input_df)} products for testing")
            return input_df
        
        print("❌ No suitable data files found")
        return None
    
    def process_unknown_skus(self, limit=None):
        """Process products with unknown SKUs using two-pass AI extraction."""
        print("🚀 Processing Unknown SKUs with Two-Pass AI Extraction")
        print("=" * 60)
        
        if not self.initialize_ai_extractor():
            return None
            
        # Load unknown SKUs
        input_df = self.load_unknown_skus(limit)
        if input_df is None or len(input_df) == 0:
            print("❌ No unknown SKUs found to process")
            return None
        
        print(f"\n📋 Sample products to process:")
        for idx, row in input_df.head(3).iterrows():
            print(f"  {idx+1}. {row['StockCode']}: {row['ProductName']}")
            print(f"     Description: {row['Description'][:100]}...")
            print()
        
        if len(input_df) > 3:
            print(f"  ... and {len(input_df) - 3} more products")
        
        # Process with AI
        print(f"\n🔄 Processing {len(input_df)} products...")
        start_time = datetime.now()
        
        try:
            results_df = self.extractor.process_unknown_skus_batch_with_expansion(input_df)
            end_time = datetime.now()
            
            processing_time = end_time - start_time
            print(f"\n⏱️  Processing completed in {processing_time}")
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.results_dir / f"unknown_skus_processed_{timestamp}.csv"
            results_df.to_csv(output_file, index=False)
            
            print(f"\n💾 Results saved to: {output_file}")
            
            # Print summary
            self.print_processing_summary(input_df, results_df)
            
            return str(output_file)
            
        except Exception as e:
            print(f"❌ Error during processing: {e}")
            return None
    
    def process_batch_size(self, batch_size=100):
        """Process a specific batch size of products."""
        print(f"🚀 Processing Batch of {batch_size} Products")
        print("=" * 60)
        
        return self.process_unknown_skus(limit=batch_size)
    
    def run_full_pipeline(self):
        """Run the complete production pipeline."""
        print("🚀 Running Full Production Pipeline")
        print("=" * 60)
        
        try:
            from run_full_production_pipeline import ProductionPipelineRunner
            
            runner = ProductionPipelineRunner(use_ai=True, batch_size=1000)
            output_file = runner.run_full_pipeline()
            
            print(f"\n✅ Full pipeline completed!")
            print(f"📁 Output saved to: {output_file}")
            
            return output_file
            
        except ImportError:
            print("❌ Full production pipeline script not available")
            return None
        except Exception as e:
            print(f"❌ Error running full pipeline: {e}")
            return None
    
    def run_test_sample(self, sample_size=5):
        """Run a small test sample to verify everything is working."""
        print(f"🧪 Running Test Sample ({sample_size} products)")
        print("=" * 40)
        
        # Create a small test dataset
        test_data = pd.DataFrame([
            {
                'StockCode': '10-0108-45',
                'ProductName': 'Glass weatherstrip kit',
                'Description': 'Running Board Step Pad. Made from top quality rubber to ensure durability. This part is compatible with models built by Independent (1920 - 1929) automobile manufacturers.'
            },
            {
                'StockCode': '10-0127-52', 
                'ProductName': 'Windshield to Cowl Weatherstrip',
                'Description': 'This Windshield to Cowl Weatherstrip comes in a thirty-six inch (36") strip and seals the lower windshield frame to the cowl of the car. This generic part is known to fit Independent vehicle models produced from 1920 - 1924.'
            },
            {
                'StockCode': '10-0001-40',
                'ProductName': 'Accelerator Pedal Pad',
                'Description': 'Pad, accelerator pedal. Cements and fastens with screws to original metal pedal. Length: 6.50 inches. For 1931 Stutz Model MB vehicles.'
            },
            {
                'StockCode': '10-0128-35',
                'ProductName': 'Pad, rear axle rebound',
                'Description': 'Axle Rebound Pad - Rear. Made from top quality rubber to ensure durability. This part is compatible with (1912 - 1915) Sterns-Knight models.'
            },
            {
                'StockCode': '10-0130-108',
                'ProductName': 'Accelerator Pedal Pad',
                'Description': 'Steele Rubber Products has the best quality Accelerator Pedal Pads for your Street Rod or Custom Build project. These Accelerator Pads are made of top-quality EPDM rubber with a high level of fit and finish for the Builder\'s creative application and use.'
            }
        ])
        
        # Limit to requested sample size
        test_data = test_data.head(sample_size)
        
        if not self.initialize_ai_extractor():
            return None
        
        print(f"🔄 Processing {len(test_data)} test products...")
        
        try:
            results_df = self.extractor.process_unknown_skus_batch_with_expansion(test_data)
            
            # Save test results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.results_dir / f"test_sample_{sample_size}_products_{timestamp}.csv"
            results_df.to_csv(output_file, index=False)
            
            print(f"\n💾 Test results saved to: {output_file}")
            
            # Print detailed results for test
            print(f"\n📊 TEST RESULTS:")
            print("=" * 50)
            
            for idx, row in results_df.iterrows():
                print(f"\n🏷️  {row['StockCode']}: {row['ProductName']}")
                print(f"   Generated Tags: {row['generated_tags']}")
                print(f"   Confidence: {row['ai_confidence']:.2f}")
                print(f"   Make: {row['ai_extracted_make']}")
                print(f"   Model: {row['ai_extracted_model']}")
                
                if row['extraction_error']:
                    print(f"   ⚠️  Error: {row['extraction_error']}")
                else:
                    print("   ✅ Success")
            
            return str(output_file)
            
        except Exception as e:
            print(f"❌ Error during test: {e}")
            return None
    
    def print_processing_summary(self, input_df, results_df):
        """Print a summary of processing results."""
        print(f"\n📈 PROCESSING SUMMARY:")
        print("=" * 50)
        
        successful = len(results_df[results_df['extraction_error'].isna() | (results_df['extraction_error'] == '')])
        
        print(f"   Total products processed: {len(input_df)}")
        print(f"   Successful extractions: {successful}")
        print(f"   Success rate: {(successful/len(input_df)*100):.1f}%")
        
        # Count improvements (products that got real tags instead of 0_Unknown_UNKNOWN)
        improved = 0
        for idx, row in results_df.iterrows():
            if '0_Unknown_UNKNOWN' not in str(row['generated_tags']):
                improved += 1
        
        print(f"   Products with improved tags: {improved}")
        print(f"   Improvement rate: {(improved/len(input_df)*100):.1f}%")
        
        # Sample of improvements
        if improved > 0:
            print(f"\n📋 Sample improvements:")
            count = 0
            for idx, row in results_df.iterrows():
                if '0_Unknown_UNKNOWN' not in str(row['generated_tags']) and count < 3:
                    print(f"   {row['StockCode']}: {row['generated_tags']}")
                    count += 1

def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description='Steele Data Processing Master Script')
    parser.add_argument('mode', choices=['test', 'unknown', 'batch', 'full'], 
                       help='Processing mode: test (small sample), unknown (all unknown SKUs), batch (specific size), full (complete pipeline)')
    parser.add_argument('--size', type=int, default=100, 
                       help='Batch size or test size (default: 100)')
    parser.add_argument('--limit', type=int, 
                       help='Limit number of products to process')
    
    args = parser.parse_args()
    
    processor = SteeleMasterProcessor()
    
    # Check environment
    if not processor.check_environment():
        sys.exit(1)
    
    print(f"🎯 Running in {args.mode} mode")
    print(f"📅 Started at: {datetime.now()}")
    print()
    
    try:
        if args.mode == 'test':
            result = processor.run_test_sample(args.size)
        elif args.mode == 'unknown':
            result = processor.process_unknown_skus(args.limit)
        elif args.mode == 'batch':
            result = processor.process_batch_size(args.size)
        elif args.mode == 'full':
            result = processor.run_full_pipeline()
        
        if result:
            print(f"\n🎉 Processing completed successfully!")
            print(f"📁 Output location: {result}")
        else:
            print(f"\n❌ Processing failed or returned no results")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n⚠️  Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 