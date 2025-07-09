#!/usr/bin/env python3

"""
Download and Process OpenAI Batch Results for Steele Data
"""

import pandas as pd
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from openai import OpenAI

# Add utils to path
sys.path.append(str(Path(__file__).parent))

from utils.ai_fitment_extractor import SteeleAIFitmentExtractor

class SteeleBatchResultsProcessor:
    """Download and process completed OpenAI batch results."""
    
    def __init__(self):
        self.steele_root = Path(__file__).parent
        self.data_dir = self.steele_root / "data"
        self.results_dir = self.data_dir / "results"
        
        # Ensure directories exist
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize OpenAI client
        self.client = OpenAI()
        
        # Initialize extractor for golden master data and expansion logic
        self.extractor = SteeleAIFitmentExtractor()
        
    def check_batch_status(self, batch_id):
        """Check the status of a batch job."""
        try:
            batch = self.client.batches.retrieve(batch_id)
            
            print(f"🆔 Batch ID: {batch.id}")
            print(f"📊 Status: {batch.status}")
            print(f"⏰ Created: {batch.created_at}")
            
            if hasattr(batch, 'request_counts'):
                counts = batch.request_counts
                print(f"📈 Progress:")
                print(f"   Total: {counts.total}")
                print(f"   Completed: {counts.completed}")
                print(f"   Failed: {counts.failed}")
                
                if counts.total > 0:
                    progress = (counts.completed / counts.total) * 100
                    print(f"   Progress: {progress:.1f}%")
            
            if batch.status == "completed":
                print(f"✅ Batch completed successfully!")
                if hasattr(batch, 'output_file_id'):
                    print(f"📁 Output file ID: {batch.output_file_id}")
                    return batch.output_file_id
            elif batch.status == "failed":
                print(f"❌ Batch failed!")
                if hasattr(batch, 'errors'):
                    print(f"🔍 Errors: {batch.errors}")
            elif batch.status in ["validating", "in_progress"]:
                print(f"⏳ Batch is still processing...")
            
            return None
            
        except Exception as e:
            print(f"❌ Failed to check batch status: {e}")
            return None
    
    def download_batch_results(self, output_file_id):
        """Download and process batch results."""
        print(f"📥 Downloading batch results...")
        
        try:
            # Download the results file
            file_response = self.client.files.content(output_file_id)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = self.results_dir / f"steele_batch_results_{timestamp}.jsonl"
            
            with open(results_file, 'wb') as f:
                f.write(file_response.content)
            
            print(f"💾 Raw results downloaded to: {results_file}")
            
            # Process the results
            return self.process_batch_results(results_file)
            
        except Exception as e:
            print(f"❌ Failed to download results: {e}")
            return None
    
    def process_batch_results(self, results_file):
        """Process the batch results and expand fitment data."""
        print("🔄 Processing batch results...")
        
        results = []
        
        try:
            with open(results_file, 'r') as f:
                for line in f:
                    result = json.loads(line.strip())
                    results.append(result)
            
            print(f"📊 Loaded {len(results)} results")
            
            # Process each result
            processed_results = []
            
            for result in results:
                stock_code = result['custom_id'].replace('steele_', '')
                
                try:
                    # Extract the AI response
                    response_content = result['response']['body']['choices'][0]['message']['content']
                    
                    # Parse JSON response
                    fitment_data = json.loads(response_content)
                    
                    # Expand fitment using golden master (similar to two-pass approach)
                    expanded_tags = self.expand_fitment_to_tags(fitment_data)
                    
                    processed_results.append({
                        'StockCode': stock_code,
                        'ai_extracted_years': ', '.join(fitment_data.get('years', [])),
                        'ai_extracted_makes': ', '.join(fitment_data.get('makes', [])),
                        'ai_extracted_models': ', '.join(fitment_data.get('models', [])),
                        'ai_confidence': fitment_data.get('confidence', 0),
                        'generated_tags': ', '.join(expanded_tags),
                        'extraction_error': '',
                        'raw_ai_response': response_content
                    })
                    
                except Exception as e:
                    processed_results.append({
                        'StockCode': stock_code,
                        'ai_extracted_years': '',
                        'ai_extracted_makes': '',
                        'ai_extracted_models': '',
                        'ai_confidence': 0,
                        'generated_tags': '0_Unknown_UNKNOWN',
                        'extraction_error': str(e),
                        'raw_ai_response': str(result.get('response', {}))
                    })
            
            # Create DataFrame
            results_df = pd.DataFrame(processed_results)
            
            # Save processed results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.results_dir / f"steele_batch_processed_{timestamp}.csv"
            results_df.to_csv(output_file, index=False)
            
            print(f"💾 Processed results saved to: {output_file}")
            
            # Print summary
            self.print_processing_summary(results_df)
            
            return str(output_file)
            
        except Exception as e:
            print(f"❌ Failed to process results: {e}")
            return None
    
    def expand_fitment_to_tags(self, fitment_data):
        """Expand fitment data to vehicle tags using golden master."""
        tags = []
        
        try:
            years = fitment_data.get('years', [])
            makes = fitment_data.get('makes', [])
            models = fitment_data.get('models', [])
            
            # Handle "ALL" cases by expanding with golden master
            if 'ALL' in makes:
                # Get all makes for the given years
                expanded_makes = self.extractor.expand_all_makes(years)
                makes = expanded_makes
            
            if 'ALL' in models:
                # Get all models for the given makes and years
                expanded_models = []
                for make in makes:
                    if make != 'ALL':
                        models_for_make = self.extractor.expand_all_models(make, years)
                        expanded_models.extend(models_for_make)
                models = list(set(expanded_models))
            
            # Generate tags
            for year in years:
                for make in makes:
                    for model in models:
                        if year and make and model and make != 'ALL' and model != 'ALL':
                            # Use golden master format: preserve spaces in model names
                            tag = f"{year}_{make.replace(' ', '_')}_{model.replace(' ', '_')}"
                            tags.append(tag)
            
            # If no valid tags generated, return unknown
            if not tags:
                tags = ['0_Unknown_UNKNOWN']
                
        except Exception as e:
            print(f"⚠️  Error expanding fitment: {e}")
            tags = ['0_Unknown_UNKNOWN']
        
        return tags
    
    def print_processing_summary(self, results_df):
        """Print a summary of processing results."""
        print(f"\n📈 PROCESSING SUMMARY:")
        print("=" * 50)
        
        total_products = len(results_df)
        successful = len(results_df[results_df['extraction_error'] == ''])
        
        print(f"   Total products processed: {total_products:,}")
        print(f"   Successful extractions: {successful:,}")
        print(f"   Success rate: {(successful/total_products*100):.1f}%")
        
        # Count improvements (products that got real tags instead of 0_Unknown_UNKNOWN)
        improved = len(results_df[~results_df['generated_tags'].str.contains('0_Unknown_UNKNOWN', na=False)])
        
        print(f"   Products with improved tags: {improved:,}")
        print(f"   Improvement rate: {(improved/total_products*100):.1f}%")
        
        # Sample of improvements
        if improved > 0:
            print(f"\n📋 Sample improvements:")
            improved_samples = results_df[~results_df['generated_tags'].str.contains('0_Unknown_UNKNOWN', na=False)].head(5)
            for idx, row in improved_samples.iterrows():
                print(f"   {row['StockCode']}: {row['generated_tags'][:100]}...")

def main():
    """Main function to process the specified batch."""
    batch_id = "batch_685c79d08ee481909d5ed6756b4add5e"
    
    processor = SteeleBatchResultsProcessor()
    
    print(f"🎯 Steele Batch Results Processor")
    print(f"📅 Started at: {datetime.now()}")
    print(f"🆔 Processing batch: {batch_id}")
    print()
    
    # Check batch status and get output file ID
    output_file_id = processor.check_batch_status(batch_id)
    
    if output_file_id:
        print(f"\n📥 Downloading and processing results...")
        result_file = processor.download_batch_results(output_file_id)
        
        if result_file:
            print(f"\n🎉 Processing completed successfully!")
            print(f"📁 Final CSV output: {result_file}")
        else:
            print(f"❌ Failed to process results")
    else:
        print(f"❌ Batch not ready or failed")

if __name__ == "__main__":
    main() 