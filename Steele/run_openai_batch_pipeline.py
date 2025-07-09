#!/usr/bin/env python3

"""
OpenAI Batch API Pipeline for Steele Data Processing
Uses OpenAI's Batch API to process ALL unique products efficiently and cost-effectively
"""

import pandas as pd
import json
import os
import sys
from datetime import datetime
from pathlib import Path
import argparse
import time
from openai import OpenAI

# Add utils to path
sys.path.append(str(Path(__file__).parent))

from utils.ai_fitment_extractor import SteeleAIFitmentExtractor

class SteeleOpenAIBatchProcessor:
    """Process all Steele products using OpenAI's Batch API."""
    
    def __init__(self):
        self.steele_root = Path(__file__).parent
        self.data_dir = self.steele_root / "data"
        self.processed_dir = self.data_dir / "processed"
        self.results_dir = self.data_dir / "results"
        self.batch_dir = self.steele_root / "batch_ids"
        
        # Ensure directories exist
        self.results_dir.mkdir(exist_ok=True)
        self.batch_dir.mkdir(exist_ok=True)
        
        # Initialize OpenAI client
        self.client = OpenAI()
        
        # Initialize extractor for golden master data and prompts
        self.extractor = SteeleAIFitmentExtractor()
        
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
    
    def create_batch_requests(self, input_df):
        """Create batch requests for OpenAI Batch API."""
        print("🔄 Creating batch requests for OpenAI Batch API...")
        
        # Get the extraction prompt from the extractor
        system_prompt = """You are an expert automotive parts specialist. Extract vehicle fitment information from product descriptions.

For each product, analyze the description and extract:
1. Years (as comma-separated list)
2. Makes (as comma-separated list) 
3. Models (as comma-separated list)

IMPORTANT RULES:
- If description is vague like "Independent (1920-1929) automobile manufacturers" or "Street Rod/Custom Build", use "ALL" for make and model
- If specific make but vague models, use the specific make and "ALL" for model
- If no clear fitment info, use "UNKNOWN" for all fields
- Always provide confidence score 0-100

Return JSON format:
{
  "years": ["1931", "1932"],
  "makes": ["Stutz"],
  "models": ["Model MB", "Model MA"],
  "confidence": 85
}

For vague descriptions, return:
{
  "years": ["1920", "1921", "1922", "1923", "1924", "1925", "1926", "1927", "1928", "1929"],
  "makes": ["ALL"],
  "models": ["ALL"],
  "confidence": 70
}"""
        
        batch_requests = []
        
        for idx, row in input_df.iterrows():
            user_prompt = f"""Product: {row['ProductName']}
Description: {row['Description']}

Extract vehicle fitment information following the rules above."""
            
            request = {
                "custom_id": f"steele_{row['StockCode']}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "max_tokens": 500,
                    "temperature": 0.1
                }
            }
            
            batch_requests.append(request)
        
        print(f"✅ Created {len(batch_requests)} batch requests")
        return batch_requests
    
    def upload_batch_file(self, batch_requests):
        """Upload batch requests to OpenAI."""
        print("📤 Uploading batch file to OpenAI...")
        
        # Create JSONL file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_file_path = self.results_dir / f"steele_batch_requests_{timestamp}.jsonl"
        
        with open(batch_file_path, 'w') as f:
            for request in batch_requests:
                f.write(json.dumps(request) + '\n')
        
        print(f"💾 Batch file created: {batch_file_path}")
        print(f"📊 File size: {batch_file_path.stat().st_size / 1024 / 1024:.2f} MB")
        
        # Upload to OpenAI
        try:
            with open(batch_file_path, 'rb') as f:
                batch_input_file = self.client.files.create(
                    file=f,
                    purpose="batch"
                )
            
            print(f"✅ File uploaded successfully")
            print(f"📁 File ID: {batch_input_file.id}")
            
            return batch_input_file.id, batch_file_path
            
        except Exception as e:
            print(f"❌ Failed to upload batch file: {e}")
            return None, batch_file_path
    
    def create_batch_job(self, input_file_id):
        """Create a batch job with OpenAI."""
        print("🚀 Creating batch job with OpenAI...")
        
        try:
            batch = self.client.batches.create(
                input_file_id=input_file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={
                    "description": "Steele automotive parts fitment extraction",
                    "created_at": datetime.now().isoformat()
                }
            )
            
            print(f"✅ Batch job created successfully")
            print(f"🆔 Batch ID: {batch.id}")
            print(f"📊 Status: {batch.status}")
            print(f"⏰ Created: {batch.created_at}")
            
            # Save batch ID for tracking
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            batch_id_file = self.batch_dir / f"batch_id_{timestamp}.txt"
            
            with open(batch_id_file, 'w') as f:
                f.write(f"Batch ID: {batch.id}\n")
                f.write(f"Status: {batch.status}\n")
                f.write(f"Created: {batch.created_at}\n")
                f.write(f"Input File ID: {input_file_id}\n")
                f.write(f"Description: Steele automotive parts fitment extraction\n")
            
            print(f"💾 Batch ID saved to: {batch_id_file}")
            
            return batch.id
            
        except Exception as e:
            print(f"❌ Failed to create batch job: {e}")
            return None

if __name__ == "__main__":
    processor = SteeleOpenAIBatchProcessor()
    
    print(f"🎯 Steele OpenAI Batch API Processing Pipeline")
    print(f"📅 Started at: {datetime.now()}")
    print()
    
    # Load data
    input_df = processor.load_unique_products()
    if input_df is None:
        print("❌ Failed to load data")
        sys.exit(1)
    
    print(f"🔍 Would create batch for:")
    print(f"   📊 Products: {len(input_df):,}")
    print(f"   💰 Estimated cost: ${len(input_df) * 0.00015:.2f} (at $0.00015 per request)")
    print(f"   ⏱️  Estimated processing time: 2-24 hours")
    print()
    
    response = input("🚀 Create batch job? (y/N): ")
    if response.lower() == 'y':
        # Create batch requests
        batch_requests = processor.create_batch_requests(input_df)
        
        # Upload batch file
        input_file_id, batch_file_path = processor.upload_batch_file(batch_requests)
        if not input_file_id:
            sys.exit(1)
        
        # Create batch job
        batch_id = processor.create_batch_job(input_file_id)
        if not batch_id:
            sys.exit(1)
        
        print(f"\n🎉 Batch created successfully!")
        print(f"🆔 Batch ID: {batch_id}")
        print(f"💡 Check status with OpenAI dashboard or API")
    else:
        print("❌ Batch creation cancelled")
