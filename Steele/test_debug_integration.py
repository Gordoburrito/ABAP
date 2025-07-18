#!/usr/bin/env python3
"""
Test script to demonstrate the enhanced debugging for batch match integration.
This will show the improved debugging output when processing batch results.
"""

import pandas as pd
from pathlib import Path
from utils.batch_steele_data_transformer import BatchSteeleDataTransformer

def test_enhanced_debugging():
    """Test the enhanced debugging capabilities."""
    
    print("🔧 TESTING ENHANCED BATCH MATCH DEBUGGING")
    print("=" * 60)
    
    try:
        # Initialize transformer
        transformer = BatchSteeleDataTransformer(use_ai=True)
        
        # Load a small sample for testing
        print("📊 Loading sample data...")
        steele_df = transformer.load_sample_data("data/samples/steele_test_1000.csv")
        print(f"✅ Loaded {len(steele_df)} products")
        
        # Load golden dataset
        print("📊 Loading golden dataset...")
        transformer.load_golden_dataset()
        print(f"✅ Loaded golden dataset")
        
        # Check if we have batch results from previous runs
        batch_results_dir = Path("data/batch")
        jsonl_files = list(batch_results_dir.glob("batch_results_*.jsonl"))
        
        if not jsonl_files:
            print("❌ No batch results found. Run main.py --submit-only first.")
            return
        
        # Use the most recent batch results file
        jsonl_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        latest_batch = jsonl_files[0]
        
        print(f"📁 Using batch results: {latest_batch.name}")
        
        # Try to retrieve batch results (they should already be there)
        batch_id = latest_batch.stem.replace("batch_results_", "")
        
        # Load batch results if they exist
        if transformer.batch_ai_matcher.retrieve_batch_results(batch_id):
            print(f"✅ Loaded {len(transformer.batch_ai_matcher.batch_results)} batch results")
        else:
            print("⚠️  Could not load batch results")
            return
        
        # Test with just a few products to see the debugging
        test_df = steele_df.head(10)  # Just first 10 products
        
        print("\n🔄 Running validation to trigger AI processing...")
        validation_df = transformer.validate_against_golden_dataset_batch(test_df)
        
        print(f"\n🔄 Testing update_validation_with_ai_results with enhanced debugging...")
        
        # This will show the enhanced debugging output
        updated_validation_df = transformer.update_validation_with_ai_results(validation_df)
        
        print(f"\n📊 Results Summary:")
        print(f"   Total products tested: {len(updated_validation_df)}")
        validated_count = len(updated_validation_df[updated_validation_df['golden_validated'] == True])
        print(f"   Successfully validated: {validated_count}")
        failed_count = len(updated_validation_df[updated_validation_df['golden_validated'] == False])
        print(f"   Failed validation: {failed_count}")
        
        # Show examples of each type
        if validated_count > 0:
            print(f"\n✅ Example successful validation:")
            success_example = updated_validation_df[updated_validation_df['golden_validated'] == True].iloc[0]
            print(f"   Stock Code: {success_example.get('stock_code', 'N/A')}")
            print(f"   Car IDs: {success_example.get('car_ids', [])}")
            print(f"   Match Type: {success_example.get('match_type', 'N/A')}")
        
        if failed_count > 0:
            print(f"\n❌ Example failed validation:")
            failed_example = updated_validation_df[updated_validation_df['golden_validated'] == False].iloc[0]
            print(f"   Stock Code: {failed_example.get('stock_code', 'N/A')}")
            print(f"   Match Type: {failed_example.get('match_type', 'N/A')}")
        
        print(f"\n💡 The enhanced debugging should have shown:")
        print(f"   • Detailed AI response analysis for empty matches")
        print(f"   • Available car_ids and models for failed matches")
        print(f"   • Raw AI responses for troubleshooting")
        print(f"   • Input parameters that led to failures")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_debugging()