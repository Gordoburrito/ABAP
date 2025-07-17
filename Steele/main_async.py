from utils.batch_steele_data_transformer import BatchSteeleDataTransformer
import time
import os
import sys
from pathlib import Path

def submit_batch():
    """
    Submit a batch job and return the batch ID for later processing.
    Use this when you want to submit the batch and check results later.
    """
    try:
        steele_dir = Path(__file__).parent

        print("=" * 60)
        print("🚀 STEELE BATCH API SUBMISSION")
        print("   Submitting batch job for asynchronous processing")
        print("   Check back later with retrieve_results.py")
        print("=" * 60)

        # Initialize batch transformer
        transformer = BatchSteeleDataTransformer(use_ai=True)

        print("🔄 Step 1: Loading and validating data...")
        steele_df = transformer.load_sample_data("data/samples/steele_test_1000.csv")
        print(f"✅ Loaded {len(steele_df):,} products")

        print("🔄 Step 2: Golden master validation (queuing AI tasks)...")
        transformer.load_golden_dataset()
        validation_df = transformer.validate_against_golden_dataset_batch(steele_df)
        
        ai_tasks = transformer.batch_ai_matcher.get_queue_size()
        if ai_tasks == 0:
            print("ℹ️  No AI tasks needed - all products have exact matches")
            return None

        print(f"🔄 Step 3: Submitting {ai_tasks} AI tasks to batch API...")
        batch_id = transformer.batch_ai_matcher.process_batch("steele_validation")
        
        if batch_id:
            print(f"✅ Batch submitted successfully!")
            print(f"📋 Batch ID: {batch_id}")
            print(f"⏳ Processing time: Up to 24 hours")
            print(f"💰 Cost savings: 50% with Batch API")
            print("")
            print("📝 To check status and retrieve results:")
            print(f"   python retrieve_results.py {batch_id}")
            
            # Save batch info for later retrieval
            batch_info_file = steele_dir / "data" / "batch" / f"batch_info_{batch_id}.txt"
            os.makedirs(batch_info_file.parent, exist_ok=True)
            
            with open(batch_info_file, "w") as f:
                f.write(f"Batch ID: {batch_id}\n")
                f.write(f"Submitted: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"AI Tasks: {ai_tasks}\n")
                f.write(f"Input File: data/samples/steele_test_1000.csv\n")
            
            print(f"💾 Batch info saved to: {batch_info_file}")
            
            return batch_id
        else:
            print("❌ Failed to submit batch")
            return None

    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")
        raise

def check_batch_status(batch_id: str):
    """
    Check the status of a submitted batch job.
    
    Args:
        batch_id: The batch ID to check
    """
    try:
        print(f"📊 Checking status for batch: {batch_id}")
        
        transformer = BatchSteeleDataTransformer(use_ai=True)
        status = transformer.batch_ai_matcher.check_batch_status(batch_id)
        
        print(f"Current status: {status}")
        
        if status == "completed":
            print("✅ Batch completed! Ready to retrieve results.")
            print(f"Run: python retrieve_results.py {batch_id}")
        elif status in ["validating", "in_progress"]:
            print("⏳ Batch is still processing. Check back later.")
        elif status in ["failed", "expired", "cancelled"]:
            print("❌ Batch failed or was cancelled.")
        
        return status

    except Exception as e:
        print(f"❌ Error checking batch status: {str(e)}")
        return "error"

def retrieve_and_process_results(batch_id: str):
    """
    Retrieve results from a completed batch and finish processing.
    
    Args:
        batch_id: The completed batch ID
    """
    try:
        steele_dir = Path(__file__).parent

        print("=" * 60)
        print("🚀 STEELE BATCH RESULTS RETRIEVAL")
        print(f"   Retrieving results for batch: {batch_id}")
        print("=" * 60)

        # Initialize transformer and retrieve results
        transformer = BatchSteeleDataTransformer(use_ai=True)
        
        print("📥 Retrieving batch results...")
        if not transformer.batch_ai_matcher.retrieve_batch_results(batch_id):
            print("❌ Failed to retrieve batch results")
            return None

        # Now complete the processing pipeline
        print("🔄 Reloading data to complete processing...")
        steele_df = transformer.load_sample_data("data/samples/steele_test_1000.csv")
        transformer.load_golden_dataset()
        
        # Re-run validation (this time it will be fast since no AI calls needed)
        validation_df = transformer.validate_against_golden_dataset_batch(steele_df)
        
        # Update with AI results
        print("🔄 Updating validation with AI results...")
        validation_df = transformer.update_validation_with_ai_results(validation_df)
        validated_count = len(validation_df[validation_df['golden_validated'] == True])
        print(f"✅ {validated_count}/{len(steele_df)} products validated after AI processing")

        # Continue with rest of pipeline
        print("🔄 Completing transformation pipeline...")
        standard_products = transformer.transform_to_standard_format(steele_df, validation_df)
        enhanced_products = transformer.enhance_with_templates(standard_products)
        final_df = transformer.transform_to_formatted_shopify_import(enhanced_products)

        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = steele_dir / "data" / "transformed" / f"steele_batch_completed_{timestamp}.csv"
        os.makedirs(output_file.parent, exist_ok=True)

        final_df.to_csv(output_file, index=False)
        print(f"💾 Results saved to: {output_file}")

        # Display summary
        print("")
        print("=" * 60)
        print("✅ BATCH PROCESSING COMPLETE")
        print(f"   Batch ID: {batch_id}")
        print(f"   Products processed: {len(final_df)}")
        print(f"   Golden validated: {validated_count}")
        print(f"   Output file: {output_file}")
        print(f"   Cost savings: 50% with Batch API")
        print("=" * 60)

        # Show final cost report
        transformer.batch_ai_matcher.print_cost_report()

        return str(output_file)

    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python main_async.py submit                    # Submit new batch")
        print("  python main_async.py status <batch_id>         # Check batch status")
        print("  python main_async.py retrieve <batch_id>       # Retrieve completed results")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "submit":
        batch_id = submit_batch()
        if batch_id:
            print(f"\n🎉 Batch submitted: {batch_id}")
    
    elif command == "status":
        if len(sys.argv) < 3:
            print("❌ Batch ID required for status check")
            sys.exit(1)
        batch_id = sys.argv[2]
        check_batch_status(batch_id)
    
    elif command == "retrieve":
        if len(sys.argv) < 3:
            print("❌ Batch ID required for retrieval")
            sys.exit(1)
        batch_id = sys.argv[2]
        retrieve_and_process_results(batch_id)
    
    else:
        print(f"❌ Unknown command: {command}")
        print("Valid commands: submit, status, retrieve")
        sys.exit(1)