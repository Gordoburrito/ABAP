from utils.batch_steele_data_transformer import BatchSteeleDataTransformer
import time
import os
import sys
from pathlib import Path

def main(submit_only=False):
    """
    Main entry point for Steele data transformation using Batch API.
    Uses OpenAI's Batch API for 50% cost savings on AI requests.
    
    Args:
        submit_only: If True, only submit the batch and save the batch ID for later
    """
    try:
        # Get the directory where this script is located (Steele directory)
        steele_dir = Path(__file__).parent

        if submit_only:
            print("=" * 60)
            print("🚀 STEELE BATCH SUBMISSION ONLY")
            print("   Submitting batch job - get results later")
            print("   50% cost savings with Batch API")
            print("=" * 60)
            
            # Initialize batch transformer
            transformer = BatchSteeleDataTransformer(use_ai=True)

            print("🔄 Step 1: Loading and validating data...")
            # final_df = transformer.process_complete_pipeline_batch("data/processed/steele_processed_complete.csv")

            steele_df = transformer.load_sample_data("data/processed/steele_processed_complete.csv")
            print(f"✅ Loaded {len(steele_df):,} products")

            print("🔄 Step 2: Golden master validation (queuing AI tasks)...")
            transformer.load_golden_dataset()
            validation_df = transformer.validate_against_golden_dataset_batch(steele_df)
            
            ai_tasks = transformer.batch_ai_matcher.get_queue_size()
            if ai_tasks == 0:
                print("ℹ️  No AI tasks needed - all products have exact matches")
                print("✅ Processing can complete immediately without batch API")
                return "no_batch_needed"

            print(f"🔄 Step 3: Submitting {ai_tasks} AI tasks to batch API...")
            batch_id = transformer.batch_ai_matcher.process_batch("steele_validation")
            
            if batch_id:
                print(f"✅ Batch submitted successfully!")
                print(f"📋 Batch ID: {batch_id}")
                print(f"⏳ Processing time: Up to 24 hours")
                print(f"💰 Cost savings: 50% with Batch API")
                print("")
                print("📝 To retrieve results later:")
                print(f"   python main.py --retrieve {batch_id}")
                
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
        else:
            print("=" * 60)
            print("🚀 STEELE BATCH API TRANSFORMATION")
            print("   Using OpenAI Batch API for 50% cost savings")
            print("   Template-Based + Batch AI • Ultra-Efficient")
            print("=" * 60)

            # Initialize batch transformer with AI enabled
            transformer = BatchSteeleDataTransformer(use_ai=True)

            # Process with batch API
            # For testing with small sample:
            # final_df = transformer.process_complete_pipeline_batch("data/samples/steele_test_1000.csv")

            # # Or for full processing:
            final_df = transformer.process_complete_pipeline_batch("data/processed/steele_processed_complete.csv")

            # Save results with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = steele_dir / "data" / "transformed" / f"steele_batch_transformed_{timestamp}.csv"
            os.makedirs(output_file.parent, exist_ok=True)

            # Save the results
            final_df.to_csv(output_file, index=False)
            print(f"💾 Results saved to: {output_file}")

            # Display summary
            print("")
            print("=" * 60)
            print("✅ BATCH TRANSFORMATION COMPLETE")
            print(f"   Products processed: {len(final_df)}")
            print(f"   Processing method: Template-based + Batch AI")
            print(f"   Golden validated: {len(final_df[final_df.get('Tags', '') != ''])} products")
            print(f"   Output file: {output_file}")
            print(f"   Cost savings: 50% with Batch API")
            print("=" * 60)

            return str(output_file)

    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")
        raise

def retrieve_results(batch_id):
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
        
        # Re-run validation to rebuild the ai_processing_needed state
        print("🔄 Rebuilding validation state...")
        validation_df = transformer.validate_against_golden_dataset_batch(steele_df)
        
        # Check if we have batch results
        if not transformer.batch_ai_matcher.batch_results:
            print("⚠️  No batch results available - processing without AI results")
        else:
            print(f"✅ Found {len(transformer.batch_ai_matcher.batch_results)} batch results")
            
            # Clear the queue since we're using existing results
            transformer.batch_ai_matcher.clear_batch_queue()
            print("🔄 Cleared batch queue to use existing results")
        
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
    if len(sys.argv) > 1:
        if sys.argv[1] == "--submit-only":
            # Submit batch only and save batch ID
            batch_id = main(submit_only=True)
            if batch_id and batch_id != "no_batch_needed":
                print(f"\n🎉 Batch submitted successfully: {batch_id}")
                print("📝 Save this batch ID to retrieve results later!")
        
        elif sys.argv[1] == "--retrieve" and len(sys.argv) > 2:
            # Retrieve results from completed batch
            batch_id = sys.argv[2]
            result_file = retrieve_results(batch_id)
            if result_file:
                print(f"\n🎉 Processing completed: {result_file}")
        
        elif sys.argv[1] == "--status" and len(sys.argv) > 2:
            # Check batch status
            from utils.batch_steele_data_transformer import BatchSteeleDataTransformer
            batch_id = sys.argv[2]
            transformer = BatchSteeleDataTransformer(use_ai=True)
            status = transformer.batch_ai_matcher.check_batch_status(batch_id)
            
            if status == "completed":
                print("✅ Batch completed! Ready to retrieve results.")
                print(f"Run: python main.py --retrieve {batch_id}")
            elif status in ["validating", "in_progress"]:
                print("⏳ Batch is still processing. Check back later.")
            elif status in ["failed", "expired", "cancelled"]:
                print("❌ Batch failed or was cancelled.")
        
        else:
            print("Usage:")
            print("  python main.py                        # Complete processing (wait for batch)")
            print("  python main.py --submit-only          # Submit batch and get ID for later")
            print("  python main.py --status <batch_id>    # Check batch status")
            print("  python main.py --retrieve <batch_id>  # Retrieve completed results")
            sys.exit(1)
    else:
        # Default: complete processing with waiting
        main()
