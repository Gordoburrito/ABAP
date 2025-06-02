from utils.steele_data_transformer import SteeleDataTransformer
import time
import os

def main():
    """
    Main entry point for Steele data transformation.
    Following @completed-data.mdc rule: NO AI usage, template-based processing only.
    """
    try:
        print("=" * 60)
        print("🚀 STEELE DATA TRANSFORMATION")
        print("   Following @completed-data.mdc rule")
        print("   NO AI • Template-Based • Ultra-Fast")
        print("=" * 60)
        
        # Initialize transformer with NO AI
        transformer = SteeleDataTransformer(use_ai=False)
        
        # Process complete pipeline using templates only
        final_df = transformer.process_complete_pipeline_no_ai()
        
        # Save results with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = f"data/transformed/steele_transformed_{timestamp}.csv"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save the results
        final_df.to_csv(output_file, index=False)
        print(f"💾 Results saved to: {output_file}")
        
        # Display summary
        print("")
        print("=" * 60)
        print("✅ TRANSFORMATION COMPLETE")
        print(f"   Products processed: {len(final_df)}")
        print(f"   Processing method: Template-based (NO AI)")
        print(f"   Golden validated: {len(final_df[final_df.get('Tags', '') != ''])} products")
        print(f"   Output file: {output_file}")
        print("=" * 60)
        
        return output_file

    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()
