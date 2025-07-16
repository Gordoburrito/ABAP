from utils.steele_data_transformer import SteeleDataTransformer
import time
import os
from pathlib import Path

def main():
    """
    Main entry point for Steele data transformation.
    Following @completed-data.mdc rule: NO AI usage, template-based processing only.
    """
    try:
        # Get the directory where this script is located (Steele directory)
        steele_dir = Path(__file__).parent

        print("=" * 60)
        print("🚀 STEELE DATA TRANSFORMATION")
        print("   Following @completed-data.mdc rule")
        print("   NO AI • Template-Based • Ultra-Fast")
        print("=" * 60)

        # Initialize transformer with NO AI
        transformer = SteeleDataTransformer(use_ai=False)

        # For testing with small sample:
        final_df = transformer.process_complete_pipeline_no_ai("data/samples/steele_test_1000.csv")

        # # Or for full processing:
        # final_df = transformer.process_complete_pipeline_no_ai("data/processed/steele_processed_complete.csv")

        # Save results with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = steele_dir / "data" / "transformed" / f"steele_transformed_{timestamp}.csv"
        os.makedirs(output_file.parent, exist_ok=True)

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

        return str(output_file)

    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()
