from utils.steele_data_transformer import SteeleDataTransformer
import time
import os
from pathlib import Path

def main():
    """
    Main entry point for Steele data transformation.
    Uses AI for accurate vehicle tag generation to match master_ultimate_golden format.
    """
    try:
        # Get the directory where this script is located (Steele directory)
        steele_dir = Path(__file__).parent
        
        print("=" * 60)
        print("🚀 STEELE DATA TRANSFORMATION")
        print("   AI-Powered Vehicle Tag Generation")
        print("   Template-Based Processing + Accurate Tags")
        print("=" * 60)
        
        # Initialize transformer with AI for accurate vehicle tags
        transformer = SteeleDataTransformer(use_ai=True)
        
        # Process complete pipeline using AI for vehicle tag generation
        final_df = transformer.process_complete_pipeline_no_ai()
        
        # Save results with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = steele_dir / "data" / "transformed" / f"steele_ai_tags_{timestamp}.csv"
        os.makedirs(output_file.parent, exist_ok=True)
        
        # Save the results
        final_df.to_csv(output_file, index=False)
        print(f"💾 Results saved to: {output_file}")
        
        # Display summary
        print("")
        print("=" * 60)
        print("✅ TRANSFORMATION COMPLETE")
        print(f"   Products processed: {len(final_df)}")
        print(f"   Processing method: Template-based + AI vehicle tags")
        print(f"   AI-mapped tags: {len(final_df[final_df.get('Tags', '') != ''])} products")
        print(f"   Output file: {output_file}")
        print("=" * 60)
        
        return str(output_file)

    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()
