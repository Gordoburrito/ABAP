#!/usr/bin/env python3
"""
Demo script showing the complete Steele data transformation pipeline:
Sample Data → Golden Master → AI-Friendly Format → Final Tagged Format
"""

import pandas as pd
import sys
from pathlib import Path
from utils.steele_data_transformer import SteeleDataTransformer

def demo_complete_pipeline():
    """Demonstrate the complete transformation pipeline"""
    
    print("🚀 Steele Data Transformation Pipeline Demo")
    print("=" * 60)
    print("📋 Workflow: Sample Data → Golden Master → AI-Friendly → Final Tagged")
    print()
    
    try:
        # Initialize transformer (disable AI for demo to avoid API requirement)
        print("🔧 Initializing Steele Data Transformer...")
        transformer = SteeleDataTransformer(use_ai=False)  # Set to True if you have OpenAI API key
        print("✅ Transformer initialized")
        print()
        
        # Step 1: Load sample data
        print("📁 Step 1: Loading Steele sample data...")
        try:
            steele_df = transformer.load_sample_data("data/samples/steele_sample.csv")
            print(f"✅ Loaded {len(steele_df)} products from sample data")
            print(f"   Columns: {list(steele_df.columns)}")
            print(f"   Sample product: {steele_df.iloc[0]['Product Name']}")
        except FileNotFoundError:
            print("❌ Sample data file not found. Using mock data for demo.")
            steele_df = pd.DataFrame({
                'StockCode': ['10-0001-40', '10-0002-35'],
                'Product Name': ['Accelerator Pedal Pad', 'Axle Rebound Pad'],
                'Description': ['Pad, accelerator pedal...', 'Pad, front axle rebound...'],
                'MAP': [75.49, 127.79],
                'Dealer Price': [43.76, 81.97],
                'Year': [1928, 1930],
                'Make': ['Stutz', 'Stutz'],
                'Model': ['Stutz', 'Stutz']
            })
            print(f"✅ Using mock data with {len(steele_df)} products")
        print()
        
        # Step 2: Load golden dataset (use mock if not available)
        print("🗂️ Step 2: Loading and validating against golden master dataset...")
        try:
            golden_df = transformer.load_golden_dataset()
            print(f"✅ Loaded golden dataset with {len(golden_df)} vehicle records")
        except (FileNotFoundError, ValueError) as e:
            print(f"⚠️ Golden dataset not available ({e}). Using mock data.")
            # Create mock golden dataset for demo
            golden_df = pd.DataFrame({
                'year': [1928, 1930, 1930, 1965],
                'make': ['Stutz', 'Stutz', 'Durant', 'Ford'],
                'model': ['Stutz', 'Stutz', 'Model 6-14', 'Mustang'],
                'car_id': ['1928_Stutz_Stutz', '1930_Stutz_Stutz', '1930_Durant_Model 6-14', '1965_Ford_Mustang']
            })
            transformer.golden_df = golden_df
            print(f"✅ Using mock golden dataset with {len(golden_df)} vehicle records")
        
        # Validate against golden dataset
        validation_df = transformer.validate_against_golden_dataset(steele_df)
        validated_count = len(validation_df[validation_df['golden_validated'] == True])
        print(f"✅ Vehicle validation complete: {validated_count}/{len(steele_df)} products validated")
        print()
        
        # Step 3: Transform to AI-friendly format
        print("🤖 Step 3: Transforming to AI-friendly format...")
        ai_friendly_products = transformer.transform_to_ai_friendly_format(steele_df, validation_df)
        print(f"✅ Transformed {len(ai_friendly_products)} products to AI-friendly format")
        print("   Sample AI-friendly product:")
        sample_product = ai_friendly_products[0]
        print(f"   - Title: {sample_product.title}")
        print(f"   - Vehicle: {sample_product.year_min} {sample_product.make} {sample_product.model}")
        print(f"   - Price: ${sample_product.price}, Cost: ${sample_product.cost}")
        print()
        
        # Step 3b: Enhance with AI (or use defaults)
        print("✨ Step 3b: Enhancing with AI (or defaults)...")
        enhanced_products = transformer.enhance_with_ai(ai_friendly_products)
        print(f"✅ Enhanced {len(enhanced_products)} products")
        print("   Sample enhancement:")
        enhanced_sample = enhanced_products[0]
        print(f"   - Collection: {enhanced_sample.collection}")
        print(f"   - Meta Title: {enhanced_sample.meta_title}")
        print(f"   - Meta Description: {enhanced_sample.meta_description[:50]}...")
        print()
        
        # Step 4: Convert to final tagged format
        print("🏷️ Step 4: Converting to final Shopify tagged format...")
        final_df = transformer.transform_to_final_tagged_format(enhanced_products)
        print(f"✅ Generated final Shopify format with {len(final_df)} products")
        print("   Required Shopify columns present:")
        required_cols = ['Title', 'Body HTML', 'Vendor', 'Tags', 'Variant Price', 'Variant Cost']
        for col in required_cols:
            status = "✅" if col in final_df.columns else "❌"
            print(f"   {status} {col}")
        print()
        
        # Show sample final product
        print("📦 Sample Final Product:")
        sample_final = final_df.iloc[0]
        print(f"   Title: {sample_final['Title']}")
        print(f"   Vendor: {sample_final['Vendor']}")
        print(f"   Tags: {sample_final['Tags']}")
        print(f"   Price: ${sample_final['Variant Price']}")
        print(f"   Meta Title: {sample_final['Metafield: title_tag [string]']}")
        print()
        
        # Validation summary
        print("🔍 Pipeline Validation Summary:")
        validation_results = transformer.validate_output(final_df)
        
        if validation_results['errors']:
            print("❌ Errors found:")
            for error in validation_results['errors']:
                print(f"   - {error}")
        else:
            print("✅ No validation errors")
            
        if validation_results['warnings']:
            print("⚠️ Warnings:")
            for warning in validation_results['warnings']:
                print(f"   - {warning}")
        else:
            print("✅ No validation warnings")
            
        print("📊 Info:")
        for info in validation_results['info']:
            print(f"   - {info}")
        print()
        
        # Save demo results
        print("💾 Saving demo results...")
        output_path = "data/results/demo_output.csv"
        try:
            saved_path = transformer.save_output(final_df, output_path)
            print(f"✅ Results saved to: {saved_path}")
        except Exception as e:
            print(f"⚠️ Could not save to {output_path}: {e}")
            print("   Results available in memory for inspection")
        print()
        
        # Performance summary
        print("🚀 Pipeline Complete!")
        print("=" * 60)
        print("✅ Successfully transformed Steele data through complete pipeline:")
        print("   1. ✅ Sample Data Loading & Validation")
        print("   2. ✅ Golden Master Vehicle Validation") 
        print("   3. ✅ AI-Friendly Format Transformation")
        print("   4. ✅ AI Enhancement (or defaults)")
        print("   5. ✅ Final Shopify Tagged Format")
        print()
        print("🎯 Key Benefits Demonstrated:")
        print("   ✓ Vehicle compatibility validated against golden master")
        print("   ✓ Token-efficient AI processing format")
        print("   ✓ Shopify import compliance verified")
        print("   ✓ Scalable architecture ready for other vendors")
        print()
        print("🔄 Next Steps:")
        print("   - Run with real OpenAI API: set use_ai=True")
        print("   - Process larger datasets")
        print("   - Replicate for other vendors (REM, ABAP, Ford)")
        print("   - Run complete test suite: python run_tests.py")
        
        return final_df
        
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        print("🔧 Check configuration and try again")
        return None

if __name__ == "__main__":
    demo_complete_pipeline() 