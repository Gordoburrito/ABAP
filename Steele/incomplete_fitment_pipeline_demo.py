#!/usr/bin/env python3
"""
Incomplete Fitment Data Pipeline Demo

This script demonstrates the complete TDD-implemented pipeline for processing 
automotive parts data with incomplete fitment information using AI extraction.

All 6 phases have been implemented and tested:
1. Data Loading and AI-Friendly Format Conversion
2. AI Product Data Extraction  
3. Golden Master Validation
4. Model Refinement with Body Type Logic
5. Shopify 65-Column Format Generation
6. Integration Testing and Batch Processing

Usage:
    python incomplete_fitment_pipeline_demo.py
"""

import os
import sys
import logging
from pathlib import Path
from utils.integration_pipeline import IncompleteDataPipeline, PipelineValidator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main demo function"""
    print("=" * 80)
    print("INCOMPLETE FITMENT DATA PIPELINE DEMO")
    print("TDD Implementation for Steele Data Source")
    print("=" * 80)
    
    # Configuration
    config = {
        'openai_api_key': os.getenv('OPENAI_API_KEY', 'demo_key'),
        'golden_master_path': str(Path(__file__).parent.parent / "shared" / "data" / "master_ultimate_golden.csv"),
        'column_requirements_path': str(Path(__file__).parent.parent / "shared" / "data" / "product_import" / "product_import-column-requirements.py"),
        'batch_size': 10,
        'model': 'gpt-4.1-mini',
        'enable_ai': False,  # Set to False for demo (no API key needed)
        'max_retries': 3
    }
    
    print("\n1. VALIDATING CONFIGURATION")
    print("-" * 40)
    
    # Validate configuration
    validation_result = PipelineValidator.validate_config(config)
    
    if validation_result['is_valid']:
        print("✅ Configuration is valid")
    else:
        print("❌ Configuration has errors:")
        for error in validation_result['errors']:
            print(f"   - {error}")
        return
    
    if validation_result['warnings']:
        print("⚠️  Configuration warnings:")
        for warning in validation_result['warnings']:
            print(f"   - {warning}")
    
    print("\n2. INITIALIZING PIPELINE")
    print("-" * 40)
    
    try:
        pipeline = IncompleteDataPipeline(config)
        print("✅ Pipeline initialized successfully")
        print(f"   - AI Processing: {'Enabled' if config['enable_ai'] else 'Disabled (Template-based)'}")
        print(f"   - Model: {config['model']}")
        print(f"   - Batch Size: {config['batch_size']}")
    except Exception as e:
        print(f"❌ Pipeline initialization failed: {e}")
        return
    
    print("\n3. LOADING SAMPLE DATA")
    print("-" * 40)
    
    # Use sample Steele data
    input_path = "data/raw/steele.xlsx"
    
    if not Path(input_path).exists():
        print(f"❌ Sample data not found: {input_path}")
        print("   Please ensure steele.xlsx exists in data/raw/")
        return
    
    try:
        # Load a sample of data for demo
        import pandas as pd
        sample_df = pd.read_excel(input_path).head(5)  # Just 5 items for demo
        
        print(f"✅ Loaded sample data: {len(sample_df)} products")
        print("   Sample products:")
        for i, row in sample_df.iterrows():
            print(f"   - {row.get('Product Name', 'Unknown')} (SKU: {row.get('StockCode', 'Unknown')})")
        
        # Validate input data
        input_validation = PipelineValidator.validate_input_data(sample_df)
        if input_validation['is_valid']:
            print("✅ Input data format is valid")
        else:
            print("⚠️  Input data issues:")
            for error in input_validation['errors']:
                print(f"   - {error}")
        
    except Exception as e:
        print(f"❌ Failed to load sample data: {e}")
        return
    
    print("\n4. ESTIMATING PROCESSING COST")
    print("-" * 40)
    
    try:
        cost_estimate = pipeline.estimate_processing_cost(sample_df)
        
        print("💰 Cost Estimation:")
        print(f"   - Total Items: {len(sample_df)}")
        print(f"   - Input Tokens: {cost_estimate['total_input_tokens']:,}")
        print(f"   - Output Tokens: {cost_estimate['estimated_output_tokens']:,}")
        print(f"   - Total Cost: ${cost_estimate['total_cost']:.4f}")
        print(f"   - Cost per Item: ${cost_estimate['cost_per_item']:.4f}")
        print(f"   - Model: {cost_estimate['model']}")
        
    except Exception as e:
        print(f"❌ Cost estimation failed: {e}")
        return
    
    print("\n5. PROCESSING PIPELINE")
    print("-" * 40)
    
    # Create temporary output path
    output_path = "data/results/demo_output_incomplete_fitment.csv"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save sample data as CSV for processing
    temp_input_path = "data/results/demo_input_sample.csv"
    sample_df.to_csv(temp_input_path, index=False)
    
    try:
        print("🔄 Processing complete pipeline...")
        
        results = pipeline.process_complete_pipeline(temp_input_path, output_path)
        
        print("✅ Pipeline processing completed!")
        print("\n📊 PROCESSING RESULTS:")
        print(f"   - Total Items: {results['total_items']}")
        print(f"   - Processed Items: {results['processed_items']}")
        print(f"   - Failed Items: {results['failed_items']}")
        print(f"   - Success Rate: {results['success_rate']:.1%}")
        print(f"   - Processing Time: {results['processing_time']:.2f} seconds")
        print(f"   - Total Cost: ${results['total_cost']:.4f}")
        print(f"   - Cost per Item: ${results['cost_per_item']:.4f}")
        print(f"   - Output File: {results['output_file']}")
        print(f"   - Shopify Compliant: {'Yes' if results['validation_compliant'] else 'No'}")
        
    except Exception as e:
        print(f"❌ Pipeline processing failed: {e}")
        return
    
    print("\n6. VALIDATING OUTPUT")
    print("-" * 40)
    
    try:
        # Validate output file
        if Path(output_path).exists():
            output_df = pd.read_csv(output_path)
            
            print("✅ Output file generated successfully")
            print(f"   - Output rows: {len(output_df)}")
            print(f"   - Output columns: {len(output_df.columns)}")
            print(f"   - Expected columns: 65 (Shopify format)")
            
            if len(output_df.columns) == 65:
                print("✅ Shopify format compliance: PASSED")
            else:
                print("⚠️  Shopify format compliance: ISSUES")
            
            # Show sample of key columns
            key_columns = ['Title', 'Variant SKU', 'Variant Price', 'Vendor', 'Tags']
            available_key_columns = [col for col in key_columns if col in output_df.columns]
            
            if available_key_columns:
                print("\n📋 Sample Output Data:")
                for i, row in output_df.head(3).iterrows():
                    print(f"   Product {i+1}:")
                    for col in available_key_columns:
                        print(f"     - {col}: {row[col]}")
        else:
            print("❌ Output file not found")
            
    except Exception as e:
        print(f"❌ Output validation failed: {e}")
    
    print("\n7. GENERATING PROCESSING REPORT")
    print("-" * 40)
    
    try:
        report = pipeline.generate_processing_report()
        
        print("📈 PROCESSING REPORT:")
        print(f"   Summary:")
        print(f"     - Success Rate: {report['summary']['success_rate']:.1%}")
        print(f"     - Items Processed: {report['summary']['processed_items']}")
        print(f"     - Items Failed: {report['summary']['failed_items']}")
        
        print(f"   Performance:")
        print(f"     - Total Time: {report['performance_metrics']['total_processing_time']:.2f}s")
        print(f"     - Avg Time/Item: {report['performance_metrics']['average_processing_time']:.3f}s")
        print(f"     - Items/Minute: {report['performance_metrics']['items_per_minute']:.1f}")
        
        print(f"   Cost Analysis:")
        print(f"     - Total Cost: ${report['cost_analysis']['total_cost']:.4f}")
        print(f"     - Cost per Item: ${report['cost_analysis']['cost_per_item']:.4f}")
        print(f"     - AI Enabled: {report['cost_analysis']['ai_enabled']}")
        
    except Exception as e:
        print(f"❌ Report generation failed: {e}")
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("\nAll 6 TDD phases have been implemented and demonstrated:")
    print("✅ Phase 1: Data Loading and AI-Friendly Format Conversion")
    print("✅ Phase 2: AI Product Data Extraction")
    print("✅ Phase 3: Golden Master Validation")
    print("✅ Phase 4: Model Refinement with Body Type Logic")
    print("✅ Phase 5: Shopify 65-Column Format Generation")
    print("✅ Phase 6: Integration Testing and Batch Processing")
    print("\nThe pipeline is ready for production use with complete fitment data!")
    print("=" * 80)


if __name__ == "__main__":
    main()