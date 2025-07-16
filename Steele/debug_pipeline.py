#!/usr/bin/env python3
"""
Debug Pipeline: Complete pipeline orchestrator
Runs all stages in sequence and provides comprehensive debugging
"""

import os
import pandas as pd
import time
from datetime import datetime
import subprocess
import sys

def run_stage(stage_name, script_name, description):
    """Run a single stage and capture results"""
    print(f"\n🚀 RUNNING {stage_name}")
    print("=" * 60)
    print(f"📋 {description}")
    print(f"🔧 Script: {script_name}")
    
    start_time = time.time()
    
    try:
        # Run the script
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, timeout=300)
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ {stage_name} completed successfully in {duration:.1f}s")
            if result.stdout:
                print("📤 Output:")
                print(result.stdout)
            return True, duration, result.stdout
        else:
            print(f"❌ {stage_name} failed with return code {result.returncode}")
            if result.stderr:
                print("📤 Error:")
                print(result.stderr)
            return False, duration, result.stderr
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {stage_name} timed out after 5 minutes")
        return False, 300, "Timeout"
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"❌ {stage_name} failed with exception: {str(e)}")
        return False, duration, str(e)

def check_prerequisites():
    """Check if all prerequisites are met"""
    print("🔍 CHECKING PREREQUISITES")
    print("=" * 60)
    
    checks = []
    
    # Check API key
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        print("✅ OPENAI_API_KEY found")
        checks.append(True)
    else:
        print("❌ OPENAI_API_KEY not found")
        checks.append(False)
    
    # Check test data
    test_data_path = 'data/results/debug_test_input.csv'
    if os.path.exists(test_data_path):
        print("✅ Test data found")
        checks.append(True)
    else:
        print("❌ Test data not found")
        checks.append(False)
    
    # Check golden master
    golden_path = '../shared/data/master_ultimate_golden.csv'
    if os.path.exists(golden_path):
        print("✅ Golden master found")
        checks.append(True)
    else:
        print("❌ Golden master not found")
        checks.append(False)
    
    # Check utils modules
    utils_files = [
        'utils/ai_extraction.py',
        'utils/golden_master_tag_generator.py'
    ]
    
    for file_path in utils_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} found")
            checks.append(True)
        else:
            print(f"❌ {file_path} not found")
            checks.append(False)
    
    all_good = all(checks)
    print(f"\n📋 Prerequisites: {'✅ All good' if all_good else '❌ Issues found'}")
    return all_good

def debug_full_pipeline():
    """Run the complete debugging pipeline"""
    
    print("🔍 STEELE TWO-PASS AI DEBUGGING PIPELINE")
    print("=" * 60)
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please fix the issues above.")
        return
    
    # Define pipeline stages
    stages = [
        {
            'name': 'STAGE 1: PASS 1 AI EXTRACTION',
            'script': 'debug_pass1.py',
            'description': 'Extract initial vehicle information using AI',
            'output_file': 'data/results/debug_pass1_results.csv'
        },
        {
            'name': 'STAGE 2: PASS 2 REFINEMENT',
            'script': 'debug_pass2.py',
            'description': 'Refine vehicle information using golden master',
            'output_file': 'data/results/debug_pass2_results.csv'
        },
        {
            'name': 'STAGE 3: GOLDEN MASTER LOOKUP',
            'script': 'debug_golden_master.py',
            'description': 'Test golden master lookups and validation',
            'output_file': 'data/results/debug_golden_master_results.csv'
        },
        {
            'name': 'STAGE 4: TAG GENERATION',
            'script': 'debug_tags.py',
            'description': 'Generate final vehicle-specific tags',
            'output_file': 'data/results/debug_tags_results.csv'
        }
    ]
    
    # Run each stage
    results = []
    total_start = time.time()
    
    for stage in stages:
        success, duration, output = run_stage(
            stage['name'], 
            stage['script'], 
            stage['description']
        )
        
        results.append({
            'Stage': stage['name'],
            'Script': stage['script'],
            'Success': success,
            'Duration': duration,
            'Output_File': stage['output_file'],
            'File_Exists': os.path.exists(stage['output_file']) if success else False
        })
        
        if not success:
            print(f"\n⚠️  {stage['name']} failed. You can run it individually with:")
            print(f"   python {stage['script']}")
            print(f"\n🛑 Pipeline stopped at {stage['name']}")
            break
    
    total_end = time.time()
    total_duration = total_end - total_start
    
    # Generate summary
    print(f"\n📋 PIPELINE SUMMARY")
    print("=" * 60)
    print(f"🕐 Total time: {total_duration:.1f}s")
    print(f"✅ Successful stages: {sum(1 for r in results if r['Success'])}/{len(results)}")
    
    for result in results:
        status = "✅" if result['Success'] else "❌"
        file_status = "📁" if result['File_Exists'] else "📄"
        print(f"   {status} {result['Stage']}: {result['Duration']:.1f}s {file_status}")
    
    # Save pipeline results
    results_df = pd.DataFrame(results)
    pipeline_output = 'data/results/debug_pipeline_summary.csv'
    os.makedirs('data/results', exist_ok=True)
    results_df.to_csv(pipeline_output, index=False)
    
    print(f"\n📊 Pipeline summary saved to: {pipeline_output}")
    
    # Show final results if all stages succeeded
    if all(r['Success'] for r in results):
        print(f"\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"📋 Final Results:")
        
        # Show key metrics from each stage
        try:
            # Pass 1 results
            pass1_df = pd.read_csv('data/results/debug_pass1_results.csv')
            print(f"   Pass 1: {len(pass1_df)} products processed")
            
            # Pass 2 results
            pass2_df = pd.read_csv('data/results/debug_pass2_results.csv')
            valid_makes = pass2_df['Make_In_Golden_Master'].sum()
            print(f"   Pass 2: {valid_makes}/{len(pass2_df)} products have valid makes")
            
            # Golden master results
            golden_df = pd.read_csv('data/results/debug_golden_master_results.csv')
            products_with_tags = len(golden_df[golden_df['Tags_Count'] > 0])
            print(f"   Golden Master: {products_with_tags}/{len(golden_df)} products generated tags")
            
            # Tag results
            tags_df = pd.read_csv('data/results/debug_tags_results.csv')
            successful_tags = len(tags_df[tags_df['Success'] == True])
            print(f"   Tags: {successful_tags}/{len(tags_df)} scenarios successful")
            
        except Exception as e:
            print(f"   📊 Could not load final metrics: {str(e)}")
    
    print(f"\n🔧 INDIVIDUAL STAGE COMMANDS:")
    for stage in stages:
        print(f"   python {stage['script']}")
    
    print(f"\n📁 OUTPUT FILES:")
    for stage in stages:
        exists = "✅" if os.path.exists(stage['output_file']) else "❌"
        print(f"   {exists} {stage['output_file']}")


if __name__ == "__main__":
    debug_full_pipeline()