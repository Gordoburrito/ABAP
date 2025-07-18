#!/usr/bin/env python3
"""
Debug script for analyzing JSON-L batch results and CSV integration issues.
Shows where matches are not being retrieved properly and displays patterns.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import sys

def analyze_jsonl_file(jsonl_path: str) -> Dict:
    """
    Analyze a JSON-L batch results file to understand match patterns.
    
    Args:
        jsonl_path: Path to the JSON-L file
        
    Returns:
        Dictionary with analysis results
    """
    print(f"🔍 Analyzing JSON-L file: {jsonl_path}")
    
    analysis = {
        'total_tasks': 0,
        'successful_tasks': 0,
        'failed_tasks': 0,
        'model_match_tasks': 0,
        'make_match_tasks': 0,
        'empty_car_ids_count': 0,
        'successful_matches': [],
        'failed_matches': [],
        'empty_matches': [],
        'task_types': {},
        'patterns': {}
    }
    
    try:
        with open(jsonl_path, 'r') as file:
            for line_number, line in enumerate(file, 1):
                try:
                    result = json.loads(line.strip())
                    analysis['total_tasks'] += 1
                    
                    task_id = result.get("custom_id", f"unknown_{line_number}")
                    
                    # Check for errors
                    if result.get("error") is not None:
                        analysis['failed_tasks'] += 1
                        analysis['failed_matches'].append({
                            'task_id': task_id,
                            'error': result['error'],
                            'line': line_number
                        })
                        continue
                    
                    # Check response structure
                    if "response" not in result or result["response"]["status_code"] != 200:
                        analysis['failed_tasks'] += 1
                        analysis['failed_matches'].append({
                            'task_id': task_id,
                            'error': f"Bad response: {result.get('response', {}).get('status_code', 'no_response')}",
                            'line': line_number
                        })
                        continue
                    
                    analysis['successful_tasks'] += 1
                    
                    # Extract response content
                    try:
                        response_content = result["response"]["body"]["choices"][0]["message"]["content"]
                    except (KeyError, IndexError) as e:
                        analysis['failed_tasks'] += 1
                        analysis['failed_matches'].append({
                            'task_id': task_id,
                            'error': f"Invalid response structure: {e}",
                            'line': line_number
                        })
                        continue
                    
                    # Categorize by task type
                    if task_id.startswith("model_match"):
                        analysis['model_match_tasks'] += 1
                        task_type = 'model_match'
                    elif task_id.startswith("make_match"):
                        analysis['make_match_tasks'] += 1
                        task_type = 'make_match'
                    else:
                        task_type = 'unknown'
                    
                    analysis['task_types'][task_type] = analysis['task_types'].get(task_type, 0) + 1
                    
                    # Parse the response content based on task type
                    if task_type == 'model_match':
                        try:
                            response_data = json.loads(response_content)
                            selected_car_ids = response_data.get('selected_car_ids', [])
                            confidence = response_data.get('confidence', 0.0)
                            
                            if len(selected_car_ids) == 0:
                                analysis['empty_car_ids_count'] += 1
                                analysis['empty_matches'].append({
                                    'task_id': task_id,
                                    'confidence': confidence,
                                    'response': response_content,
                                    'line': line_number
                                })
                            else:
                                analysis['successful_matches'].append({
                                    'task_id': task_id,
                                    'car_ids': selected_car_ids,
                                    'confidence': confidence,
                                    'line': line_number
                                })
                        except json.JSONDecodeError as e:
                            analysis['failed_matches'].append({
                                'task_id': task_id,
                                'error': f"JSON decode error: {e}",
                                'response': response_content,
                                'line': line_number
                            })
                    
                    elif task_type == 'make_match':
                        make_result = response_content.strip()
                        if make_result == "NO_MATCH":
                            analysis['empty_matches'].append({
                                'task_id': task_id,
                                'response': make_result,
                                'line': line_number
                            })
                        else:
                            analysis['successful_matches'].append({
                                'task_id': task_id,
                                'corrected_make': make_result,
                                'line': line_number
                            })
                    
                except json.JSONDecodeError as e:
                    analysis['failed_tasks'] += 1
                    analysis['failed_matches'].append({
                        'task_id': f"line_{line_number}",
                        'error': f"JSON decode error: {e}",
                        'line': line_number
                    })
    
    except FileNotFoundError:
        print(f"❌ File not found: {jsonl_path}")
        return analysis
    except Exception as e:
        print(f"❌ Error analyzing file: {e}")
        return analysis
    
    return analysis

def show_patterns_and_debug_info(analysis: Dict):
    """
    Display analysis results with patterns and debugging information.
    
    Args:
        analysis: Analysis results from analyze_jsonl_file
    """
    print("\n" + "="*60)
    print("📊 BATCH RESULTS ANALYSIS")
    print("="*60)
    
    print(f"Total Tasks: {analysis['total_tasks']}")
    print(f"Successful Tasks: {analysis['successful_tasks']}")
    print(f"Failed Tasks: {analysis['failed_tasks']}")
    print(f"Model Match Tasks: {analysis['model_match_tasks']}")
    print(f"Make Match Tasks: {analysis['make_match_tasks']}")
    print(f"Empty Car IDs Count: {analysis['empty_car_ids_count']}")
    
    success_rate = (analysis['successful_tasks'] / analysis['total_tasks'] * 100) if analysis['total_tasks'] > 0 else 0
    print(f"Success Rate: {success_rate:.1f}%")
    
    # Show task type distribution
    print(f"\n📋 Task Type Distribution:")
    for task_type, count in analysis['task_types'].items():
        percentage = (count / analysis['total_tasks'] * 100) if analysis['total_tasks'] > 0 else 0
        print(f"  {task_type}: {count} ({percentage:.1f}%)")
    
    # Show examples of successful matches
    print(f"\n✅ SUCCESSFUL MATCHES (showing first 5):")
    for i, match in enumerate(analysis['successful_matches'][:5]):
        print(f"  {i+1}. Task: {match['task_id']}")
        if 'car_ids' in match:
            print(f"     Car IDs: {match['car_ids']}")
            print(f"     Confidence: {match.get('confidence', 'N/A')}")
        elif 'corrected_make' in match:
            print(f"     Corrected Make: {match['corrected_make']}")
        print()
    
    # Show examples of empty matches (the problem!)
    print(f"\n⚠️  EMPTY MATCHES (showing first 10 - THIS IS THE ISSUE):")
    for i, match in enumerate(analysis['empty_matches'][:10]):
        print(f"  {i+1}. Task: {match['task_id']}")
        print(f"     Response: {match.get('response', 'N/A')}")
        if 'confidence' in match:
            print(f"     Confidence: {match['confidence']}")
        print(f"     Line: {match['line']}")
        print()
    
    # Show failed matches
    if analysis['failed_matches']:
        print(f"\n❌ FAILED MATCHES (showing first 5):")
        for i, match in enumerate(analysis['failed_matches'][:5]):
            print(f"  {i+1}. Task: {match['task_id']}")
            print(f"     Error: {match['error']}")
            print(f"     Line: {match['line']}")
            print()

def extract_task_patterns(analysis: Dict) -> Dict:
    """
    Extract patterns from task IDs to understand what the AI was supposed to match.
    
    Args:
        analysis: Analysis results
        
    Returns:
        Dictionary with pattern analysis
    """
    patterns = {
        'year_distribution': {},
        'make_distribution': {},
        'model_patterns': {},
        'empty_vs_success_by_year': {},
        'empty_vs_success_by_make': {}
    }
    
    all_matches = analysis['successful_matches'] + analysis['empty_matches'] + analysis['failed_matches']
    
    for match in all_matches:
        task_id = match['task_id']
        
        # Parse task_id patterns like "model_match_1912_Buick_Model 24"
        if task_id.startswith("model_match_"):
            parts = task_id.replace("model_match_", "").split("_", 2)
            if len(parts) >= 3:
                year = parts[0]
                make = parts[1]
                model = parts[2]
                
                # Track distributions
                patterns['year_distribution'][year] = patterns['year_distribution'].get(year, 0) + 1
                patterns['make_distribution'][make] = patterns['make_distribution'].get(make, 0) + 1
                patterns['model_patterns'][f"{make}_{model}"] = patterns['model_patterns'].get(f"{make}_{model}", 0) + 1
                
                # Track success vs empty by year/make
                is_successful = match in analysis['successful_matches']
                is_empty = match in analysis['empty_matches']
                
                if year not in patterns['empty_vs_success_by_year']:
                    patterns['empty_vs_success_by_year'][year] = {'success': 0, 'empty': 0, 'failed': 0}
                
                if make not in patterns['empty_vs_success_by_make']:
                    patterns['empty_vs_success_by_make'][make] = {'success': 0, 'empty': 0, 'failed': 0}
                
                if is_successful:
                    patterns['empty_vs_success_by_year'][year]['success'] += 1
                    patterns['empty_vs_success_by_make'][make]['success'] += 1
                elif is_empty:
                    patterns['empty_vs_success_by_year'][year]['empty'] += 1
                    patterns['empty_vs_success_by_make'][make]['empty'] += 1
                else:
                    patterns['empty_vs_success_by_year'][year]['failed'] += 1
                    patterns['empty_vs_success_by_make'][make]['failed'] += 1
    
    return patterns

def show_pattern_analysis(patterns: Dict):
    """
    Display pattern analysis to understand why matches are failing.
    
    Args:
        patterns: Pattern analysis results
    """
    print("\n" + "="*60)
    print("🔍 PATTERN ANALYSIS - WHY MATCHES ARE FAILING")
    print("="*60)
    
    # Year analysis
    print("\n📅 Success/Empty Rate by Year (top 10):")
    year_stats = []
    for year, counts in patterns['empty_vs_success_by_year'].items():
        total = counts['success'] + counts['empty'] + counts['failed']
        empty_rate = (counts['empty'] / total * 100) if total > 0 else 0
        success_rate = (counts['success'] / total * 100) if total > 0 else 0
        year_stats.append((year, total, success_rate, empty_rate, counts))
    
    # Sort by total count
    year_stats.sort(key=lambda x: x[1], reverse=True)
    
    for year, total, success_rate, empty_rate, counts in year_stats[:10]:
        print(f"  {year}: {total} tasks, {success_rate:.1f}% success, {empty_rate:.1f}% empty")
        print(f"       Success: {counts['success']}, Empty: {counts['empty']}, Failed: {counts['failed']}")
    
    # Make analysis
    print(f"\n🚗 Success/Empty Rate by Make (top 10):")
    make_stats = []
    for make, counts in patterns['empty_vs_success_by_make'].items():
        total = counts['success'] + counts['empty'] + counts['failed']
        empty_rate = (counts['empty'] / total * 100) if total > 0 else 0
        success_rate = (counts['success'] / total * 100) if total > 0 else 0
        make_stats.append((make, total, success_rate, empty_rate, counts))
    
    # Sort by total count
    make_stats.sort(key=lambda x: x[1], reverse=True)
    
    for make, total, success_rate, empty_rate, counts in make_stats[:10]:
        print(f"  {make}: {total} tasks, {success_rate:.1f}% success, {empty_rate:.1f}% empty")
        print(f"        Success: {counts['success']}, Empty: {counts['empty']}, Failed: {counts['failed']}")

def check_csv_integration_debug(steele_dir: Path):
    """
    Check if there are CSV files and see if matches are being applied correctly.
    
    Args:
        steele_dir: Path to Steele directory
    """
    print("\n" + "="*60)
    print("🗂️  CSV INTEGRATION DEBUG")
    print("="*60)
    
    # Look for processed/validated CSV files
    csv_files = []
    for pattern in ["data/**/*.csv", "data/processed/*.csv", "data/transformed/*.csv"]:
        csv_files.extend(list(steele_dir.glob(pattern)))
    
    print(f"Found {len(csv_files)} CSV files:")
    for csv_file in csv_files[:10]:  # Show first 10
        relative_path = csv_file.relative_to(steele_dir)
        print(f"  📄 {relative_path}")
        
        try:
            df = pd.read_csv(csv_file, nrows=5)  # Read first 5 rows
            print(f"      Columns: {list(df.columns)}")
            print(f"      Rows: {len(pd.read_csv(csv_file))}")
            
            # Check for car_ids or Tags columns
            if 'Tags' in df.columns:
                non_empty_tags = len(df[df['Tags'].notna() & (df['Tags'] != '')])
                print(f"      Non-empty Tags: {non_empty_tags}/5 (showing)")
            
            if 'car_ids' in df.columns:
                non_empty_car_ids = len(df[df['car_ids'].notna() & (df['car_ids'] != '')])
                print(f"      Non-empty car_ids: {non_empty_car_ids}/5 (showing)")
                
        except Exception as e:
            print(f"      Error reading: {e}")
        print()

def main():
    """Main debugging function."""
    steele_dir = Path("/Users/gordonlewis/ABAP/Steele")
    
    print("🔧 BATCH MATCH DEBUGGING TOOL")
    print("=" * 60)
    print("This tool analyzes JSON-L batch results and debugs CSV integration issues.")
    print()
    
    # Find the most recent batch results file
    batch_dir = steele_dir / "data" / "batch"
    jsonl_files = list(batch_dir.glob("batch_results_*.jsonl"))
    
    if not jsonl_files:
        print("❌ No batch results files found in data/batch/")
        return
    
    # Sort by modification time (most recent first)
    jsonl_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    print(f"📁 Found {len(jsonl_files)} batch result files:")
    for i, file in enumerate(jsonl_files[:5]):  # Show first 5
        print(f"  {i+1}. {file.name}")
    
    # Use the most recent file
    latest_file = jsonl_files[0]
    print(f"\n🎯 Analyzing most recent file: {latest_file.name}")
    
    # Analyze the JSON-L file
    analysis = analyze_jsonl_file(str(latest_file))
    
    # Show results
    show_patterns_and_debug_info(analysis)
    
    # Extract and show patterns
    patterns = extract_task_patterns(analysis)
    show_pattern_analysis(patterns)
    
    # Check CSV integration
    check_csv_integration_debug(steele_dir)
    
    # Provide recommendations
    print("\n" + "="*60)
    print("💡 DEBUGGING RECOMMENDATIONS")
    print("="*60)
    
    if analysis['empty_car_ids_count'] > 0:
        print("🔍 ISSUE FOUND: Many model matches returning empty car_ids")
        print("   This suggests:")
        print("   1. AI prompts may need improvement")
        print("   2. Golden dataset may not have matching models")
        print("   3. Model naming conventions are too different")
        print()
        print("🛠️  FIXES TO TRY:")
        print("   1. Examine the AI prompt in batch_ai_vehicle_matcher.py:_create_model_matching_prompt")
        print("   2. Check if the year_make_matches DataFrame is being passed correctly")
        print("   3. Verify the golden dataset has the expected models for these years/makes")
        print("   4. Consider lowering confidence thresholds or improving fuzzy matching")
    
    if analysis['failed_tasks'] > 0:
        print(f"⚠️  {analysis['failed_tasks']} tasks failed completely")
        print("   Check the failed_matches list above for specific error messages")
    
    print(f"\n📈 Overall statistics:")
    print(f"   • Success rate: {(analysis['successful_tasks'] / analysis['total_tasks'] * 100):.1f}%")
    print(f"   • Empty matches: {analysis['empty_car_ids_count']} ({(analysis['empty_car_ids_count'] / analysis['total_tasks'] * 100):.1f}%)")
    print(f"   • Failed tasks: {analysis['failed_tasks']} ({(analysis['failed_tasks'] / analysis['total_tasks'] * 100):.1f}%)")

if __name__ == "__main__":
    main()