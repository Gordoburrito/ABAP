#!/usr/bin/env python3
"""
Fix script for JSON-L batch match integration with CSV processing.
Analyzes specific failing cases and shows how to fix the integration.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

def load_golden_dataset(file_path: str = None) -> pd.DataFrame:
    """Load the golden master dataset for comparison."""
    if file_path is None:
        # Try shared data path  
        shared_path = Path(__file__).parent.parent / "shared" / "data" / "master_ultimate_golden.csv"
        if shared_path.exists():
            file_path = str(shared_path)
        else:
            raise FileNotFoundError("Golden dataset not found")
    
    try:
        golden_df = pd.read_csv(file_path)
        
        # Standardize column names
        column_mapping = {
            'Year': 'year',
            'Make': 'make', 
            'Model': 'model',
            'Car ID': 'car_id'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in golden_df.columns:
                golden_df = golden_df.rename(columns={old_col: new_col})
        
        return golden_df
    except Exception as e:
        raise ValueError(f"Error loading golden dataset: {str(e)}")

def analyze_failing_model_match(task_id: str, golden_df: pd.DataFrame) -> Dict:
    """
    Analyze why a specific model match is failing by examining the golden dataset.
    
    Args:
        task_id: Task ID like "model_match_1912_Cadillac_Four"
        golden_df: Golden master dataset
        
    Returns:
        Analysis results showing what should have matched
    """
    # Parse task_id
    if not task_id.startswith("model_match_"):
        return {"error": "Not a model match task"}
    
    parts = task_id.replace("model_match_", "").split("_", 2)
    if len(parts) < 3:
        return {"error": "Invalid task_id format"}
    
    year = int(parts[0])
    make = parts[1]
    model = parts[2]
    
    print(f"\n🔍 Analyzing failing task: {task_id}")
    print(f"   Input: {year} {make} {model}")
    
    analysis = {
        "input_year": year,
        "input_make": make,
        "input_model": model,
        "exact_matches": [],
        "year_make_matches": [],
        "year_only_matches": [],
        "make_variations": [],
        "model_variations": [],
        "recommendations": []
    }
    
    # Step 1: Check exact matches
    exact_matches = golden_df[
        (golden_df['year'] == year) &
        (golden_df['make'] == make) &
        (golden_df['model'] == model)
    ]
    
    if len(exact_matches) > 0:
        analysis["exact_matches"] = exact_matches[['year', 'make', 'model', 'car_id']].to_dict('records')
        analysis["recommendations"].append("✅ Exact match exists - this should have worked!")
        return analysis
    
    # Step 2: Check year + make matches
    year_make_matches = golden_df[
        (golden_df['year'] == year) &
        (golden_df['make'] == make)
    ]
    
    if len(year_make_matches) > 0:
        analysis["year_make_matches"] = year_make_matches[['year', 'make', 'model', 'car_id']].to_dict('records')
        
        # Show available models for this year+make
        available_models = year_make_matches['model'].unique().tolist()
        analysis["available_models"] = available_models
        
        print(f"   📋 Available models for {year} {make}:")
        for available_model in available_models[:10]:  # Show first 10
            print(f"      • {available_model}")
        
        # Look for similar models
        similar_models = []
        input_model_lower = model.lower().replace(" ", "").replace("-", "")
        
        for available_model in available_models:
            available_lower = str(available_model).lower().replace(" ", "").replace("-", "")
            
            # Check various similarity patterns
            if input_model_lower in available_lower or available_lower in input_model_lower:
                similar_models.append({
                    "available": available_model,
                    "similarity": "substring_match",
                    "car_ids": year_make_matches[year_make_matches['model'] == available_model]['car_id'].tolist()
                })
            elif input_model_lower.replace("model", "") == available_lower.replace("model", "").replace("series", "").replace("type", ""):
                similar_models.append({
                    "available": available_model,
                    "similarity": "prefix_normalized",
                    "car_ids": year_make_matches[year_make_matches['model'] == available_model]['car_id'].tolist()
                })
        
        analysis["similar_models"] = similar_models
        
        if similar_models:
            analysis["recommendations"].append(f"🔧 Found {len(similar_models)} similar models - AI should have matched these")
            for sim in similar_models[:3]:
                analysis["recommendations"].append(f"   → '{sim['available']}' ({sim['similarity']})")
        else:
            analysis["recommendations"].append("❌ No similar models found - genuine mismatch")
    
    else:
        # Step 3: Check year only matches
        year_only_matches = golden_df[golden_df['year'] == year]
        
        if len(year_only_matches) > 0:
            analysis["year_only_matches"] = year_only_matches[['year', 'make', 'model', 'car_id']].to_dict('records')
            
            # Show available makes for this year
            available_makes = year_only_matches['make'].unique().tolist()
            analysis["available_makes"] = available_makes
            
            print(f"   📋 Available makes for {year}:")
            for available_make in available_makes[:10]:
                print(f"      • {available_make}")
            
            # Look for similar makes
            similar_makes = []
            for available_make in available_makes:
                if make.lower() in available_make.lower() or available_make.lower() in make.lower():
                    similar_makes.append(available_make)
            
            analysis["similar_makes"] = similar_makes
            
            if similar_makes:
                analysis["recommendations"].append(f"🔧 Found similar makes: {similar_makes} - AI make matching should handle this")
            else:
                analysis["recommendations"].append("❌ No similar makes found for this year")
        else:
            analysis["recommendations"].append(f"❌ Year {year} not found in golden dataset")
    
    return analysis

def show_successful_pattern_examples(batch_results_file: str, golden_df: pd.DataFrame, limit: int = 5):
    """
    Show examples of successful matches to understand what works.
    
    Args:
        batch_results_file: Path to JSON-L batch results
        golden_df: Golden master dataset
        limit: Number of examples to show
    """
    print(f"\n✅ SUCCESSFUL MATCH PATTERNS (showing {limit} examples):")
    print("=" * 60)
    
    successful_count = 0
    
    try:
        with open(batch_results_file, 'r') as file:
            for line in file:
                if successful_count >= limit:
                    break
                
                try:
                    result = json.loads(line.strip())
                    task_id = result.get("custom_id", "")
                    
                    if not task_id.startswith("model_match_"):
                        continue
                    
                    # Extract response content
                    try:
                        response_content = result["response"]["body"]["choices"][0]["message"]["content"]
                        response_data = json.loads(response_content)
                        selected_car_ids = response_data.get('selected_car_ids', [])
                        confidence = response_data.get('confidence', 0.0)
                        
                        if len(selected_car_ids) > 0:  # This is a successful match
                            # Parse the task
                            parts = task_id.replace("model_match_", "").split("_", 2)
                            if len(parts) >= 3:
                                year = int(parts[0])
                                make = parts[1]
                                model = parts[2]
                                
                                print(f"{successful_count + 1}. Task: {task_id}")
                                print(f"   Input: {year} {make} {model}")
                                print(f"   Matched Car IDs: {selected_car_ids}")
                                print(f"   Confidence: {confidence}")
                                
                                # Show what models these car_ids correspond to
                                matched_models = []
                                for car_id in selected_car_ids:
                                    matches = golden_df[golden_df['car_id'] == car_id]
                                    if len(matches) > 0:
                                        matched_models.append(matches.iloc[0]['model'])
                                
                                print(f"   Golden Models: {matched_models}")
                                print()
                                
                                successful_count += 1
                    
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
                
                except json.JSONDecodeError:
                    continue
    
    except Exception as e:
        print(f"❌ Error reading batch results: {e}")

def demonstrate_csv_integration_fix(steele_dir: Path):
    """
    Demonstrate how to fix the CSV integration to properly apply batch results.
    
    Args:
        steele_dir: Path to Steele directory
    """
    print(f"\n🔧 CSV INTEGRATION FIX DEMONSTRATION")
    print("=" * 60)
    
    # Look for a sample CSV to demonstrate with
    sample_files = [
        "data/samples/steele_test_1000.csv",
        "data/samples/steele_sample.csv"
    ]
    
    sample_df = None
    sample_file = None
    
    for file_path in sample_files:
        full_path = steele_dir / file_path
        if full_path.exists():
            try:
                sample_df = pd.read_csv(full_path, nrows=10)  # Just first 10 rows for demo
                sample_file = file_path
                break
            except Exception as e:
                print(f"⚠️  Could not read {file_path}: {e}")
    
    if sample_df is None:
        print("❌ No sample CSV files found")
        return
    
    print(f"📄 Using sample file: {sample_file}")
    print(f"📊 Sample data (first 5 rows):")
    print(sample_df[['StockCode', 'Year', 'Make', 'Model']].head())
    
    print(f"\n🔍 Current CSV Integration Issue:")
    print(f"   1. Batch AI processes model matches correctly")
    print(f"   2. AI returns results in JSON-L format")
    print(f"   3. get_model_match_result() extracts the car_ids")
    print(f"   4. ❌ BUT: Empty car_ids arrays are not being handled properly")
    print(f"   5. ❌ CSV ends up with empty Tags columns")
    
    print(f"\n🛠️  Required Fix in batch_steele_data_transformer.py:")
    print(f"""
    In update_validation_with_ai_results() around line 470:
    
    CURRENT CODE:
    ai_matches = self.batch_ai_matcher.get_model_match_result(
        task_id, task_info['year_make_matches']
    )
    
    if len(ai_matches) > 0:
        # AI found matches - this works fine
        updated_row = row.to_dict()
        updated_row.update({{
            'golden_validated': True,
            'golden_matches': len(ai_matches),
            'car_ids': ai_matches['car_id'].unique().tolist(),
            'match_type': 'ai_model_match'
        }})
    else:
        # ❌ PROBLEM: This case needs better debugging
        print(f"⚠️  AI returned empty matches for {{task_id}}")
        print(f"    Input: {{task_info}}")
        print(f"    Available models: {{task_info['year_make_matches']['model'].unique()}}")
    """)

def create_enhanced_debug_script():
    """Create an enhanced version of the batch AI matcher with better debugging."""
    print(f"\n📝 ENHANCED DEBUG VERSION")
    print("=" * 60)
    print("""
To fix the match integration, add this debug code to batch_ai_vehicle_matcher.py 
in the get_model_match_result() method around line 405:

def get_model_match_result(self, task_id: str, year_make_matches: pd.DataFrame) -> pd.DataFrame:
    if task_id not in self.batch_results:
        print(f"❌ No result found for task: {task_id}")
        return pd.DataFrame()
    
    try:
        content = self.batch_results[task_id]["content"]
        
        # Parse JSON response
        try:
            result_data = json.loads(content)
            result = ModelMatchResult(**result_data)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"❌ Failed to parse model match result for {task_id}: {e}")
            print(f"   Raw content: {content}")
            return pd.DataFrame()
        
        print(f"🤖 AI Result for {task_id}:")
        print(f"   Selected car_ids: {result.selected_car_ids}")
        print(f"   Confidence: {result.confidence}")
        
        # 🔧 ADD THIS DEBUG CODE:
        if len(result.selected_car_ids) == 0:
            print(f"   ⚠️  EMPTY RESULT - Debugging...")
            print(f"   Available car_ids in year_make_matches:")
            for idx, row in year_make_matches.iterrows():
                print(f"      • {row['car_id']}: {row['model']}")
            print(f"   This suggests the AI prompt needs improvement or confidence threshold is too high")
        
        # Filter year_make_matches to only selected car_ids
        selected_matches = year_make_matches[year_make_matches['car_id'].isin(result.selected_car_ids)]
        
        if len(selected_matches) == 0 and len(result.selected_car_ids) > 0:
            print(f"⚠️  AI selected car_ids not found in year_make_matches: {result.selected_car_ids}")
            return pd.DataFrame()
        
        if len(selected_matches) == 0:
            print(f"✋ Found 0 matches for {task_id} (AI confidence: {result.confidence})")
            return pd.DataFrame()
        
        print(f"✅ Found {len(selected_matches)} matches for {task_id}")
        return selected_matches
        
    except Exception as e:
        print(f"❌ Error processing model match result for {task_id}: {e}")
        return pd.DataFrame()
""")

def main():
    """Main function to demonstrate debugging and fixes."""
    steele_dir = Path("/Users/gordonlewis/ABAP/Steele")
    
    print("🔧 BATCH MATCH INTEGRATION FIXER")
    print("=" * 60)
    print("This tool analyzes failing matches and shows how to fix CSV integration.")
    print()
    
    try:
        # Load golden dataset
        print("📊 Loading golden master dataset...")
        golden_df = load_golden_dataset()
        print(f"✅ Loaded {len(golden_df)} vehicle records")
        
        # Find the most recent batch results file
        batch_dir = steele_dir / "data" / "batch"
        jsonl_files = list(batch_dir.glob("batch_results_*.jsonl"))
        
        if not jsonl_files:
            print("❌ No batch results files found")
            return
        
        # Sort by modification time (most recent first)
        jsonl_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        latest_file = str(jsonl_files[0])
        
        print(f"📁 Using batch results: {jsonl_files[0].name}")
        
        # Show successful patterns first
        show_successful_pattern_examples(latest_file, golden_df, limit=3)
        
        # Analyze some failing cases
        failing_examples = [
            "model_match_1912_Cadillac_Four",
            "model_match_1913_Buick_Model 24", 
            "model_match_1913_Buick_Model 25"
        ]
        
        print(f"\n❌ FAILING MATCH ANALYSIS:")
        print("=" * 60)
        
        for task_id in failing_examples:
            analysis = analyze_failing_model_match(task_id, golden_df)
            
            print(f"\n📋 Recommendations for {task_id}:")
            for rec in analysis["recommendations"]:
                print(f"   {rec}")
        
        # Show integration fix
        demonstrate_csv_integration_fix(steele_dir)
        
        # Show enhanced debug script
        create_enhanced_debug_script()
        
        print(f"\n💡 SUMMARY OF FIXES NEEDED:")
        print("=" * 60)
        print("1. 🔧 Add debug output to get_model_match_result() to see why AI returns empty arrays")
        print("2. 🔧 Improve AI prompt in _create_model_matching_prompt() to be more flexible")
        print("3. 🔧 Add confidence threshold adjustment or fuzzy fallback matching")
        print("4. 🔧 Enhance update_validation_with_ai_results() error handling")
        print("5. 📊 Verify that year_make_matches DataFrame is populated correctly")
        print("\nThe main issue is that AI is being too strict and returning empty car_ids")
        print("when it should be making reasonable matches with lower confidence.")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()