#!/usr/bin/env python3
"""
Simple script to create pattern → car IDs mapping from batch results.

Takes custom_ids like "model_match_1929_Stutz_Stutz" and creates mapping:
"1929_Stutz_Stutz" → ["1929_Stutz_Model M"]
"""

import json
import re
from pathlib import Path
from typing import Dict, List


def extract_pattern_from_custom_id(custom_id: str) -> str:
    """
    Extract pattern key from custom_id by stripping prefix until first digit.
    
    Examples:
        "model_match_1929_Stutz_Stutz" → "1929_Stutz_Stutz"
        "make_match_1912_Buick_Model 28" → "1912_Buick_Model 28"
    """
    # Find first digit in the string
    match = re.search(r'\d', custom_id)
    if match:
        # Return everything from first digit onwards
        return custom_id[match.start():]
    else:
        # No digit found, return as-is
        return custom_id


def parse_content_for_car_ids(content_str: str) -> List[str]:
    """
    Parse content string to extract selected_car_ids.
    
    Handles both:
    - JSON responses: {"selected_car_ids": ["1929_Stutz_Model M"], ...}
    - NO_MATCH responses: "NO_MATCH"
    """
    if content_str.strip() == "NO_MATCH":
        return []
    
    try:
        content_json = json.loads(content_str)
        return content_json.get("selected_car_ids", [])
    except json.JSONDecodeError:
        print(f"Warning: Could not parse content JSON: {content_str}")
        return []


def create_pattern_car_id_mapping_from_batch_results(
    batch_results_file: str = "data/batch/batch_results_20250718_121034.jsonl"
) -> Dict[str, List[str]]:
    """
    Create simple pattern → car IDs mapping from batch results file.
    
    Args:
        batch_results_file: Path to JSONL batch results file
        
    Returns:
        Dictionary mapping pattern keys to car ID arrays
    """
    batch_file_path = Path(batch_results_file)
    if not batch_file_path.exists():
        raise FileNotFoundError(f"Batch results file not found: {batch_results_file}")
    
    pattern_mapping = {}
    
    print(f"📖 Reading batch results from: {batch_results_file}")
    
    with open(batch_file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                # Parse the JSONL line
                batch_result = json.loads(line)
                
                # Extract custom_id
                custom_id = batch_result.get("custom_id", "")
                if not custom_id:
                    continue
                
                # Extract pattern key from custom_id
                pattern_key = extract_pattern_from_custom_id(custom_id)
                
                # Extract content from response
                response = batch_result.get("response", {})
                body = response.get("body", {})
                choices = body.get("choices", [])
                
                if choices and len(choices) > 0:
                    message = choices[0].get("message", {})
                    content = message.get("content", "")
                    
                    # Parse content for car IDs
                    car_ids = parse_content_for_car_ids(content)
                    
                    # Add to mapping
                    pattern_mapping[pattern_key] = car_ids
                    
                    if car_ids:
                        print(f"✅ {pattern_key} → {car_ids}")
                    else:
                        print(f"⚪ {pattern_key} → [] (no matches)")
                else:
                    print(f"⚠️  Line {line_num}: No choices in response")
                    
            except json.JSONDecodeError as e:
                print(f"❌ Line {line_num}: JSON parse error: {e}")
            except Exception as e:
                print(f"❌ Line {line_num}: Unexpected error: {e}")
    
    print(f"\n📊 Created mapping for {len(pattern_mapping)} patterns")
    
    # Show summary stats
    with_matches = sum(1 for car_ids in pattern_mapping.values() if car_ids)
    without_matches = len(pattern_mapping) - with_matches
    
    print(f"   ✅ Patterns with matches: {with_matches}")
    print(f"   ⚪ Patterns without matches: {without_matches}")
    
    return pattern_mapping


def save_pattern_mapping(pattern_mapping: Dict[str, List[str]], output_file: str = "data/pattern_car_id_mapping.json"):
    """Save pattern mapping to JSON file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(pattern_mapping, f, indent=2)
    
    print(f"💾 Pattern mapping saved to: {output_file}")


def main():
    """Main function to create and save pattern mapping."""
    print("🚀 Creating Pattern → Car IDs Mapping from Batch Results")
    print("=" * 60)
    
    try:
        # Create the mapping
        pattern_mapping = create_pattern_car_id_mapping_from_batch_results()
        
        # Save to file
        save_pattern_mapping(pattern_mapping)
        
        # Show some examples
        print("\n📋 Example mappings:")
        for i, (pattern, car_ids) in enumerate(list(pattern_mapping.items())[:5]):
            print(f"   {pattern} → {car_ids}")
            if i >= 4:  # Show max 5 examples
                break
        
        if len(pattern_mapping) > 5:
            print(f"   ... and {len(pattern_mapping) - 5} more")
        
        print("\n✅ Pattern mapping creation complete!")
        
        return pattern_mapping
        
    except Exception as e:
        print(f"❌ Error creating pattern mapping: {e}")
        return {}


if __name__ == "__main__":
    main()