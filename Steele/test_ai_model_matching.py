#!/usr/bin/env python3
"""
Test AI model matching with improved prompts for Model DC vs Series DC
"""

import pandas as pd
import os
from openai import OpenAI
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

load_dotenv()

class ModelMatchResult(BaseModel):
    """Pydantic model for AI model matching responses"""
    selected_car_ids: List[str]
    confidence: float
    match_type: str = "ai_model_match"

def create_improved_prompt(input_model, input_submodel, input_type, input_doors, input_body_type, car_id_model_map):
    """Create improved AI prompt for model matching"""
    
    # Build input specifications
    input_specs = [f"Model: {input_model}"]
    if input_submodel:
        input_specs.append(f"Submodel: {input_submodel}")
    if input_type:
        input_specs.append(f"Type: {input_type}")
    if input_doors:
        input_specs.append(f"Doors: {input_doors}")
    if input_body_type:
        input_specs.append(f"Body Type: {input_body_type}")
    
    # Build available options
    options_text = []
    for car_id, model in car_id_model_map.items():
        options_text.append(f"- {car_id}: {model}")
    
    prompt = f"""Match vehicle to car_ids. Be VERY flexible with model naming variations.

INPUT: {' | '.join(input_specs)}
OPTIONS: {' | '.join(options_text)}

MATCHING RULES - Be generous:
- "Model DC" = "Series DC" = "Type DC" (same core identifier)
- "Model 6-14" = "Series 614" (number variations)  
- Focus on letters/numbers, ignore prefixes like Model/Series/Type/Class
- Consider body type and doors as secondary factors
- When in doubt, include the match

EXAMPLES:
- Input "Model DC" → Select "Series DC" car_ids (confidence 0.8+)
- Input "Type 350" → Select "Model 350", "Series 350" car_ids
- Input "Version XL" → Select "Class XL", "Grade XL" car_ids

JSON response:
{{
    "selected_car_ids": ["car_id1", "car_id2"],
    "confidence": 0.8,
    "match_type": "ai_model_match"
}}

Rules: Use exact car_ids from options. Confidence 0.5-1.0. Include all reasonable matches."""

    return prompt

def test_ai_model_matching():
    """Test AI model matching with the improved prompt"""
    
    # Test data matching your failing case
    test_data = {
        '1970_Make_Series_DC': 'Series DC',
        '1970_Make_Series_DA': 'Series DA', 
        '1970_Make_Series_DD': 'Series DD',
        '1970_Make_Series_DB': 'Series DB'
    }
    
    # Setup OpenAI client
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not found")
        return
    
    client = OpenAI(api_key=api_key)
    
    print("=== TESTING AI MODEL MATCHING ===")
    print("Input: Model DC")
    print("Available options:")
    for car_id, model in test_data.items():
        print(f"  - {car_id}: {model}")
    print()
    
    # Create prompt
    prompt = create_improved_prompt(
        "Model DC",  # input_model
        "Base",      # input_submodel
        "Car",       # input_type
        "2",         # input_doors
        "Convertible", # input_body_type
        test_data
    )
    
    print("AI Prompt:")
    print("-" * 50)
    print(prompt)
    print("-" * 50)
    print()
    
    try:
        # Call AI
        response = client.beta.chat.completions.parse(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are an automotive expert. Be flexible with model name variations."},
                {"role": "user", "content": prompt}
            ],
            response_format=ModelMatchResult,
            temperature=0.1
        )
        
        result = ModelMatchResult.model_validate_json(response.choices[0].message.content)
        
        print("🤖 AI RESPONSE:")
        print(f"Selected car_ids: {result.selected_car_ids}")
        print(f"Confidence: {result.confidence}")
        print(f"Match type: {result.match_type}")
        
        if result.confidence >= 0.5:
            print(f"✅ SUCCESS! AI found matches with confidence {result.confidence}")
            for car_id in result.selected_car_ids:
                if car_id in test_data:
                    print(f"   - {car_id}: {test_data[car_id]}")
        else:
            print(f"❌ FAILED! AI confidence {result.confidence} too low")
            
        # Show token usage if available
        if hasattr(response, 'usage') and response.usage:
            print(f"💰 Tokens: {response.usage.prompt_tokens} input, {response.usage.completion_tokens} output")
        
    except Exception as e:
        print(f"❌ AI Error: {e}")

if __name__ == "__main__":
    test_ai_model_matching()