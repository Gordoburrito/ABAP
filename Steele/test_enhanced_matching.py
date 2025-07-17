#!/usr/bin/env python3
"""
Test script for the enhanced Model DC vs Series DC matching
"""

import pandas as pd
import re

def extract_core_identifier(model_str):
    """Extract core identifier after removing common prefixes"""
    parts = re.split(r'[^a-zA-Z0-9]+', model_str.lower())
    parts = [p for p in parts if p]
    
    common_prefixes = ['model', 'series', 'type', 'class', 'version', 'grade']
    
    if len(parts) >= 2 and parts[0] in common_prefixes:
        return ''.join(parts[1:])
    return ''.join(parts)

def enhanced_model_similarity(input_model, golden_model):
    """Enhanced model similarity that handles Model/Series equivalence"""
    
    print(f"         Comparing: '{input_model}' vs '{golden_model}'")
    
    # Extract core identifiers
    core1 = extract_core_identifier(input_model)
    core2 = extract_core_identifier(golden_model)
    
    print(f"         Core identifiers: '{core1}' vs '{core2}'")
    
    if core1 == core2 and len(core1) >= 2:
        print(f"         🎯 CORE MATCH! (Model/Series equivalence)")
        return 0.95
    
    # Original normalization for other cases
    def normalize(s):
        return s.lower().replace(" ", "").replace("-", "").replace("_", "")
    
    str1_norm = normalize(input_model)
    str2_norm = normalize(golden_model)
    
    if str1_norm == str2_norm:
        print(f"         Exact normalized match!")
        return 1.0
    
    # Substring matching
    if len(str1_norm) >= 3 and len(str2_norm) >= 3:
        if str1_norm in str2_norm or str2_norm in str1_norm:
            print(f"         Substring match")
            return 0.9
    
    # Character-based similarity
    max_len = max(len(str1_norm), len(str2_norm))
    if max_len == 0:
        return 0.0
    
    matches = 0
    i = j = 0
    while i < len(str1_norm) and j < len(str2_norm):
        if str1_norm[i] == str2_norm[j]:
            matches += 1
            i += 1
            j += 1
        else:
            i += 1
    
    similarity = matches / max_len
    print(f"         Character-based: {similarity:.3f}")
    
    return similarity if similarity >= 0.4 else 0.0

def test_enhanced_matching():
    """Test the enhanced matching with the failing case"""
    
    # Test data matching your example
    test_models = ['Series DC', 'Series DA', 'Series DD', 'Series DB']
    input_model = 'Model DC'
    
    print("=== ENHANCED MATCHING TEST ===")
    print(f"Input: {input_model}")
    print(f"Available models: {test_models}")
    print()
    
    matches = []
    threshold = 0.5
    
    for model in test_models:
        similarity = enhanced_model_similarity(input_model, model)
        print(f"      '{input_model}' vs '{model}' = {similarity:.3f}")
        
        if similarity >= threshold:
            matches.append((model, similarity))
            print(f"         ✅ Above threshold ({threshold})")
        else:
            print(f"         ❌ Below threshold ({threshold})")
        print()
    
    print(f"RESULTS:")
    if matches:
        print(f"✅ Found {len(matches)} matches:")
        for model, score in matches:
            print(f"   - {model} (score: {score:.3f})")
    else:
        print("❌ No matches found")

if __name__ == "__main__":
    test_enhanced_matching()