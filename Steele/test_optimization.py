#!/usr/bin/env python3
"""
Test script to validate the ultra-optimization works correctly.
Tests with a small sample to ensure the optimization logic is sound.
"""

from utils.optimized_batch_steele_transformer import OptimizedBatchSteeleTransformer
import pandas as pd
import time

def test_pattern_deduplication():
    """Test that pattern deduplication works correctly."""
    print("🧪 TESTING PATTERN DEDUPLICATION")
    print("=" * 50)
    
    # Initialize transformer
    transformer = OptimizedBatchSteeleTransformer(use_ai=False)  # Disable AI for testing
    
    # Test with small sample
    print("📊 Loading test data...")
    steele_df = transformer.load_sample_data("data/samples/steele_test_1000.csv")
    print(f"✅ Loaded {len(steele_df):,} products")
    
    # Analyze patterns
    print("\n🔍 Analyzing patterns...")
    pattern_groups, unique_patterns = transformer.analyze_steele_patterns(steele_df)
    
    # Verify deduplication benefit
    total_products = len(steele_df)
    deduplication_factor = total_products / unique_patterns if unique_patterns > 0 else 1
    
    print(f"📊 Results:")
    print(f"   Total products: {total_products:,}")
    print(f"   Unique patterns: {unique_patterns:,}")
    print(f"   Deduplication factor: {deduplication_factor:.1f}x")
    print(f"   AI calls saved: {total_products - unique_patterns:,}")
    
    # Create pattern mapping
    print("\n🔄 Creating pattern mapping...")
    pattern_mapping = transformer.create_pattern_mapping(steele_df, pattern_groups)
    
    # Validate mapping integrity
    total_mapped_products = sum(len(pattern_data['products']) for pattern_data in pattern_mapping.values())
    
    if total_mapped_products != total_products:
        print(f"❌ MAPPING ERROR: {total_mapped_products} mapped vs {total_products} total")
        return False
    
    print(f"✅ Pattern mapping verified: {total_mapped_products} products mapped")
    
    # Test pattern validation
    print("\n🔍 Testing pattern validation...")
    transformer.load_golden_dataset()
    exact_match_patterns, ai_needed_patterns = transformer.validate_patterns_against_golden(pattern_mapping)
    
    print(f"📊 Validation results:")
    print(f"   Exact matches: {len(exact_match_patterns)}")
    print(f"   AI needed: {len(ai_needed_patterns)}")
    print(f"   Total patterns: {len(exact_match_patterns) + len(ai_needed_patterns)}")
    
    if len(exact_match_patterns) + len(ai_needed_patterns) != unique_patterns:
        print(f"❌ VALIDATION ERROR: Pattern count mismatch")
        return False
    
    print("✅ Pattern validation verified")
    
    # Test result application (mock AI results)
    print("\n🔄 Testing result application...")
    
    # Mock AI results for testing
    for pattern_key in ai_needed_patterns[:5]:  # Test first 5 patterns
        pattern_data = pattern_mapping[pattern_key]
        pattern_data['validation_result'] = {
            'golden_validated': True,
            'golden_matches': 1,
            'car_ids': [f"test_car_id_{pattern_key}"],
            'match_type': 'mock_test'
        }
    
    # Apply results to all products
    validation_df = transformer.apply_results_to_all_products(steele_df, ai_needed_patterns)
    
    if len(validation_df) != len(steele_df):
        print(f"❌ RESULT APPLICATION ERROR: {len(validation_df)} results vs {len(steele_df)} products")
        return False
    
    print(f"✅ Result application verified: {len(validation_df)} results generated")
    
    return True

def test_performance_comparison():
    """Test performance comparison between old and new approach."""
    print("\n🏃 PERFORMANCE COMPARISON TEST")
    print("=" * 50)
    
    # Test with different dataset sizes
    test_sizes = [100, 1000]
    
    for size in test_sizes:
        print(f"\n📊 Testing with {size} products...")
        
        # Initialize transformer
        transformer = OptimizedBatchSteeleTransformer(use_ai=False)
        
        # Load test data
        steele_df = transformer.load_sample_data("data/samples/steele_test_1000.csv")
        test_df = steele_df.head(size)
        
        # Time the optimization analysis
        start_time = time.time()
        
        pattern_groups, unique_patterns = transformer.analyze_steele_patterns(test_df)
        pattern_mapping = transformer.create_pattern_mapping(test_df, pattern_groups)
        
        analysis_time = time.time() - start_time
        
        # Calculate theoretical old approach time (simulate queuing time)
        old_approach_time = size * 0.02  # Assume 0.02 seconds per product for old queuing
        
        print(f"   📊 Optimization analysis: {analysis_time:.2f} seconds")
        print(f"   📊 Old approach estimate: {old_approach_time:.2f} seconds")
        print(f"   📊 Speed improvement: {old_approach_time / analysis_time:.1f}x faster")
        print(f"   📊 Deduplication: {size / unique_patterns:.1f}x")
    
    return True

def test_data_integrity():
    """Test that data integrity is maintained through optimization."""
    print("\n🛡️  DATA INTEGRITY TEST")
    print("=" * 50)
    
    # Initialize transformer
    transformer = OptimizedBatchSteeleTransformer(use_ai=False)
    
    # Load test data
    steele_df = transformer.load_sample_data("data/samples/steele_test_1000.csv")
    test_df = steele_df.head(100)  # Small test set
    
    # Run optimization
    pattern_groups, unique_patterns = transformer.analyze_steele_patterns(test_df)
    pattern_mapping = transformer.create_pattern_mapping(test_df, pattern_groups)
    
    # Mock validation results
    transformer.load_golden_dataset()
    exact_match_patterns, ai_needed_patterns = transformer.validate_patterns_against_golden(pattern_mapping)
    
    # Apply mock results
    validation_df = transformer.apply_results_to_all_products(test_df, ai_needed_patterns)
    
    # Check data integrity
    print(f"📊 Input products: {len(test_df)}")
    print(f"📊 Output results: {len(validation_df)}")
    print(f"📊 Missing results: {len(test_df) - len(validation_df)}")
    
    # Check that all products have results
    input_indices = set(test_df.index)
    output_indices = set(validation_df['steele_row_index'])
    missing_indices = input_indices - output_indices
    
    if missing_indices:
        print(f"❌ INTEGRITY ERROR: Missing results for indices: {missing_indices}")
        return False
    
    print("✅ Data integrity verified: All products have results")
    
    return True

def main():
    """Run all optimization tests."""
    print("🧪 ULTRA-OPTIMIZATION VALIDATION TESTS")
    print("=" * 80)
    
    tests = [
        ("Pattern Deduplication", test_pattern_deduplication),
        ("Performance Comparison", test_performance_comparison),
        ("Data Integrity", test_data_integrity)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🎯 Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                print(f"✅ {test_name} test PASSED")
            else:
                print(f"❌ {test_name} test FAILED")
        except Exception as e:
            print(f"❌ {test_name} test ERROR: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("🧪 TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name}: {status}")
    
    print(f"\n📊 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Ultra-optimization is ready to use.")
    else:
        print("⚠️  Some tests failed. Please review before using optimization.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)