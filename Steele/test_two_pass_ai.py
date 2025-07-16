#!/usr/bin/env python3
"""
Test script for two-pass AI approach with golden master tag generation
"""

import os
import pandas as pd
from utils.ai_extraction import TwoPassAIEngine
from utils.golden_master_tag_generator import GoldenMasterTagGenerator
from openai import OpenAI

def test_two_pass_ai():
    """Test the two-pass AI implementation"""
    
    print("🔧 TESTING TWO-PASS AI WITH GOLDEN MASTER TAG GENERATION")
    print("=" * 70)
    
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not found. Testing with mock data only.")
        test_golden_master_tags_only()
        return
    
    # Load golden master data
    try:
        golden_master_path = '../shared/data/master_ultimate_golden.csv'
        if not os.path.exists(golden_master_path):
            golden_master_path = 'shared/data/master_ultimate_golden.csv'
        
        print(f"📋 Loading golden master: {golden_master_path}")
        golden_df = pd.read_csv(golden_master_path)
        print(f"✅ Loaded golden master: {len(golden_df)} records")
        
        # Show sample of golden master data
        print("\n📊 Golden Master Sample:")
        sample_cols = ['year', 'make', 'model', 'car_id']
        available_cols = [col for col in sample_cols if col in golden_df.columns]
        print(golden_df[available_cols].head(3).to_string(index=False))
        
    except Exception as e:
        print(f"❌ Could not load golden master: {e}")
        return
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Initialize two-pass engine
    two_pass_engine = TwoPassAIEngine(client, golden_df)
    
    # Test data - sample Steele product descriptions
    test_products = [
        {
            'description': 'Front Door Vent Window Weatherstrips for 1939-1941 Hupmobile Skylark and Graham Hollywood 4 door sedans',
            'stockcode': '10-0111-55',
            'price': 79.99
        },
        {
            'description': 'Glass Weatherstrip Kit for Independent Models (1920-1929) running board step pad',
            'stockcode': '10-0108-45', 
            'price': 45.99
        },
        {
            'description': 'Accelerator Pedal Pad - Universal fit for various makes and models',
            'stockcode': '10-0001-40',
            'price': 39.99
        }
    ]
    
    print(f"\n🧪 Testing {len(test_products)} products with two-pass AI:")
    print("-" * 50)
    
    for i, product in enumerate(test_products, 1):
        print(f"\n🔍 Product {i}: {product['description'][:60]}...")
        
        # Create product info string
        product_info = f"""
Product Name: {product['description']}
Stock Code: {product['stockcode']}
Price: ${product['price']}
"""
        
        try:
            # Pass 1: Extract initial vehicle info
            print("   🔄 Pass 1: Extracting initial vehicle information...")
            initial_data = two_pass_engine.extract_initial_vehicle_info(product_info)
            print(f"   ✅ Pass 1 Results:")
            print(f"      - Year Range: {initial_data['year_min']}-{initial_data['year_max']}")
            print(f"      - Make: {initial_data['make']}")
            print(f"      - Model: {initial_data['model']}")
            
            # Pass 2: Refine with golden master
            print("   🔄 Pass 2: Refining with golden master context...")
            refined_data = two_pass_engine.refine_with_golden_master(initial_data, product_info)
            print(f"   ✅ Pass 2 Results:")
            print(f"      - Final Make: {refined_data['make']}")
            print(f"      - Final Model: {refined_data['model']}")
            print(f"      - Title: {refined_data['title']}")
            
            # Test tag generation
            print("   🏷️  Generating vehicle tags from golden master...")
            tag_generator = GoldenMasterTagGenerator(golden_df)
            
            # Parse models (handle string format)
            models = []
            if isinstance(refined_data['model'], str):
                models = [m.strip() for m in refined_data['model'].replace(',', '|').split('|') if m.strip()]
            
            if refined_data['make'].lower() not in ['universal', 'unknown'] and models:
                vehicle_tags = tag_generator.generate_vehicle_tags_from_car_ids(
                    refined_data['year_min'],
                    refined_data['year_max'], 
                    refined_data['make'],
                    models
                )
                
                if vehicle_tags:
                    print(f"   ✅ Generated {len(vehicle_tags)} vehicle tags:")
                    for tag in vehicle_tags[:5]:  # Show first 5
                        print(f"      - {tag}")
                    if len(vehicle_tags) > 5:
                        print(f"      ... and {len(vehicle_tags) - 5} more")
                else:
                    print("   ⚠️  No vehicle tags generated (no matching combinations)")
            else:
                print("   ℹ️  Universal product - no vehicle-specific tags")
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
    
    print("\n" + "=" * 70)
    print("✅ Two-pass AI testing completed!")


def test_golden_master_tags_only():
    """Test only golden master tag generation without AI"""
    
    print("\n🏷️  TESTING GOLDEN MASTER TAG GENERATION ONLY")
    print("=" * 50)
    
    try:
        golden_master_path = '../shared/data/master_ultimate_golden.csv'
        if not os.path.exists(golden_master_path):
            golden_master_path = 'shared/data/master_ultimate_golden.csv'
        
        golden_df = pd.read_csv(golden_master_path)
        tag_generator = GoldenMasterTagGenerator(golden_df)
        
        # Test known vehicle combinations
        test_cases = [
            {'year_min': 1939, 'year_max': 1941, 'make': 'Hupmobile', 'models': ['Skylark']},
            {'year_min': 1965, 'year_max': 1970, 'make': 'Ford', 'models': ['Mustang']},
            {'year_min': 1928, 'year_max': 1932, 'make': 'Stutz', 'models': ['Model M']}
        ]
        
        for i, case in enumerate(test_cases, 1):
            print(f"\n🧪 Test Case {i}: {case['year_min']}-{case['year_max']} {case['make']} {case['models']}")
            
            tags = tag_generator.generate_vehicle_tags_from_car_ids(
                case['year_min'], case['year_max'], case['make'], case['models']
            )
            
            if tags:
                print(f"   ✅ Generated {len(tags)} tags:")
                for tag in tags[:3]:
                    print(f"      - {tag}")
                if len(tags) > 3:
                    print(f"      ... and {len(tags) - 3} more")
            else:
                print("   ⚠️  No tags generated")
        
        # Show statistics
        stats = tag_generator.get_statistics()
        print(f"\n📊 Golden Master Statistics:")
        print(f"   - Total Records: {stats.get('total_records', 'N/A')}")
        print(f"   - Unique Car IDs: {stats.get('unique_car_ids', 'N/A')}")
        print(f"   - Year Range: {stats.get('year_range', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    test_two_pass_ai()