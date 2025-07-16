#!/usr/bin/env python3
"""
Test script for updated two-pass AI approach with tag refinement focus
"""

import os
import pandas as pd
from utils.ai_extraction import TwoPassAIEngine
from openai import OpenAI

def test_updated_two_pass_ai():
    """Test the updated two-pass AI implementation focused on tag generation"""
    
    print("🔧 TESTING UPDATED TWO-PASS AI WITH TAG GENERATION FOCUS")
    print("=" * 70)
    
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not found. Cannot test AI functionality.")
        return
    
    # Load golden master data
    try:
        golden_master_path = '../shared/data/master_ultimate_golden.csv'
        if not os.path.exists(golden_master_path):
            golden_master_path = 'shared/data/master_ultimate_golden.csv'
        
        print(f"📋 Loading golden master: {golden_master_path}")
        golden_df = pd.read_csv(golden_master_path)
        print(f"✅ Loaded golden master: {len(golden_df)} records")
        
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
            'description': 'Universal Accelerator Pedal Pad - Fits various makes and models',
            'stockcode': '10-0001-40',
            'price': 39.99
        }
    ]
    
    print(f"\n🧪 Testing {len(test_products)} products with updated two-pass AI:")
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
            
            # Pass 2: Generate comprehensive tags using Pass 1 context
            print("   🔄 Pass 2: Generating comprehensive tags using Pass 1 context...")
            refined_data = two_pass_engine.refine_with_golden_master(initial_data, product_info)
            print(f"   ✅ Pass 2 Results:")
            print(f"      - Title: {refined_data['title']}")
            print(f"      - Vehicle Info: {refined_data['year_min']}-{refined_data['year_max']} {refined_data['make']} {refined_data['model']}")
            
            # Show the generated tags
            if 'tags' in refined_data and refined_data['tags']:
                print(f"      - Combined Tags: {refined_data['tags']}")
            
            # Show individual tag categories
            tag_categories = ['vehicle_compatibility_tags', 'category_tags', 'fitment_tags', 'brand_tags', 'seo_tags', 'material_tags']
            for tag_category in tag_categories:
                if tag_category in refined_data and refined_data[tag_category]:
                    print(f"      - {tag_category.replace('_', ' ').title()}: {refined_data[tag_category]}")
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            import traceback
            print(f"   ❌ Traceback: {traceback.format_exc()}")
    
    print("\n" + "=" * 70)
    print("✅ Updated two-pass AI testing completed!")
    print("\n📋 KEY CHANGES:")
    print("   - Pass 1: Still extracts vehicle information")
    print("   - Pass 2: NOW GENERATES COMPREHENSIVE TAGS using Pass 1 context")
    print("   - Tags include: vehicle compatibility, category, fitment, brand, SEO, material")
    print("   - All tags are combined into a single 'tags' field for Shopify")


if __name__ == "__main__":
    test_updated_two_pass_ai()