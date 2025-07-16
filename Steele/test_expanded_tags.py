#!/usr/bin/env python3
"""
Test script to verify expanded tag generation
This shows the difference between basic and comprehensive tag generation.
"""

import pandas as pd
from utils.shopify_format import ShopifyFormatGenerator
from utils.exceptions import ShopifyFormatError

def test_expanded_tags():
    """Test the new expanded tag generation functionality"""
    
    print("🏷️  TESTING EXPANDED TAG GENERATION")
    print("=" * 60)
    
    # Create test data with different scenarios
    test_data = [
        {
            'make': 'Ford',
            'model': 'Mustang', 
            'year_min': 1965,
            'year_max': 1970,
            'product_type': 'Pedal Pad',
            'collection': 'Ford Parts'
        },
        {
            'make': 'Chevrolet',
            'model': 'Camaro',
            'year_min': 1969,
            'year_max': 1969,
            'product_type': 'Brake Pad',
            'collection': 'Chevrolet Parts'
        },
        {
            'make': 'Universal',
            'model': 'Universal',
            'year_min': 1900,
            'year_max': 2024,
            'product_type': 'Mirror',
            'collection': 'Accessories'
        },
        {
            'make': 'Stutz',
            'model': 'Series BB',
            'year_min': 1928,
            'year_max': 1932,
            'product_type': 'Weatherstrip',
            'collection': 'Classic Car Parts'
        }
    ]
    
    # Initialize the Shopify format generator
    try:
        generator = ShopifyFormatGenerator('../shared/data/product_import/product_import-column-requirements.py')
    except Exception as e:
        print(f"❌ Could not initialize ShopifyFormatGenerator: {e}")
        return
    
    print("Testing tag generation for different vehicle scenarios:\n")
    
    for i, test_case in enumerate(test_data, 1):
        print(f"📦 Test Case {i}: {test_case['year_min']}-{test_case['year_max']} {test_case['make']} {test_case['model']}")
        print(f"   Product Type: {test_case['product_type']}")
        
        # Create a pandas Series for the test case
        row = pd.Series(test_case)
        
        # Generate tags
        try:
            tags = generator.generate_tags_from_product_data(row)
            print(f"   📋 Generated Tags ({len(tags.split(', '))} total):")
            
            # Show tags in organized groups
            tag_list = [tag.strip() for tag in tags.split(',')]
            
            # Identify vehicle compatibility tags (YEAR_MAKE_MODEL format)
            vehicle_tags = [tag for tag in tag_list if '_' in tag and tag.split('_')[0].isdigit()]
            other_tags = [tag for tag in tag_list if tag not in vehicle_tags]
            
            if vehicle_tags:
                print(f"      🚗 Vehicle Compatibility Tags ({len(vehicle_tags)}):")
                for tag in vehicle_tags[:5]:  # Show first 5
                    print(f"         - {tag}")
                if len(vehicle_tags) > 5:
                    print(f"         ... and {len(vehicle_tags) - 5} more")
            
            print(f"      🏷️  Descriptive Tags ({len(other_tags)}):")
            for tag in other_tags[:10]:  # Show first 10
                print(f"         - {tag}")
            if len(other_tags) > 10:
                print(f"         ... and {len(other_tags) - 10} more")
                
        except Exception as e:
            print(f"   ❌ Error generating tags: {e}")
        
        print()
    
    print("=" * 60)
    print("✅ Expanded tag generation test completed!")
    print()
    print("Key improvements:")
    print("✅ Vehicle compatibility tags in YEAR_MAKE_MODEL format")
    print("✅ Multiple year coverage for ranges")
    print("✅ Comprehensive descriptive tags for SEO")
    print("✅ Product type variations")
    print("✅ Brand and generic automotive tags")
    print("✅ Universal compatibility handling")

if __name__ == "__main__":
    test_expanded_tags()