#!/usr/bin/env python3
"""
Extract 100 products with vehicle-specific information for testing
"""

import pandas as pd
import re
import os

def extract_vehicle_products():
    """Extract 100 products that contain vehicle-specific information"""
    
    print("🔍 EXTRACTING 100 VEHICLE-SPECIFIC PRODUCTS")
    print("=" * 50)
    
    # Load the full dataset
    try:
        df = pd.read_excel('data/raw/steele.xlsx')
        print(f"✅ Loaded {len(df)} total products")
    except Exception as e:
        print(f"❌ Could not load steele.xlsx: {e}")
        return
    
    # Vehicle-specific keywords that indicate specific fitment
    vehicle_keywords = [
        # Makes
        'Cord', 'Hupmobile', 'Ford', 'Chevrolet', 'Buick', 'Oldsmobile', 'Pontiac',
        'Cadillac', 'Lincoln', 'Mercury', 'Dodge', 'Plymouth', 'Chrysler', 'DeSoto',
        'Packard', 'Studebaker', 'Hudson', 'Nash', 'Kaiser', 'Frazer', 'Willys',
        'Graham', 'LaSalle', 'Auburn', 'Duesenberg', 'Pierce-Arrow', 'Marmon',
        
        # Year patterns
        '193', '194', '195', '196', '197', '198', '199', '200',
        
        # Model indicators
        'Skylark', 'Mustang', 'Camaro', 'Corvette', 'Thunderbird', 'Galaxy',
        'Fairlane', 'Falcon', 'Torino', 'Maverick', 'Pinto', 'Escort',
        'Impala', 'Chevelle', 'Nova', 'Malibu', 'Monte Carlo', 'Bel Air',
        'Model A', 'Model T', 'Coupe', 'Sedan', 'Convertible', 'Roadster'
    ]
    
    # Find products with vehicle-specific information
    vehicle_products = []
    
    for idx, row in df.iterrows():
        product_text = f"{row['Product Name']} {row['Description']}".lower()
        
        # Check for vehicle keywords
        has_vehicle_info = any(keyword.lower() in product_text for keyword in vehicle_keywords)
        
        # Check for year patterns (4-digit years or year ranges)
        has_year_pattern = bool(re.search(r'\b(19|20)\d{2}\b|(\d{4}[-–]\d{4})', str(row['Description'])))
        
        # Look for specific model numbers or names
        has_model_info = bool(re.search(r'\b(Model|Series|Type)\s+[A-Z0-9]+\b|^\d{3,4}$|\b[A-Z]{2,}\s+\d+\b', str(row['Description'])))
        
        if has_vehicle_info or has_year_pattern or has_model_info:
            vehicle_products.append({
                'StockCode': row['StockCode'],
                'Product Name': row['Product Name'],
                'Description': row['Description'],
                'StockUom': row.get('StockUom', 'ea.'),
                'UPC Code': row.get('UPC Code', ''),
                'MAP': row.get('MAP', 0),
                'Dealer Price': row.get('Dealer Price', 0),
                'Vehicle_Score': (int(has_vehicle_info) + int(has_year_pattern) + int(has_model_info))
            })
        
        # Stop when we have enough products
        if len(vehicle_products) >= 150:  # Get extra to pick the best 100
            break
    
    print(f"✅ Found {len(vehicle_products)} products with vehicle information")
    
    # Sort by vehicle score (most vehicle-specific first)
    vehicle_products.sort(key=lambda x: x['Vehicle_Score'], reverse=True)
    
    # Take the top 100
    top_100 = vehicle_products[:100]
    
    print(f"📋 Selected top 100 products:")
    print(f"   High vehicle specificity: {len([p for p in top_100 if p['Vehicle_Score'] >= 2])}")
    print(f"   Medium vehicle specificity: {len([p for p in top_100 if p['Vehicle_Score'] == 1])}")
    
    # Show sample products
    print(f"\n📋 SAMPLE VEHICLE-SPECIFIC PRODUCTS:")
    for i, product in enumerate(top_100[:10], 1):
        score_desc = "High" if product['Vehicle_Score'] >= 2 else "Medium" if product['Vehicle_Score'] == 1 else "Low"
        print(f"   {i:2d}. [{score_desc}] {product['Product Name'][:60]}...")
        if product['Description'] and len(str(product['Description'])) > 10:
            print(f"       {str(product['Description'])[:80]}...")
    
    # Create DataFrame and save
    test_df = pd.DataFrame(top_100)
    test_df = test_df.drop('Vehicle_Score', axis=1)  # Remove scoring column
    
    # Save to test file
    output_path = 'data/results/test_100_products.csv'
    os.makedirs('data/results', exist_ok=True)
    test_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Saved 100 test products to: {output_path}")
    print(f"📊 Ready for vehicle-specific tag testing")
    
    # Show statistics
    print(f"\n📋 DATASET STATISTICS:")
    print(f"   Products with known vehicle makes: {len([p for p in top_100 if any(make.lower() in str(p['Product Name']).lower() + str(p['Description']).lower() for make in ['Ford', 'Chevrolet', 'Cord', 'Hupmobile', 'Buick', 'Dodge'])])}")
    print(f"   Products with year information: {len([p for p in top_100 if re.search(r'\b(19|20)\d{2}\b', str(p['Description']))])}")
    print(f"   Products with model information: {len([p for p in top_100 if re.search(r'Model|Series|Skylark|Mustang', str(p['Description']))])}")
    
    return output_path


if __name__ == "__main__":
    extract_vehicle_products()