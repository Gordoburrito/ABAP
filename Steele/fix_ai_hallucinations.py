#!/usr/bin/env python3
"""
Fix AI hallucinations in vehicle tags by referencing the master_ultimate_golden.csv file.

This script corrects common AI extraction errors like:
- "Galaxie_500" -> "Galaxie 500" (space instead of underscore)
- "Galaxie_500_XL" -> "Galaxie 500 XL"
- Other common model splits

The script uses simple string replacement patterns based on known issues.
"""

import pandas as pd
import re
from datetime import datetime

def fix_vehicle_tags(tags_string):
    """Fix vehicle tags using simple string replacement patterns."""
    if pd.isna(tags_string) or tags_string == "":
        return tags_string
    
    fixed_tags = tags_string
    
    # Fix common Ford Galaxie issues
    fixed_tags = re.sub(r'(\d{4}_Ford_)Galaxie_500_XL', r'\1Galaxie 500 XL', fixed_tags)
    fixed_tags = re.sub(r'(\d{4}_Ford_)Galaxie_500', r'\1Galaxie 500', fixed_tags)
    
    # Fix Mercury Galaxie issues
    fixed_tags = re.sub(r'(\d{4}_Mercury_)Galaxie_500_XL', r'\1Galaxie 500 XL', fixed_tags)
    fixed_tags = re.sub(r'(\d{4}_Mercury_)Galaxie_500', r'\1Galaxie 500', fixed_tags)
    
    # Fix other common model splits (add more as needed)
    fixed_tags = re.sub(r'(\d{4}_Ford_)Custom_500', r'\1Custom 500', fixed_tags)
    fixed_tags = re.sub(r'(\d{4}_Ford_)Country_Sedan', r'\1Country Sedan', fixed_tags)
    fixed_tags = re.sub(r'(\d{4}_Ford_)Country_Squire', r'\1Country Squire', fixed_tags)
    fixed_tags = re.sub(r'(\d{4}_Ford_)Ranch_Wagon', r'\1Ranch Wagon', fixed_tags)
    fixed_tags = re.sub(r'(\d{4}_Ford_)Station_Bus', r'\1Station Bus', fixed_tags)
    fixed_tags = re.sub(r'(\d{4}_Ford_)Club_Wagon', r'\1Club Wagon', fixed_tags)
    fixed_tags = re.sub(r'(\d{4}_Ford_)Falcon_Sedan_Delivery', r'\1Falcon Sedan Delivery', fixed_tags)
    fixed_tags = re.sub(r'(\d{4}_Ford_)Courier_Sedan_Delivery', r'\1Courier Sedan Delivery', fixed_tags)
    
    # Fix Edsel model splits
    fixed_tags = re.sub(r'(\d{4}_Edsel_)Galaxie_500', r'\1Galaxie 500', fixed_tags)
    fixed_tags = re.sub(r'(\d{4}_Edsel_)Custom_500', r'\1Custom 500', fixed_tags)
    fixed_tags = re.sub(r'(\d{4}_Edsel_)Country_Sedan', r'\1Country Sedan', fixed_tags)
    fixed_tags = re.sub(r'(\d{4}_Edsel_)Country_Squire', r'\1Country Squire', fixed_tags)
    fixed_tags = re.sub(r'(\d{4}_Edsel_)Ranch_Wagon', r'\1Ranch Wagon', fixed_tags)
    fixed_tags = re.sub(r'(\d{4}_Edsel_)Station_Bus', r'\1Station Bus', fixed_tags)
    fixed_tags = re.sub(r'(\d{4}_Edsel_)Club_Wagon', r'\1Club Wagon', fixed_tags)
    fixed_tags = re.sub(r'(\d{4}_Edsel_)Falcon_Sedan_Delivery', r'\1Falcon Sedan Delivery', fixed_tags)
    
    # Fix Mercury model splits
    fixed_tags = re.sub(r'(\d{4}_Mercury_)Galaxie_500', r'\1Galaxie 500', fixed_tags)
    fixed_tags = re.sub(r'(\d{4}_Mercury_)Custom_500', r'\1Custom 500', fixed_tags)
    fixed_tags = re.sub(r'(\d{4}_Mercury_)Country_Sedan', r'\1Country Sedan', fixed_tags)
    fixed_tags = re.sub(r'(\d{4}_Mercury_)Country_Squire', r'\1Country Squire', fixed_tags)
    fixed_tags = re.sub(r'(\d{4}_Mercury_)Ranch_Wagon', r'\1Ranch Wagon', fixed_tags)
    fixed_tags = re.sub(r'(\d{4}_Mercury_)Station_Bus', r'\1Station Bus', fixed_tags)
    fixed_tags = re.sub(r'(\d{4}_Mercury_)Club_Wagon', r'\1Club Wagon', fixed_tags)
    fixed_tags = re.sub(r'(\d{4}_Mercury_)Falcon_Sedan_Delivery', r'\1Falcon Sedan Delivery', fixed_tags)
    
    return fixed_tags

def main():
    """Main function to fix AI hallucinations in the Shopify format file."""
    print("=== Fixing AI Hallucinations in Vehicle Tags ===")
    
    # Load the Shopify format file
    shopify_file = 'data/results/corrected_shopify_format_20250705_170607.csv'
    print(f"Loading Shopify file: {shopify_file}")
    
    df = pd.read_csv(shopify_file)
    print(f"Loaded {len(df)} products")
    
    # Show some examples before fixing
    print("\n=== Examples Before Fixing ===")
    sample_tags = df['Tags'].iloc[0] if len(df) > 0 else ""
    if sample_tags and 'Galaxie_500' in sample_tags:
        print(f"Sample tags (before): {sample_tags[:300]}...")
    
    # Fix the Tags column
    print("\nFixing vehicle tags...")
    df['Tags'] = df['Tags'].apply(fix_vehicle_tags)
    
    # Show some examples after fixing
    print("\n=== Examples After Fixing ===")
    sample_tags_after = df['Tags'].iloc[0] if len(df) > 0 else ""
    if sample_tags_after and 'Galaxie 500' in sample_tags_after:
        print(f"Sample tags (after): {sample_tags_after[:300]}...")
    
    # Save the corrected file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'data/results/fixed_hallucinations_shopify_{timestamp}.csv'
    
    print(f"\nSaving corrected file: {output_file}")
    df.to_csv(output_file, index=False)
    
    print(f"✅ Successfully fixed AI hallucinations!")
    print(f"   Input file: {shopify_file}")
    print(f"   Output file: {output_file}")
    
    # Count how many Galaxie_500 tags were fixed
    galaxie_500_count = sum(1 for tags in df['Tags'] if 'Galaxie 500' in str(tags))
    print(f"   Products with 'Galaxie 500' tags: {galaxie_500_count}")
    
    return output_file

if __name__ == "__main__":
    main() 