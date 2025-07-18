#!/usr/bin/env python3
"""
Pattern-based tag mapping processor for Steele data.

This script:
1. Loads the pattern car ID mapping from data/pattern_car_id_mapping.json
2. Processes Steele data to create year_make_model keys with underscores
3. Looks up corresponding tags from the mapping
4. Groups products by StockCode (variant SKU) and consolidates tags
5. Outputs in Shopify-compatible format

Usage:
    python pattern_processor.py
"""

import json
import pandas as pd
import sys
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict
from datetime import datetime

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import Shopify column requirements
sys.path.append(str(project_root / "shared" / "data" / "product_import"))
try:
    exec(open(str(project_root / "shared" / "data" / "product_import" / "product_import-column-requirements.py")).read())
except Exception as e:
    print(f"Warning: Could not load product import requirements: {e}")
    # Fallback to minimal columns
    cols_list = ["Title", "Body HTML", "Vendor", "Tags", "Variant SKU", "Variant Price", "Variant Cost"]

class PatternProcessor:
    """
    Processes Steele data using pattern car ID mapping for tag assignment
    """
    
    def __init__(self):
        self.steele_dir = Path(__file__).parent
        self.pattern_mapping_path = self.steele_dir / "data" / "pattern_car_id_mapping.json"
        self.processed_data_path = self.steele_dir / "data" / "processed" / "steele_processed_complete.csv"
        self.vendor_name = "Steele Rubber Products"
        
        # Load pattern mapping
        self.pattern_mapping = self._load_pattern_mapping()
        print(f"Loaded pattern mapping with {len(self.pattern_mapping)} patterns")
        
    def _load_pattern_mapping(self) -> Dict[str, List[str]]:
        """Load the pattern car ID mapping from JSON file"""
        try:
            with open(self.pattern_mapping_path, 'r') as f:
                mapping = json.load(f)
            return mapping
        except Exception as e:
            print(f"Error loading pattern mapping: {e}")
            return {}
    
    def _create_year_make_model_key(self, row: pd.Series) -> str:
        """
        Create year_make_model key with underscores from CSV row
        
        Args:
            row: Pandas Series with Year, Make, Model columns
            
        Returns:
            String key like "1928_Stutz_Stutz"
        """
        year = str(int(row['Year'])) if pd.notna(row['Year']) else "Unknown"
        make = str(row['Make']).strip() if pd.notna(row['Make']) else "Unknown"
        model = str(row['Model']).strip() if pd.notna(row['Model']) else "Unknown"
        
        return f"{year}_{make}_{model}"
    
    def _lookup_tags_for_pattern(self, pattern_key: str) -> List[str]:
        """
        Look up tags for a given pattern key
        
        Args:
            pattern_key: Key like "1928_Stutz_Stutz"
            
        Returns:
            List of car ID tags, empty list if not found
        """
        return self.pattern_mapping.get(pattern_key, [])
    
    def _consolidate_products_by_sku(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Group products by StockCode and consolidate their data
        
        Args:
            df: DataFrame with pattern-tagged products
            
        Returns:
            Dictionary mapping StockCode to consolidated product data
        """
        consolidated = {}
        
        for stock_code, group in df.groupby('StockCode'):
            # Get first row for basic product info (all should be same for same SKU)
            first_row = group.iloc[0]
            
            # Consolidate all tags from all rows for this SKU
            all_tags = set()
            for _, row in group.iterrows():
                if pd.notna(row.get('tags', '')) and row['tags']:
                    # Split tags and add to set
                    row_tags = [tag.strip() for tag in str(row['tags']).split(',') if tag.strip()]
                    all_tags.update(row_tags)
            
            # Create consolidated product with safe data handling
            consolidated[stock_code] = {
                'stock_code': str(stock_code),
                'title': str(first_row['Product Name']) if pd.notna(first_row['Product Name']) else f"Product {stock_code}",
                'description': str(first_row['Description']) if pd.notna(first_row['Description']) else "",
                'price': float(first_row['MAP']) if pd.notna(first_row['MAP']) else 0.0,
                'cost': float(first_row['Dealer Price']) if pd.notna(first_row['Dealer Price']) else 0.0,
                'part_number': str(first_row['PartNumber']) if pd.notna(first_row['PartNumber']) else str(stock_code),
                'upc': str(first_row.get('UPC Code', '')) if pd.notna(first_row.get('UPC Code', '')) else '',
                'tags': list(all_tags),
                'vehicle_count': len(group)  # How many vehicle applications
            }
            
        return consolidated
    
    def _create_shopify_record(self, product_data: Dict) -> Dict[str, str]:
        """
        Create a complete Shopify import record from consolidated product data
        
        Args:
            product_data: Consolidated product information
            
        Returns:
            Dictionary with all 65 Shopify columns
        """
        # Join tags with commas
        tags_string = ', '.join(product_data['tags']) if product_data['tags'] else ''
        
        # Create record with all required columns
        record = {}
        
        for col in cols_list:
            if col == "ID":
                record[col] = ""  # Shopify auto-generates
            elif col == "Command":
                record[col] = "MERGE"
            elif col == "Title":
                record[col] = product_data['title']
            elif col == "Body HTML":
                record[col] = product_data['description']
            elif col == "Vendor":
                record[col] = self.vendor_name
            elif col == "Tags":
                record[col] = tags_string
            elif col == "Tags Command":
                record[col] = "MERGE"
            elif col == "Custom Collections":
                record[col] = "Automotive Parts"
            elif col == "Image Command":
                record[col] = "MERGE"
            elif col == "Image Position":
                record[col] = 1
            elif col == "Image Alt Text":
                record[col] = product_data['title']
            elif col == "Variant Command":
                record[col] = "MERGE"
            elif col == "Variant Position":
                record[col] = 1
            elif col == "Variant SKU":
                record[col] = product_data['stock_code']
            elif col == "Variant Price":
                record[col] = product_data['price']
            elif col == "Variant Cost":
                record[col] = product_data['cost']
            elif col == "Variant Taxable":
                record[col] = "TRUE"
            elif col == "Variant Requires Shipping":
                record[col] = "TRUE"
            elif col == "Metafield: title_tag [string]":
                title_str = str(product_data['title']) if product_data['title'] is not None else ""
                record[col] = title_str[:60]  # SEO title limit
            elif col == "Metafield: description_tag [string]":
                title_str = str(product_data['title']) if product_data['title'] is not None else ""
                record[col] = f"Quality {title_str} automotive part"[:160]  # SEO desc limit
            elif col == "Variant Metafield: mm-google-shopping.mpn [single_line_text_field]":
                record[col] = product_data['part_number']
            elif col == "Metafield: mm-google-shopping.mpn [single_line_text_field]":
                record[col] = product_data['part_number']
            else:
                # All other columns empty
                record[col] = ""
                
        return record
    
    def process_steele_data(self) -> pd.DataFrame:
        """
        Main processing function - loads data, applies pattern matching, consolidates by SKU
        
        Returns:
            DataFrame in Shopify import format with consolidated products and tags
        """
        print("Loading Steele processed data...")
        df = pd.read_csv(self.processed_data_path)
        print(f"Loaded {len(df)} rows")
        
        # Add pattern matching and tag lookup
        print("Applying pattern matching...")
        df['pattern_key'] = df.apply(self._create_year_make_model_key, axis=1)
        df['tags'] = df['pattern_key'].apply(self._lookup_tags_for_pattern)
        df['tags'] = df['tags'].apply(lambda x: ', '.join(x) if x else '')
        
        # Log pattern matching results
        matched_patterns = df[df['tags'] != '']['pattern_key'].unique()
        total_patterns = df['pattern_key'].unique()
        print(f"Pattern matching: {len(matched_patterns)}/{len(total_patterns)} patterns found tags")
        
        # Consolidate by StockCode
        print("Consolidating products by StockCode...")
        consolidated_products = self._consolidate_products_by_sku(df)
        print(f"Consolidated {len(df)} rows into {len(consolidated_products)} unique products")
        
        # Create Shopify records
        print("Creating Shopify format records...")
        shopify_records = []
        for stock_code, product_data in consolidated_products.items():
            record = self._create_shopify_record(product_data)
            shopify_records.append(record)
        
        # Convert to DataFrame
        result_df = pd.DataFrame(shopify_records)
        
        # Ensure column order matches Shopify requirements
        result_df = result_df.reindex(columns=cols_list, fill_value="")
        
        return result_df
    
    def save_results(self, df: pd.DataFrame, filename_prefix: str = "pattern_tagged") -> str:
        """
        Save results to CSV file
        
        Args:
            df: DataFrame to save
            filename_prefix: Prefix for output filename
            
        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.steele_dir / "data" / "results" / f"{filename_prefix}_{timestamp}.csv"
        
        # Ensure results directory exists
        output_path.parent.mkdir(exist_ok=True)
        
        df.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")
        
        return str(output_path)

def main():
    """Main execution function"""
    print("=== Steele Pattern-Based Tag Processor ===")
    
    processor = PatternProcessor()
    
    # Process the data
    result_df = processor.process_steele_data()
    
    # Save results
    output_path = processor.save_results(result_df, "pattern_tagged_shopify")
    
    # Print summary
    print(f"\n=== Processing Summary ===")
    print(f"Total products: {len(result_df)}")
    print(f"Products with tags: {len(result_df[result_df['Tags'] != ''])}")
    print(f"Output file: {output_path}")
    
    # Show sample of results
    print(f"\n=== Sample Results (first 5 products) ===")
    sample_cols = ['Title', 'Tags', 'Variant SKU', 'Variant Price']
    print(result_df[sample_cols].head())

if __name__ == "__main__":
    main()