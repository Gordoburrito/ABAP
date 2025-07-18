#!/usr/bin/env python3
"""
Enhanced Pattern Processor for Steele Data

This processor implements a two-stage matching system:
1. Stage 1: Exact matches against golden master dataset
2. Stage 2: Pattern mapping fallback for previously AI-matched patterns

This provides comprehensive coverage without requiring AI processing.
"""

import json
import pandas as pd
import sys
import time
import os
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import Shopify column requirements and utilities
sys.path.append(str(project_root / "shared" / "data" / "product_import"))
try:
    exec(open(str(project_root / "shared" / "data" / "product_import" / "product_import-column-requirements.py")).read())
except Exception as e:
    print(f"Warning: Could not load product import requirements: {e}")
    cols_list = ["Title", "Body HTML", "Vendor", "Tags", "Variant SKU", "Variant Price", "Variant Cost"]

# Import existing transformer components
from utils.optimized_batch_steele_transformer import OptimizedBatchSteeleTransformer, ProductData

class EnhancedPatternProcessor:
    """
    Enhanced pattern processor with two-stage matching:
    1. Exact golden master matching
    2. Pattern mapping fallback
    """
    
    def __init__(self):
        self.steele_dir = Path(__file__).parent
        self.pattern_mapping_path = self.steele_dir / "data" / "pattern_car_id_mapping.json"
        self.vendor_name = "Steele Rubber Products"
        
        # Initialize golden dataset
        self.golden_df = None
        self.pattern_mapping = None
        
        # Statistics tracking
        self.stats = {
            'total_rows': 0,
            'unique_products': 0,
            'exact_matches': 0,
            'pattern_matches': 0,
            'no_matches': 0,
            'exact_match_products': 0,
            'pattern_match_products': 0,
            'no_match_products': 0
        }
        
    def load_golden_dataset(self) -> pd.DataFrame:
        """Load golden master dataset for exact matching"""
        golden_path = project_root / "shared" / "data" / "master_ultimate_golden.csv"
        
        if not golden_path.exists():
            raise FileNotFoundError(f"Golden dataset not found: {golden_path}")
        
        try:
            # Load only essential columns for efficiency
            self.golden_df = pd.read_csv(golden_path)
            
            # Standardize column names if needed
            column_mapping = {
                'Year': 'year',
                'Make': 'make', 
                'Model': 'model',
                'Car ID': 'car_id'
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in self.golden_df.columns:
                    self.golden_df = self.golden_df.rename(columns={old_col: new_col})
            
            print(f"✅ Loaded golden dataset: {len(self.golden_df):,} records")
            return self.golden_df
            
        except Exception as e:
            raise ValueError(f"Error loading golden dataset: {str(e)}")
    
    def load_pattern_mapping(self) -> Dict:
        """Load pattern car ID mapping from JSON file"""
        if not self.pattern_mapping_path.exists():
            print(f"⚠️  Pattern mapping file not found: {self.pattern_mapping_path}")
            print("   Pattern mapping will be skipped")
            return {}
            
        try:
            with open(self.pattern_mapping_path, 'r') as f:
                self.pattern_mapping = json.load(f)
            print(f"✅ Loaded pattern mapping: {len(self.pattern_mapping):,} patterns")
            return self.pattern_mapping
            
        except Exception as e:
            print(f"⚠️  Error loading pattern mapping: {e}")
            return {}
    
    def create_year_make_model_key(self, row: pd.Series) -> str:
        """Create standardized year_make_model key"""
        year = str(int(row['Year'])) if pd.notna(row['Year']) else "Unknown"
        make = str(row['Make']).strip() if pd.notna(row['Make']) else "Unknown"
        model = str(row['Model']).strip() if pd.notna(row['Model']) else "Unknown"
        return f"{year}_{make}_{model}"
    
    def find_exact_matches(self, year: int, make: str, model: str) -> List[str]:
        """
        Stage 1: Find exact matches in golden master dataset
        
        Returns:
            List of car_ids for exact matches
        """
        if self.golden_df is None:
            return []
        
        try:
            exact_matches = self.golden_df[
                (self.golden_df['year'] == year) &
                (self.golden_df['make'] == make) &
                (self.golden_df['model'] == model)
            ]
            
            if len(exact_matches) > 0:
                return exact_matches['car_id'].unique().tolist()
            else:
                return []
                
        except Exception:
            return []
    
    def find_pattern_matches(self, pattern_key: str) -> List[str]:
        """
        Stage 2: Find pattern matches from pattern mapping
        
        Returns:
            List of car_ids from pattern mapping
        """
        if not self.pattern_mapping:
            return []
        
        return self.pattern_mapping.get(pattern_key, [])
    
    def process_two_stage_matching(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply two-stage matching to DataFrame
        
        Stage 1: Exact golden master matching
        Stage 2: Pattern mapping fallback
        
        Returns:
            DataFrame with car_ids and match_type columns added
        """
        print("\n🔍 STAGE 1: EXACT GOLDEN MASTER MATCHING")
        
        # Create pattern keys
        df['pattern_key'] = df.apply(self.create_year_make_model_key, axis=1)
        
        # Initialize result columns
        df['car_ids'] = None
        df['match_type'] = 'no_match'
        
        # Stage 1: Exact matching
        exact_match_count = 0
        for idx, row in df.iterrows():
            try:
                year = int(row['Year']) if pd.notna(row['Year']) else 0
                make = str(row['Make']).strip() if pd.notna(row['Make']) else ""
                model = str(row['Model']).strip() if pd.notna(row['Model']) else ""
                
                exact_car_ids = self.find_exact_matches(year, make, model)
                
                if exact_car_ids:
                    df.at[idx, 'car_ids'] = exact_car_ids
                    df.at[idx, 'match_type'] = 'exact'
                    exact_match_count += 1
                    
            except Exception:
                continue
        
        print(f"   ✅ Exact matches found: {exact_match_count:,}/{len(df):,} rows")
        self.stats['exact_matches'] = exact_match_count
        
        print("\n🔍 STAGE 2: PATTERN MAPPING FALLBACK")
        
        # Stage 2: Pattern mapping for unmatched rows
        pattern_match_count = 0
        unmatched_df = df[df['match_type'] == 'no_match']
        
        for idx, row in unmatched_df.iterrows():
            pattern_car_ids = self.find_pattern_matches(row['pattern_key'])
            
            if pattern_car_ids:
                df.at[idx, 'car_ids'] = pattern_car_ids
                df.at[idx, 'match_type'] = 'pattern'
                pattern_match_count += 1
        
        print(f"   ✅ Pattern matches found: {pattern_match_count:,}/{len(unmatched_df):,} unmatched rows")
        self.stats['pattern_matches'] = pattern_match_count
        
        # Calculate final statistics
        total_matched = exact_match_count + pattern_match_count
        no_match_count = len(df) - total_matched
        self.stats['no_matches'] = no_match_count
        
        print(f"\n📊 MATCHING SUMMARY:")
        print(f"   🎯 Exact matches: {exact_match_count:,} ({exact_match_count/len(df)*100:.1f}%)")
        print(f"   🗺️  Pattern matches: {pattern_match_count:,} ({pattern_match_count/len(df)*100:.1f}%)")
        print(f"   ❌ No matches: {no_match_count:,} ({no_match_count/len(df)*100:.1f}%)")
        print(f"   ✅ Total coverage: {total_matched:,}/{len(df):,} ({total_matched/len(df)*100:.1f}%)")
        
        return df
    
    def consolidate_products_by_sku(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Group products by StockCode and consolidate car_ids
        
        Returns:
            Dictionary mapping StockCode to consolidated product data
        """
        consolidated_products = {}
        
        for stock_code, group in df.groupby('StockCode'):
            first_row = group.iloc[0]
            
            # Consolidate all car_ids for this SKU
            all_car_ids = set()
            match_types = set()
            
            for _, row in group.iterrows():
                car_ids = row.get('car_ids')
                if car_ids is not None and len(car_ids) > 0:
                    if isinstance(car_ids, list):
                        all_car_ids.update(car_ids)
                    match_types.add(row.get('match_type', 'unknown'))
            
            # Determine primary match type
            if 'exact' in match_types:
                primary_match_type = 'exact'
            elif 'pattern' in match_types:
                primary_match_type = 'pattern'
            else:
                primary_match_type = 'no_match'
            
            # Create consolidated product data
            consolidated_products[stock_code] = {
                'stock_code': str(stock_code),
                'title': str(first_row['Product Name']) if pd.notna(first_row['Product Name']) else f"Product {stock_code}",
                'description': str(first_row['Description']) if pd.notna(first_row['Description']) else "",
                'year': str(int(first_row['Year'])) if pd.notna(first_row['Year']) else "1800",
                'make': str(first_row['Make']) if pd.notna(first_row['Make']) else "NONE",
                'model': str(first_row['Model']) if pd.notna(first_row['Model']) else "NONE",
                'price': float(first_row['MAP']) if pd.notna(first_row['MAP']) else 0.0,
                'cost': float(first_row['Dealer Price']) if pd.notna(first_row['Dealer Price']) else 0.0,
                'part_number': str(first_row['PartNumber']) if pd.notna(first_row['PartNumber']) else str(stock_code),
                'car_ids': list(all_car_ids),
                'golden_validated': len(all_car_ids) > 0,
                'vehicle_count': len(group),
                'match_type': primary_match_type,
                'fitment_source': "golden_master" if primary_match_type == "exact" else "pattern_mapping",
                'processing_method': "enhanced_pattern_based"
            }
            
            # Update product-level statistics
            if len(all_car_ids) > 0:
                if primary_match_type == 'exact':
                    self.stats['exact_match_products'] += 1
                elif primary_match_type == 'pattern':
                    self.stats['pattern_match_products'] += 1
            else:
                self.stats['no_match_products'] += 1
        
        self.stats['unique_products'] = len(consolidated_products)
        
        return consolidated_products
    
    def create_shopify_format(self, consolidated_products: Dict[str, Dict]) -> pd.DataFrame:
        """
        Convert consolidated products to Shopify import format
        
        Returns:
            DataFrame in complete 65-column Shopify format
        """
        # Initialize transformer for final formatting
        transformer = OptimizedBatchSteeleTransformer(use_ai=False)
        
        # Convert consolidated products to ProductData objects
        standard_products = []
        for stock_code, product_data in consolidated_products.items():
            product = ProductData(
                title=product_data['title'],
                year_min=product_data['year'],
                year_max=product_data['year'],
                make=product_data['make'],
                model=product_data['model'],
                mpn=product_data['part_number'],
                cost=product_data['cost'],
                price=product_data['price'],
                body_html=product_data['description'],
                car_ids=product_data['car_ids'],
                golden_validated=product_data['golden_validated'],
                fitment_source=product_data['fitment_source'],
                processing_method=product_data['processing_method']
            )
            standard_products.append(product)
        
        # Apply template enhancements
        enhanced_products = transformer.enhance_with_templates(standard_products)
        
        # Convert to final Shopify format
        final_df = transformer.transform_to_formatted_shopify_import(enhanced_products)
        
        return final_df
    
    def process_file(self, input_file: str) -> Tuple[str, pd.DataFrame]:
        """
        Main processing function for enhanced pattern matching
        
        Args:
            input_file: Path to input CSV file
            
        Returns:
            Tuple of (output_file_path, final_dataframe)
        """
        print("=" * 80)
        print("🚀 ENHANCED STEELE PATTERN PROCESSOR")
        print("   Two-stage matching: Golden Master + Pattern Mapping")
        print(f"   Input file: {input_file}")
        print("=" * 80)
        
        # Phase 1: Load datasets
        print("\n📁 PHASE 1: LOADING DATASETS")
        
        input_path = self.steele_dir / input_file
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        df = pd.read_csv(input_path)
        self.stats['total_rows'] = len(df)
        print(f"✅ Loaded input data: {len(df):,} rows")
        
        # Load golden dataset and pattern mapping
        self.load_golden_dataset()
        self.load_pattern_mapping()
        
        # Phase 2: Two-stage matching
        print("\n🔍 PHASE 2: TWO-STAGE MATCHING")
        matched_df = self.process_two_stage_matching(df)
        
        # Phase 3: Product consolidation
        print("\n🔄 PHASE 3: PRODUCT CONSOLIDATION BY SKU")
        consolidated_products = self.consolidate_products_by_sku(matched_df)
        print(f"✅ Consolidated {len(df):,} rows into {len(consolidated_products):,} unique products")
        
        # Phase 4: Shopify format transformation
        print("\n🏭 PHASE 4: SHOPIFY FORMAT TRANSFORMATION")
        final_df = self.create_shopify_format(consolidated_products)
        print(f"✅ Generated Shopify format: {len(final_df):,} products")
        
        # Phase 5: Save results
        print("\n💾 PHASE 5: SAVING RESULTS")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        input_filename = Path(input_file).stem
        output_filename = f"enhanced_pattern_{input_filename}_{timestamp}.csv"
        results_path = self.steele_dir / "data" / "results" / output_filename
        
        # Ensure results directory exists
        os.makedirs(results_path.parent, exist_ok=True)
        
        final_df.to_csv(results_path, index=False)
        
        # Phase 6: Final summary
        print(f"\n✅ ENHANCED PATTERN PROCESSING COMPLETE!")
        print(f"📁 Results saved to: {results_path}")
        
        self.print_final_statistics()
        
        return str(results_path), final_df
    
    def print_final_statistics(self):
        """Print comprehensive processing statistics"""
        print(f"\n📊 PROCESSING STATISTICS:")
        print(f"   📄 Total input rows: {self.stats['total_rows']:,}")
        print(f"   📦 Unique products: {self.stats['unique_products']:,}")
        print("")
        print(f"   🎯 Exact matches (rows): {self.stats['exact_matches']:,}")
        print(f"   🗺️  Pattern matches (rows): {self.stats['pattern_matches']:,}")
        print(f"   ❌ No matches (rows): {self.stats['no_matches']:,}")
        print("")
        print(f"   ✅ Products with exact matches: {self.stats['exact_match_products']:,}")
        print(f"   ✅ Products with pattern matches: {self.stats['pattern_match_products']:,}")
        print(f"   ❌ Products with no matches: {self.stats['no_match_products']:,}")
        print("")
        
        total_matched_rows = self.stats['exact_matches'] + self.stats['pattern_matches']
        total_matched_products = self.stats['exact_match_products'] + self.stats['pattern_match_products']
        
        if self.stats['total_rows'] > 0:
            row_coverage = total_matched_rows / self.stats['total_rows'] * 100
            print(f"   📈 Row coverage: {total_matched_rows:,}/{self.stats['total_rows']:,} ({row_coverage:.1f}%)")
        
        if self.stats['unique_products'] > 0:
            product_coverage = total_matched_products / self.stats['unique_products'] * 100
            print(f"   📈 Product coverage: {total_matched_products:,}/{self.stats['unique_products']:,} ({product_coverage:.1f}%)")
        
        if total_matched_products > 0:
            total_tags = self.stats['exact_matches'] + self.stats['pattern_matches']
            avg_tags = total_tags / total_matched_products
            print(f"   🏷️  Average tags per matched product: {avg_tags:.1f}")

def main():
    """Main execution function"""
    if len(sys.argv) < 2:
        print("Usage: python enhanced_pattern_processor.py <input_file>")
        print("Example: python enhanced_pattern_processor.py data/samples/steele_test_1000.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    try:
        processor = EnhancedPatternProcessor()
        output_path, final_df = processor.process_file(input_file)
        print(f"\n🎉 SUCCESS: Enhanced processing completed!")
        print(f"📁 Output: {output_path}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()