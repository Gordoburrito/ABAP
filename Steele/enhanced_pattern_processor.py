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
    
    def process_two_stage_matching_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        SUPER-OPTIMIZED: Apply two-stage matching using vectorized operations
        
        Stage 1: Exact golden master matching (vectorized)
        Stage 2: Pattern mapping fallback (vectorized)
        
        This version is 10x-100x faster for large datasets.
        
        Returns:
            DataFrame with car_ids and match_type columns added
        """
        print("\n🚀 SUPER-OPTIMIZED TWO-STAGE MATCHING (VECTORIZED)")
        total_start = time.time()
        
        # Create pattern keys
        df['pattern_key'] = df.apply(self.create_year_make_model_key, axis=1)
        
        # Initialize result columns
        df['car_ids'] = None
        df['match_type'] = 'no_match'
        
        print("\n🔍 STAGE 1: VECTORIZED EXACT GOLDEN MASTER MATCHING")
        stage1_start = time.time()
        
        if self.golden_df is not None:
            print(f"   Processing {len(df):,} rows with vectorized operations...")
            
            # Create lookup keys for input data (vectorized)
            df['lookup_key'] = (
                df['Year'].fillna(0).astype(int).astype(str) + '_' +
                df['Make'].fillna('').astype(str).str.strip() + '_' +
                df['Model'].fillna('').astype(str).str.strip()
            )
            
            # Create golden dataset lookup dictionary (vectorized)
            golden_clean = self.golden_df.dropna(subset=['year', 'make', 'model'])
            golden_clean['lookup_key'] = (
                golden_clean['year'].astype(int).astype(str) + '_' +
                golden_clean['make'].astype(str).str.strip() + '_' +
                golden_clean['model'].astype(str).str.strip()
            )
            
            # Group golden data by lookup key
            golden_grouped = golden_clean.groupby('lookup_key')['car_id'].apply(list).to_dict()
            
            print(f"   ✅ Built vectorized lookup index with {len(golden_grouped):,} unique combinations")
            
            # Vectorized matching using map operation
            df['exact_match'] = df['lookup_key'].map(golden_grouped)
            
            # Update results for successful matches
            exact_matches_mask = df['exact_match'].notna()
            df.loc[exact_matches_mask, 'car_ids'] = df.loc[exact_matches_mask, 'exact_match']
            df.loc[exact_matches_mask, 'match_type'] = 'exact'
            
            # Clean up temporary columns
            df = df.drop(['exact_match'], axis=1)
            
            exact_match_count = exact_matches_mask.sum()
            stage1_time = time.time() - stage1_start
            print(f"   ✅ Stage 1 complete in {stage1_time:.1f}s - Exact matches: {exact_match_count:,}/{len(df):,} rows ({exact_match_count/len(df)*100:.1f}%)")
            self.stats['exact_matches'] = exact_match_count
        else:
            print("   ⚠️  No golden dataset loaded - Stage 1 skipped")
            exact_match_count = 0
        
        print("\n🔍 STAGE 2: VECTORIZED PATTERN MAPPING FALLBACK")
        stage2_start = time.time()
        
        # Stage 2: Vectorized pattern mapping
        if self.pattern_mapping:
            unmatched_mask = df['match_type'] == 'no_match'
            unmatched_count = unmatched_mask.sum()
            
            if unmatched_count > 0:
                print(f"   Processing {unmatched_count:,} unmatched rows with vectorized operations...")
                
                # Vectorized pattern matching using map operation
                df.loc[unmatched_mask, 'pattern_match'] = df.loc[unmatched_mask, 'pattern_key'].map(self.pattern_mapping)
                
                # Update results for successful pattern matches
                pattern_matches_mask = unmatched_mask & df['pattern_match'].notna()
                df.loc[pattern_matches_mask, 'car_ids'] = df.loc[pattern_matches_mask, 'pattern_match']
                df.loc[pattern_matches_mask, 'match_type'] = 'pattern'
                
                # Clean up temporary column
                df = df.drop(['pattern_match'], axis=1, errors='ignore')
                
                pattern_match_count = pattern_matches_mask.sum()
                stage2_time = time.time() - stage2_start
                print(f"   ✅ Stage 2 complete in {stage2_time:.1f}s - Pattern matches: {pattern_match_count:,}/{unmatched_count:,} rows ({pattern_match_count/unmatched_count*100:.1f}%)")
            else:
                print("   ✅ No unmatched rows - Stage 2 skipped")
                pattern_match_count = 0
                stage2_time = 0
        else:
            print("   ⚠️  No pattern mapping loaded - Stage 2 skipped")
            pattern_match_count = 0
            stage2_time = 0
        
        self.stats['pattern_matches'] = pattern_match_count
        
        # Clean up temporary columns
        df = df.drop(['lookup_key'], axis=1, errors='ignore')
        
        # Calculate final statistics
        total_time = time.time() - total_start
        total_matches = exact_match_count + pattern_match_count
        no_matches = len(df) - total_matches
        
        print("\n📊 VECTORIZED MATCHING SUMMARY:")
        print(f"   🎯 Exact matches: {exact_match_count} ({exact_match_count/len(df)*100:.1f}%)")
        print(f"   🗺️  Pattern matches: {pattern_match_count} ({pattern_match_count/len(df)*100:.1f}%)")
        print(f"   ❌ No matches: {no_matches} ({no_matches/len(df)*100:.1f}%)")
        print(f"   ✅ Total coverage: {total_matches}/{len(df)} ({total_matches/len(df)*100:.1f}%)")
        print(f"   ⚡ Total time: {total_time:.1f}s (Stage 1: {stage1_time:.1f}s, Stage 2: {stage2_time:.1f}s)")
        
        self.stats['no_matches'] = no_matches
        
        return df
        
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
        
        # Stage 1: Optimized exact matching with progress tracking
        print(f"   Processing {len(df):,} rows...")
        start_time = time.time()
        
        # Prepare golden dataset lookup for fast matching
        if self.golden_df is not None:
            # Create a lookup dictionary for much faster matching
            golden_lookup = {}
            for _, row in self.golden_df.iterrows():
                try:
                    year = int(row['year']) if pd.notna(row['year']) else 0
                    make = str(row['make']).strip() if pd.notna(row['make']) else ""
                    model = str(row['model']).strip() if pd.notna(row['model']) else ""
                    
                    # Skip rows with invalid data
                    if year == 0 or not make or not model:
                        continue
                        
                    key = f"{year}_{make}_{model}"
                    if key not in golden_lookup:
                        golden_lookup[key] = []
                    golden_lookup[key].append(row['car_id'])
                except:
                    continue
            
            print(f"   ✅ Built lookup index with {len(golden_lookup):,} unique combinations")
        
        exact_match_count = 0
        total_rows = len(df)
        batch_size = max(1000, total_rows // 100)  # Show progress every 1% or 1000 rows
        
        # Process in batches with progress tracking
        for i in range(0, total_rows, batch_size):
            batch_end = min(i + batch_size, total_rows)
            batch_df = df.iloc[i:batch_end]
            
            # Show progress
            progress = (batch_end / total_rows) * 100
            elapsed = time.time() - start_time
            if elapsed > 0 and i > 0:
                rate = i / elapsed  # rows per second
                remaining_rows = total_rows - batch_end
                eta_seconds = remaining_rows / rate if rate > 0 else 0
                eta_mins = int(eta_seconds // 60)
                eta_secs = int(eta_seconds % 60)
                print(f"   🔄 Progress: {progress:.1f}% ({batch_end:,}/{total_rows:,}) - ETA: {eta_mins}m {eta_secs}s")
            else:
                print(f"   🔄 Progress: {progress:.1f}% ({batch_end:,}/{total_rows:,})")
            
            # Process current batch
            for idx in batch_df.index:
                try:
                    row = df.loc[idx]
                    year = int(row['Year']) if pd.notna(row['Year']) else 0
                    make = str(row['Make']).strip() if pd.notna(row['Make']) else ""
                    model = str(row['Model']).strip() if pd.notna(row['Model']) else ""
                    
                    # Fast lookup
                    lookup_key = f"{year}_{make}_{model}"
                    if lookup_key in golden_lookup:
                        df.at[idx, 'car_ids'] = golden_lookup[lookup_key]
                        df.at[idx, 'match_type'] = 'exact'
                        exact_match_count += 1
                        
                except Exception:
                    continue
        
        elapsed_total = time.time() - start_time
        print(f"   ✅ Stage 1 complete in {elapsed_total:.1f}s - Exact matches: {exact_match_count:,}/{len(df):,} rows ({exact_match_count/len(df)*100:.1f}%)")
        self.stats['exact_matches'] = exact_match_count
        
        print("\n🔍 STAGE 2: PATTERN MAPPING FALLBACK")
        
        # Stage 2: Optimized pattern mapping for unmatched rows with progress tracking
        pattern_match_count = 0
        unmatched_df = df[df['match_type'] == 'no_match']
        unmatched_count = len(unmatched_df)
        
        if unmatched_count > 0:
            print(f"   Processing {unmatched_count:,} unmatched rows...")
            stage2_start = time.time()
            
            # Process in batches with progress tracking
            batch_size = max(500, unmatched_count // 50)  # Smaller batches for Stage 2
            
            for i, idx in enumerate(unmatched_df.index):
                # Show progress every batch
                if i % batch_size == 0:
                    progress = (i / unmatched_count) * 100
                    elapsed = time.time() - stage2_start
                    if elapsed > 0 and i > 0:
                        rate = i / elapsed
                        remaining = unmatched_count - i
                        eta_seconds = remaining / rate if rate > 0 else 0
                        eta_mins = int(eta_seconds // 60)
                        eta_secs = int(eta_seconds % 60)
                        print(f"   🔄 Progress: {progress:.1f}% ({i:,}/{unmatched_count:,}) - ETA: {eta_mins}m {eta_secs}s")
                    else:
                        print(f"   🔄 Progress: {progress:.1f}% ({i:,}/{unmatched_count:,})")
                
                try:
                    row = df.loc[idx]
                    pattern_car_ids = self.find_pattern_matches(row['pattern_key'])
                    
                    if pattern_car_ids:
                        df.at[idx, 'car_ids'] = pattern_car_ids
                        df.at[idx, 'match_type'] = 'pattern'
                        pattern_match_count += 1
                except Exception:
                    continue
            
            elapsed_stage2 = time.time() - stage2_start
            print(f"   ✅ Stage 2 complete in {elapsed_stage2:.1f}s - Pattern matches: {pattern_match_count:,}/{unmatched_count:,} rows ({pattern_match_count/unmatched_count*100:.1f}%)")
        else:
            print("   ✅ No unmatched rows - Stage 2 skipped")
        
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
        matched_df = self.process_two_stage_matching_vectorized(df)
        
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
    """
    Main entry point for enhanced pattern processor.
    
    Usage:
        python enhanced_pattern_processor.py [input_file] [--mode=vectorized|progressive] [--benchmark]
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Enhanced Pattern Processor for Steele Data",
        epilog="""
Processing Modes:
  vectorized  : Super-fast vectorized operations (recommended for large datasets)
  progressive : Detailed progress tracking (useful for monitoring long processes)

Examples:
  python enhanced_pattern_processor.py data/samples/steele_test_1000.csv
  python enhanced_pattern_processor.py data/processed/steele_processed_complete.csv --mode vectorized
  python enhanced_pattern_processor.py data/samples/steele_test_1000.csv --mode progressive
  python enhanced_pattern_processor.py data/samples/steele_test_1000.csv --benchmark
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('input_file', nargs='?', 
                       default='data/samples/steele_test_1000.csv',
                       help='Input CSV file path (default: data/samples/steele_test_1000.csv)')
    
    parser.add_argument('--mode', choices=['vectorized', 'progressive'], 
                       default='vectorized',
                       help='Processing mode: vectorized (fastest) or progressive (with progress tracking)')
    
    parser.add_argument('--benchmark', action='store_true',
                       help='Run both modes and compare performance')
    
    args = parser.parse_args()
    
    if args.benchmark:
        print("🏁 BENCHMARK MODE: Comparing vectorized vs progressive processing")
        print("=" * 80)
        
        # Test vectorized mode
        print("\n🚀 TESTING VECTORIZED MODE:")
        processor_v = EnhancedPatternProcessor()
        start_v = time.time()
        try:
            output_path_v, final_df_v = processor_v.process_file(args.input_file)
            time_v = time.time() - start_v
            print(f"✅ Vectorized mode completed in {time_v:.2f}s")
        except Exception as e:
            print(f"❌ Vectorized mode failed: {e}")
            time_v = float('inf')
        
        # Test progressive mode  
        print("\n🔄 TESTING PROGRESSIVE MODE:")
        processor_p = EnhancedPatternProcessor()
        # Temporarily switch to progressive mode
        original_method = processor_p.process_two_stage_matching_vectorized
        processor_p.process_two_stage_matching_vectorized = processor_p.process_two_stage_matching
        
        start_p = time.time()
        try:
            output_path_p, final_df_p = processor_p.process_file(args.input_file)
            time_p = time.time() - start_p
            print(f"✅ Progressive mode completed in {time_p:.2f}s")
        except Exception as e:
            print(f"❌ Progressive mode failed: {e}")
            time_p = float('inf')
        
        # Show comparison
        print("\n📊 BENCHMARK RESULTS:")
        print(f"   🚀 Vectorized mode: {time_v:.2f}s")
        print(f"   🔄 Progressive mode: {time_p:.2f}s")
        if time_p != float('inf') and time_v != float('inf'):
            speedup = time_p / time_v
            print(f"   ⚡ Speedup: {speedup:.1f}x faster with vectorized mode")
        
        return
    
    # Normal processing
    processor = EnhancedPatternProcessor()
    
    # Select processing mode
    if args.mode == 'progressive':
        # Use progressive mode with detailed progress tracking
        processor.process_two_stage_matching_vectorized = processor.process_two_stage_matching
        print(f"🔄 Using PROGRESSIVE mode (detailed progress tracking)")
    else:
        print(f"🚀 Using VECTORIZED mode (super-fast processing)")
    
    try:
        output_path, final_df = processor.process_file(args.input_file)
        print(f"\n🎉 SUCCESS: Enhanced processing completed!")
        print(f"📁 Output: {output_path}")
        
    except FileNotFoundError as e:
        print(f"❌ ERROR: {e}")
        print(f"💡 Make sure the input file exists: {args.input_file}")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()