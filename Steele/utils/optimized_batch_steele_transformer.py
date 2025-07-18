import pandas as pd
import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel
from collections import defaultdict
from .batch_ai_vehicle_matcher import BatchAIVehicleMatcher

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
steele_root = Path(__file__).parent.parent  # Steele directory
sys.path.append(str(project_root))

# Import product import requirements
sys.path.append(str(project_root / "shared" / "data" / "product_import"))
try:
    # Try importing with the correct filename
    exec(open(str(project_root / "shared" / "data" / "product_import" / "product_import-column-requirements.py")).read())
    # Now cols_list and other variables are available
except Exception as e:
    print(f"Warning: Could not load product import requirements: {e}")
    # Fallback to minimal columns
    cols_list = ["Title", "Body HTML", "Vendor", "Tags", "Variant SKU", "Variant Price", "Variant Cost"]

class ProductData(BaseModel):
    """Standard format for complete fitment data sources"""
    title: str
    year_min: str = "1800"
    year_max: str = "1800"
    make: str = "NONE"
    model: str = "NONE"
    mpn: str = ""
    cost: float = 0.0
    price: float = 0.0
    body_html: str = ""
    collection: str = "Accessories"
    product_type: str = "Automotive Part"
    meta_title: str = ""
    meta_description: str = ""
    car_ids: List[str] = []
    
    # Validation flags
    golden_validated: bool = False
    fitment_source: str = "vendor_provided"
    processing_method: str = "template_based"

class TemplateGenerator:
    """Template-based enhancement for complete fitment data sources"""
    
    def generate_meta_title(self, product_name: str, year: str, make: str, model: str) -> str:
        """Generate SEO meta title using template"""
        template = f"{product_name} - {year} {make} {model}"
        return template[:60] if len(template) > 60 else template
    
    def generate_meta_description(self, product_name: str, year: str, make: str, model: str) -> str:
        """Generate SEO meta description using template"""
        template = f"Quality {product_name} for {year} {make} {model} vehicles. OEM replacement part."
        return template[:160] if len(template) > 160 else template
    
    def categorize_product(self, product_name: str) -> str:
        """Rule-based product categorization"""
        name_lower = product_name.lower()
        
        # Engine category
        if any(word in name_lower for word in ['engine', 'motor', 'piston', 'cylinder', 'valve', 'camshaft']):
            return 'Engine'
        
        # Brakes category
        elif any(word in name_lower for word in ['brake', 'pad', 'rotor', 'caliper', 'disc']):
            return 'Brakes'
        
        # Suspension category
        elif any(word in name_lower for word in ['shock', 'strut', 'spring', 'suspension']):
            return 'Suspension'
        
        # Lighting category
        elif any(word in name_lower for word in ['light', 'lamp', 'bulb', 'headlight', 'taillight']):
            return 'Lighting'
        
        # Electrical category
        elif any(word in name_lower for word in ['electrical', 'wire', 'fuse', 'relay', 'switch']):
            return 'Electrical'
        
        # Body category
        elif any(word in name_lower for word in ['door', 'window', 'mirror', 'bumper', 'fender']):
            return 'Body'
        
        # Default
        else:
            return 'Accessories'

class OptimizedBatchSteeleTransformer:
    """
    Ultra-optimized batch transformer that uses pattern deduplication
    to eliminate the 2+ hour queuing bottleneck.
    """
    
    def __init__(self, use_ai: bool = True):
        """
        Initialize the optimized batch transformer.
        
        Args:
            use_ai: Whether to use AI for edge cases
        """
        self.use_ai = use_ai
        self.vendor_name = "Steele"
        self.golden_df = None
        self.template_generator = TemplateGenerator()
        self.batch_ai_matcher = BatchAIVehicleMatcher(use_ai=use_ai)
        
        # Pattern optimization state
        self.pattern_mapping = {}
        self.pattern_results = {}
        self.optimization_stats = {
            'total_products': 0,
            'unique_patterns': 0,
            'deduplication_factor': 0,
            'ai_calls_saved': 0
        }
        
        print("🚀 Initialized Optimized Batch Steele Data Transformer")
        if use_ai:
            print("   ✅ Ultra-fast pattern deduplication enabled")
            print("   ✅ 50% batch API cost savings")
        else:
            print("   ⚠️  AI processing disabled")
    
    def load_sample_data(self, file_path: str = "data/samples/steele_sample.csv") -> pd.DataFrame:
        """Load Steele sample data for processing."""
        # Convert relative path to absolute path from Steele directory
        if not os.path.isabs(file_path):
            file_path = str(steele_root / file_path)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Sample data file not found: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            self._validate_input_data(df)
            return df
        except Exception as e:
            raise ValueError(f"Error loading sample data: {str(e)}")
    
    def load_golden_dataset(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """Load golden master dataset for vehicle validation."""
        if file_path is None:
            # Try shared data path
            shared_path = project_root / "shared" / "data" / "master_ultimate_golden.csv"
            if shared_path.exists():
                file_path = str(shared_path)
            else:
                raise FileNotFoundError("Golden dataset not found in shared/data/")
        
        try:
            # Load only essential columns for efficiency
            golden_df = pd.read_csv(
                file_path, 
                usecols=['Year', 'Make', 'Model', 'Car ID'] if 'Year' in pd.read_csv(file_path, nrows=1).columns 
                else ['year', 'make', 'model', 'car_id']
            )
            
            # Standardize column names
            column_mapping = {
                'Year': 'year',
                'Make': 'make', 
                'Model': 'model',
                'Car ID': 'car_id'
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in golden_df.columns:
                    golden_df = golden_df.rename(columns={old_col: new_col})
            
            self.golden_df = golden_df
            return golden_df
            
        except Exception as e:
            raise ValueError(f"Error loading golden dataset: {str(e)}")
    
    def analyze_steele_patterns(self, steele_df: pd.DataFrame) -> Tuple[Dict, int]:
        """
        Phase 1: Analyze data patterns and calculate deduplication potential.
        
        Args:
            steele_df: Steele data to analyze
            
        Returns:
            Tuple of (pattern_groups, unique_pattern_count)
        """
        print("🔍 Phase 1: Analyzing data patterns...")
        
        # Count unique year/make/model combinations
        pattern_groups = steele_df.groupby(['Year', 'Make', 'Model'])
        unique_patterns = len(pattern_groups)
        
        # Calculate deduplication benefit
        total_products = len(steele_df)
        reduction_factor = total_products / unique_patterns if unique_patterns > 0 else 1
        
        print(f"   📊 Total products: {total_products:,}")
        print(f"   📊 Unique patterns: {unique_patterns:,}")
        print(f"   📊 Deduplication factor: {reduction_factor:.1f}x")
        print(f"   📊 AI calls saved: {total_products - unique_patterns:,}")
        
        # Store optimization stats
        self.optimization_stats = {
            'total_products': total_products,
            'unique_patterns': unique_patterns,
            'deduplication_factor': reduction_factor,
            'ai_calls_saved': total_products - unique_patterns
        }
        
        return pattern_groups, unique_patterns
    
    def create_pattern_mapping(self, steele_df: pd.DataFrame, pattern_groups) -> Dict:
        """
        Phase 2: Create mapping from unique patterns to all products that use them.
        
        Args:
            steele_df: Original Steele data
            pattern_groups: Grouped data by year/make/model
            
        Returns:
            Dictionary mapping pattern keys to product information
        """
        print("🔄 Phase 2: Creating pattern mapping...")
        
        pattern_mapping = {}
        
        for (year, make, model), group in pattern_groups:
            pattern_key = f"{year}_{make}_{model}"
            pattern_mapping[pattern_key] = {
                'year': year,
                'make': make,
                'model': model,
                'products': group.index.tolist(),
                'count': len(group),
                'sample_product': group.iloc[0].to_dict()
            }
        
        print(f"   ✅ Created {len(pattern_mapping)} pattern mappings")
        self.pattern_mapping = pattern_mapping
        return pattern_mapping
    
    def save_pattern_mapping(self, file_path: str = "data/pattern_mapping.json") -> None:
        """Save pattern mapping to JSON file."""
        if not self.pattern_mapping:
            return
        
        with open(file_path, 'w') as f:
            json.dump(self.pattern_mapping, f, indent=2, default=str)
        
        print(f"✅ Pattern mapping saved to: {file_path}")
    
    def validate_patterns_against_golden(self, pattern_mapping: Dict) -> Tuple[List, List]:
        """
        Phase 3: Validate patterns against golden dataset and separate exact matches from AI-needed patterns.
        
        Args:
            pattern_mapping: Dictionary of pattern mappings
            
        Returns:
            Tuple of (exact_match_patterns, ai_needed_patterns)
        """
        print("🔄 Phase 3: Validating patterns against golden dataset...")
        
        if self.golden_df is None:
            self.load_golden_dataset()
        
        exact_match_patterns = []
        ai_needed_patterns = []
        
        for pattern_key, pattern_data in pattern_mapping.items():
            year = pattern_data['year']
            make = pattern_data['make']
            model = pattern_data['model']
            
            # Try exact match first
            exact_matches = self.golden_df[
                (self.golden_df['year'] == year) &
                (self.golden_df['make'] == make) &
                (self.golden_df['model'] == model)
            ]
            
            if len(exact_matches) > 0:
                # Exact match found - no AI needed
                pattern_data['validation_result'] = {
                    'golden_validated': True,
                    'golden_matches': len(exact_matches),
                    'car_ids': exact_matches['car_id'].unique().tolist(),
                    'match_type': 'exact'
                }
                exact_match_patterns.append(pattern_key)
            else:
                # Check if year+make exists for potential AI matching
                year_make_matches = self.golden_df[
                    (self.golden_df['year'] == year) &
                    (self.golden_df['make'] == make)
                ]
                
                if len(year_make_matches) > 0:
                    # Year+make found - AI model matching needed
                    pattern_data['year_make_matches'] = year_make_matches
                    ai_needed_patterns.append(pattern_key)
                else:
                    # Check year-only for make matching
                    year_matches = self.golden_df[
                        (self.golden_df['year'] == year)
                    ]
                    
                    if len(year_matches) > 0:
                        # Year found - AI make matching needed
                        pattern_data['year_matches'] = year_matches
                        ai_needed_patterns.append(pattern_key)
                    else:
                        # No matches at all
                        pattern_data['validation_result'] = {
                            'golden_validated': False,
                            'golden_matches': 0,
                            'car_ids': [],
                            'match_type': 'no_year_match'
                        }
                        exact_match_patterns.append(pattern_key)  # No AI needed
        
        print(f"   ✅ Exact matches: {len(exact_match_patterns)}")
        print(f"   ✅ AI needed: {len(ai_needed_patterns)}")
        
        return exact_match_patterns, ai_needed_patterns
    
    def create_optimized_batch_queue(self, ai_needed_patterns: List[str]) -> List:
        """
        Phase 4: Create optimized batch queue with one task per unique pattern.
        
        Args:
            ai_needed_patterns: List of pattern keys that need AI processing
            
        Returns:
            List of batch tasks
        """
        print("🔄 Phase 4: Creating optimized batch queue...")
        
        if not ai_needed_patterns:
            print("   ℹ️  No AI tasks needed - all patterns have exact matches")
            return []
        
        for pattern_key in ai_needed_patterns:
            pattern_data = self.pattern_mapping[pattern_key]
            sample_product = pattern_data['sample_product']
            
            if 'year_make_matches' in pattern_data:
                # Model matching needed
                self.batch_ai_matcher.add_model_matching_task(
                    task_id=f"model_match_{pattern_key}",
                    year_make_matches=pattern_data['year_make_matches'],
                    input_model=pattern_data['model'],
                    input_submodel=sample_product.get('Submodel'),
                    input_type=sample_product.get('Type'),
                    input_doors=sample_product.get('Doors'),
                    input_body_type=sample_product.get('BodyType')
                )
            elif 'year_matches' in pattern_data:
                # Make matching needed
                self.batch_ai_matcher.add_make_matching_task(
                    task_id=f"make_match_{pattern_key}",
                    year_matches=pattern_data['year_matches'],
                    input_make=pattern_data['make']
                )
        
        batch_size = self.batch_ai_matcher.get_queue_size()
        print(f"   ✅ Created {batch_size} optimized batch tasks")
        print(f"   💰 Reduced from {self.optimization_stats['total_products']:,} to {batch_size:,} AI calls")
        
        return batch_size
    
    def process_optimized_batch(self, wait_for_completion: bool = True, max_wait_time: int = 3600) -> bool:
        """
        Phase 5: Process the optimized batch.
        
        Args:
            wait_for_completion: Whether to wait for batch completion
            max_wait_time: Maximum wait time in seconds
            
        Returns:
            True if batch processing completed successfully
        """
        if self.batch_ai_matcher.get_queue_size() == 0:
            print("ℹ️  No AI tasks to process - all patterns had exact matches")
            return True
        
        print(f"🚀 Phase 5: Processing {self.batch_ai_matcher.get_queue_size()} optimized AI tasks...")
        
        # Submit batch
        batch_id = self.batch_ai_matcher.process_batch("steele_optimized")
        if not batch_id:
            print("❌ Failed to submit batch")
            return False
        
        if not wait_for_completion:
            print(f"📋 Batch submitted: {batch_id}")
            print("⏳ Use retrieve_optimized_results() to get results later")
            return True
        
        # Wait for completion
        print("⏳ Waiting for batch to complete...")
        if not self.batch_ai_matcher.wait_for_completion(batch_id, max_wait_time):
            print("❌ Batch did not complete within time limit")
            return False
        
        # Retrieve results
        if not self.batch_ai_matcher.retrieve_batch_results(batch_id):
            print("❌ Failed to retrieve batch results")
            return False
        
        print("✅ Batch processing completed successfully!")
        return True
    
    def apply_results_to_all_products(self, steele_df: pd.DataFrame, ai_needed_patterns: List[str]) -> pd.DataFrame:
        """
        Phase 6: Apply AI results to all products that share the same pattern.
        Enhanced with better error handling and detailed logging.
        
        Args:
            steele_df: Original Steele data
            ai_needed_patterns: List of patterns that needed AI processing
            
        Returns:
            DataFrame with validation results for all products
        """
        print("🔄 Phase 6: Applying results to all products...")
        
        validation_results = []
        
        # Counters for detailed reporting
        result_counters = {
            'exact_matches': 0,
            'ai_model_matches': 0,
            'ai_make_matches': 0,
            'ai_failures': 0,
            'no_matches': 0,
            'errors': 0
        }
        
        # Process all patterns (exact matches + AI results)
        for pattern_key, pattern_data in self.pattern_mapping.items():
            matching_products = pattern_data['products']
            
            try:
                if 'validation_result' in pattern_data:
                    # Use existing validation result (exact match or no match)
                    validation_result = pattern_data['validation_result']
                    if validation_result.get('golden_validated', False):
                        result_counters['exact_matches'] += len(matching_products)
                    else:
                        result_counters['no_matches'] += len(matching_products)
                        
                elif pattern_key in ai_needed_patterns:
                    # Apply AI result to this pattern
                    validation_result = self._process_ai_pattern_result(pattern_key, pattern_data, result_counters)
                else:
                    # Unprocessed pattern - shouldn't happen but handle gracefully
                    validation_result = {
                        'golden_validated': False,
                        'golden_matches': 0,
                        'car_ids': [],
                        'match_type': 'unprocessed'
                    }
                    result_counters['no_matches'] += len(matching_products)
                
                # Apply this result to ALL products in the pattern
                for product_index in matching_products:
                    product_result = {
                        'steele_row_index': product_index,
                        'stock_code': steele_df.iloc[product_index]['StockCode'],
                        'year': pattern_data['year'],
                        'make': pattern_data['make'],
                        'model': pattern_data['model'],
                        'pattern_key': pattern_key,
                        **validation_result
                    }
                    validation_results.append(product_result)
                    
            except Exception as e:
                print(f"⚠️  Error processing pattern {pattern_key}: {e}")
                
                # Create error result but don't drop the products
                error_result = {
                    'golden_validated': False,
                    'golden_matches': 0,
                    'car_ids': [],
                    'match_type': 'processing_error',
                    'error': str(e)
                }
                
                for product_index in matching_products:
                    product_result = {
                        'steele_row_index': product_index,
                        'stock_code': steele_df.iloc[product_index]['StockCode'],
                        'year': pattern_data['year'],
                        'make': pattern_data['make'],
                        'model': pattern_data['model'],
                        'pattern_key': pattern_key,
                        **error_result
                    }
                    validation_results.append(product_result)
                
                result_counters['errors'] += len(matching_products)
        
        validation_df = pd.DataFrame(validation_results)
        
        # Validate that we have results for all products
        if len(validation_df) != len(steele_df):
            raise ValueError(f"Result count mismatch: {len(validation_df)} results for {len(steele_df)} products")
        
        validated_count = len(validation_df[validation_df['golden_validated'] == True])
        
        # Detailed reporting
        print(f"   ✅ Applied results to {len(validation_df):,} products")
        print(f"   📊 RESULT BREAKDOWN:")
        print(f"      🎯 Exact matches: {result_counters['exact_matches']:,}")
        print(f"      🤖 AI model matches: {result_counters['ai_model_matches']:,}")
        print(f"      🔧 AI make matches: {result_counters['ai_make_matches']:,}")
        print(f"      ❌ AI failures: {result_counters['ai_failures']:,}")
        print(f"      ⚪ No matches: {result_counters['no_matches']:,}")
        print(f"      💥 Errors: {result_counters['errors']:,}")
        print(f"   ✅ Total validated: {validated_count:,}")
        
        return validation_df
    
    def _process_ai_pattern_result(self, pattern_key: str, pattern_data: Dict, result_counters: Dict) -> Dict:
        """
        Process AI result for a specific pattern with enhanced error handling.
        
        Args:
            pattern_key: The pattern key
            pattern_data: Pattern data including AI matches
            result_counters: Counters for tracking results
            
        Returns:
            Validation result dictionary
        """
        matching_products_count = len(pattern_data['products'])
        
        if 'model_match_' in pattern_key:
            task_id = f"model_match_{pattern_key}"
            year_make_matches = pattern_data.get('year_make_matches', pd.DataFrame())
            ai_matches = self.batch_ai_matcher.get_model_match_result(task_id, year_make_matches)
            
            if len(ai_matches) > 0:
                result_counters['ai_model_matches'] += matching_products_count
                return {
                    'golden_validated': True,
                    'golden_matches': len(ai_matches),
                    'car_ids': ai_matches['car_id'].unique().tolist(),
                    'match_type': 'ai_model_match'
                }
            else:
                result_counters['ai_failures'] += matching_products_count
                return {
                    'golden_validated': False,
                    'golden_matches': 0,
                    'car_ids': [],
                    'match_type': 'ai_model_match_failed'
                }
                
        elif 'make_match_' in pattern_key:
            task_id = f"make_match_{pattern_key}"
            corrected_make = self.batch_ai_matcher.get_make_match_result(task_id)
            
            if corrected_make and corrected_make != "NO_MATCH":
                # Re-validate with corrected make
                validation_result = self._validate_with_corrected_make(pattern_data, corrected_make)
                validation_result['corrected_make'] = corrected_make
                
                if validation_result.get('golden_validated', False):
                    result_counters['ai_make_matches'] += matching_products_count
                else:
                    result_counters['ai_failures'] += matching_products_count
                    
                return validation_result
            else:
                result_counters['ai_failures'] += matching_products_count
                return {
                    'golden_validated': False,
                    'golden_matches': 0,
                    'car_ids': [],
                    'match_type': 'ai_make_match_failed'
                }
        else:
            result_counters['ai_failures'] += matching_products_count
            return {
                'golden_validated': False,
                'golden_matches': 0,
                'car_ids': [],
                'match_type': 'unknown_pattern'
            }
    
    def _validate_with_corrected_make(self, pattern_data: Dict, corrected_make: str) -> Dict:
        """Validate with AI-corrected make."""
        try:
            year = pattern_data['year']
            original_model = pattern_data['model']
            
            # Step 1: Try exact match with corrected make
            exact_matches = self.golden_df[
                (self.golden_df['year'] == year) &
                (self.golden_df['make'] == corrected_make) &
                (self.golden_df['model'] == original_model)
            ]
            
            if len(exact_matches) > 0:
                return {
                    'golden_validated': True,
                    'golden_matches': len(exact_matches),
                    'car_ids': exact_matches['car_id'].unique().tolist(),
                    'match_type': 'exact_with_corrected_make'
                }
            
            # Step 2: Try year + corrected_make match
            year_make_matches = self.golden_df[
                (self.golden_df['year'] == year) &
                (self.golden_df['make'] == corrected_make)
            ]
            
            if len(year_make_matches) > 0:
                return {
                    'golden_validated': True,
                    'golden_matches': len(year_make_matches),
                    'car_ids': year_make_matches['car_id'].unique().tolist(),
                    'match_type': 'year_make_with_corrected_make'
                }
            
            # No matches found even with corrected make
            return {
                'golden_validated': False,
                'golden_matches': 0,
                'car_ids': [],
                'match_type': 'no_match_with_corrected_make'
            }
            
        except Exception as e:
            return {
                'golden_validated': False,
                'golden_matches': 0,
                'car_ids': [],
                'match_type': 'error_with_corrected_make',
                'error': str(e)
            }
    
    def transform_to_standard_format(self, steele_df: pd.DataFrame, validation_df: pd.DataFrame) -> List[ProductData]:
        """Transform to standard format preserving existing fitment data."""
        standard_products = []
        
        for idx, steele_row in steele_df.iterrows():
            # Get validation result for this row
            validation_row = validation_df[validation_df['steele_row_index'] == idx]
            is_validated = len(validation_row) > 0 and validation_row.iloc[0]['golden_validated']
            
            # Extract vehicle data
            year = str(steele_row['Year']) if pd.notna(steele_row['Year']) else "Unknown"
            make = str(steele_row['Make']) if pd.notna(steele_row['Make']) else "NONE"
            model = str(steele_row['Model']) if pd.notna(steele_row['Model']) else "NONE"
            
            # Get car_ids from validation result
            car_ids = []
            if len(validation_row) > 0 and 'car_ids' in validation_row.columns:
                car_ids_value = validation_row.iloc[0]['car_ids']
                if isinstance(car_ids_value, list):
                    car_ids = car_ids_value
                elif pd.notna(car_ids_value):
                    car_ids = [str(car_ids_value)]
            
            product_data = ProductData(
                title=str(steele_row['Product Name']),
                car_ids=car_ids,
                year_min=year,
                year_max=year,
                make=make,
                model=model,
                mpn=str(steele_row.get('PartNumber', steele_row['StockCode'])),
                cost=float(steele_row['Dealer Price']) if pd.notna(steele_row['Dealer Price']) else 0.0,
                price=float(steele_row['MAP']) if pd.notna(steele_row['MAP']) else 0.0,
                body_html=str(steele_row['Description']),
                golden_validated=is_validated,
                fitment_source="vendor_provided",
                processing_method="template_based"
            )
            
            standard_products.append(product_data)
        
        return standard_products
    
    def enhance_with_templates(self, product_data_list: List[ProductData]) -> List[ProductData]:
        """Enhance data using templates only."""
        enhanced_products = []
        
        for product_data in product_data_list:
            try:
                # Only enhance golden-validated products
                if product_data.golden_validated:
                    product_data.meta_title = self.template_generator.generate_meta_title(
                        product_data.title, product_data.year_min, product_data.make, product_data.model
                    )
                    product_data.meta_description = self.template_generator.generate_meta_description(
                        product_data.title, product_data.year_min, product_data.make, product_data.model
                    )
                    product_data.collection = self.template_generator.categorize_product(product_data.title)
                else:
                    # Basic defaults for non-validated products
                    product_data.meta_title = product_data.title[:60]
                    product_data.meta_description = f"Quality automotive {product_data.title}."[:160]
                    product_data.collection = "Accessories"
                
                enhanced_products.append(product_data)
                
            except Exception as e:
                print(f"Template enhancement failed for {product_data.title}: {e}")
                # Use fallback generation
                product_data.meta_title = product_data.title[:60]
                product_data.meta_description = f"Quality automotive {product_data.title}."[:160]
                enhanced_products.append(product_data)
        
        return enhanced_products
    
    def _consolidate_products_by_sku(self, products: List[ProductData]) -> List[ProductData]:
        """Consolidate products by SKU, combining car_ids and creating year ranges."""
        sku_groups = {}
        
        # Group products by SKU
        for product in products:
            sku = product.mpn
            if sku not in sku_groups:
                sku_groups[sku] = []
            sku_groups[sku].append(product)
        
        consolidated = []
        
        for sku, group_products in sku_groups.items():
            if len(group_products) == 1:
                consolidated.append(group_products[0])
            else:
                # Multiple products with same SKU - consolidate
                base_product = group_products[0]
                
                # Combine all car_ids and deduplicate
                all_car_ids = []
                for product in group_products:
                    all_car_ids.extend(product.car_ids)
                unique_car_ids = list(dict.fromkeys(all_car_ids))
                
                # Extract years, makes, and models from car_ids for comprehensive ranges
                years = set()
                makes = set()
                models = set()
                for car_id in unique_car_ids:
                    if car_id and '_' in car_id:
                        parts = car_id.split('_')
                        if len(parts) >= 3:
                            year_part = parts[0]
                            make_part = parts[1]
                            model_part = parts[2]
                            
                            if year_part.isdigit():
                                years.add(int(year_part))
                            makes.add(make_part)
                            models.add(model_part)
                
                # Determine year range
                if years:
                    min_year = str(min(years))
                    max_year = str(max(years))
                else:
                    min_year = base_product.year_min
                    max_year = base_product.year_max
                
                # Determine consolidated make and model
                if len(makes) == 1:
                    consolidated_make = list(makes)[0]
                else:
                    consolidated_make = "Multiple Makes"
                
                if len(models) == 1:
                    consolidated_model = list(models)[0]
                else:
                    consolidated_model = "Multiple Models"
                
                # Create consolidated product
                consolidated_product = ProductData(
                    title=base_product.title,
                    year_min=min_year,
                    year_max=max_year,
                    make=consolidated_make,
                    model=consolidated_model,
                    mpn=base_product.mpn,
                    cost=base_product.cost,
                    price=base_product.price,
                    body_html=base_product.body_html,
                    collection=base_product.collection,
                    product_type=base_product.product_type,
                    meta_title=self._generate_consolidated_meta_title(base_product.title, min_year, max_year, unique_car_ids),
                    meta_description=self._generate_consolidated_meta_description(base_product.title, min_year, max_year, unique_car_ids),
                    car_ids=unique_car_ids,
                    golden_validated=any(p.golden_validated for p in group_products),
                    fitment_source=base_product.fitment_source,
                    processing_method=base_product.processing_method
                )
                
                consolidated.append(consolidated_product)
        
        return consolidated
    
    def _generate_consolidated_meta_title(self, title: str, min_year: str, max_year: str, car_ids: List[str]) -> str:
        """Generate meta title for consolidated product"""
        if min_year == max_year:
            year_range = min_year
        else:
            year_range = f"{min_year}-{max_year}"
        
        # Count unique makes from car_ids
        makes = set()
        for car_id in car_ids:
            if car_id and '_' in car_id:
                parts = car_id.split('_')
                if len(parts) >= 2:
                    makes.add(parts[1])
        
        if len(makes) == 1:
            make_text = list(makes)[0]
        else:
            make_text = "Multiple Makes"
        
        meta_title = f"{title} - {year_range} {make_text}"
        return meta_title[:60] if len(meta_title) > 60 else meta_title
    
    def _generate_consolidated_meta_description(self, title: str, min_year: str, max_year: str, car_ids: List[str]) -> str:
        """Generate meta description for consolidated product"""
        if min_year == max_year:
            year_range = min_year
        else:
            year_range = f"{min_year}-{max_year}"
        
        vehicle_count = len(car_ids)
        meta_desc = f"Quality {title} compatible with {vehicle_count} vehicle models from {year_range}. OEM replacement part."
        return meta_desc[:160] if len(meta_desc) > 160 else meta_desc
    
    def transform_to_formatted_shopify_import(self, enhanced_products: List[ProductData]) -> pd.DataFrame:
        """Transform to complete Shopify import format."""
        # Step 1: Consolidate products by SKU
        consolidated_products = self._consolidate_products_by_sku(enhanced_products)
        
        final_records = []
        
        for product_data in consolidated_products:
            # Generate vehicle tags from car_ids
            vehicle_tags = ", ".join(product_data.car_ids) if product_data.car_ids else ""
            
            # Create complete record with ALL columns
            final_record = {}
            
            # Populate each column in the exact order from cols_list
            for col in cols_list:
                if col == "ID":
                    final_record[col] = ""
                elif col == "Command":
                    final_record[col] = "MERGE"
                elif col == "Title":
                    final_record[col] = product_data.title
                elif col == "Body HTML":
                    final_record[col] = product_data.body_html
                elif col == "Vendor":
                    final_record[col] = self.vendor_name
                elif col == "Tags":
                    final_record[col] = vehicle_tags
                elif col == "Tags Command":
                    final_record[col] = "MERGE"
                elif col == "Custom Collections":
                    final_record[col] = product_data.collection
                elif col == "Variant Position":
                    final_record[col] = 1
                elif col == "Variant SKU":
                    final_record[col] = product_data.mpn
                elif col == "Variant Price":
                    final_record[col] = product_data.price
                elif col == "Variant Cost":
                    final_record[col] = product_data.cost
                elif col == "Variant Taxable":
                    final_record[col] = "TRUE"
                elif col == "Variant Inventory Tracker":
                    final_record[col] = "shopify"
                elif col == "Variant Inventory Policy":
                    final_record[col] = "deny"
                elif col == "Variant Fulfillment Service":
                    final_record[col] = "manual"
                elif col == "Variant Requires Shipping":
                    final_record[col] = "TRUE"
                elif col == "Variant Inventory Qty":
                    final_record[col] = 0
                elif col == "Image Position":
                    final_record[col] = 1
                elif col == "Image Command":
                    final_record[col] = "MERGE"
                elif col == "Image Alt Text":
                    final_record[col] = product_data.title
                elif col == "Variant Command":
                    final_record[col] = "MERGE"
                elif col == "Metafield: title_tag [string]":
                    final_record[col] = product_data.meta_title
                elif col == "Metafield: description_tag [string]":
                    final_record[col] = product_data.meta_description
                elif col == "Variant Metafield: mm-google-shopping.mpn [single_line_text_field]":
                    final_record[col] = product_data.mpn
                elif col == "Variant Metafield: mm-google-shopping.condition [single_line_text_field]":
                    final_record[col] = "new"
                elif col == "Metafield: mm-google-shopping.mpn [single_line_text_field]":
                    final_record[col] = product_data.mpn
                else:
                    # Default empty value for any unmapped columns
                    final_record[col] = ""
            
            final_records.append(final_record)
        
        # Create DataFrame with columns in exact order
        return pd.DataFrame(final_records, columns=cols_list)
    
    def process_ultra_optimized_pipeline(self, sample_file_path: str = "data/samples/steele_sample.csv") -> pd.DataFrame:
        """
        Execute the ultra-optimized transformation pipeline with pattern deduplication.
        
        Args:
            sample_file_path: Path to Steele dataset
            
        Returns:
            Final consolidated Shopify-ready DataFrame
        """
        print("🚀 ULTRA-OPTIMIZED STEELE PIPELINE")
        print("   Pattern deduplication • 50% batch API savings • Ultra-fast")
        print("=" * 80)
        
        # Phase 1: Load and analyze data
        print("\n📊 PHASE 1: DATA ANALYSIS")
        steele_df = self.load_sample_data(sample_file_path)
        print(f"✅ Loaded {len(steele_df):,} Steele products")
        
        pattern_groups, unique_patterns = self.analyze_steele_patterns(steele_df)
        
        # Phase 2: Create pattern mapping
        print("\n🔄 PHASE 2: PATTERN MAPPING")
        pattern_mapping = self.create_pattern_mapping(steele_df, pattern_groups)
        
        # Save pattern mapping to file for analysis
        self.save_pattern_mapping()
        
        

        # Phase 3: Validate patterns
        print("\n🔍 PHASE 3: PATTERN VALIDATION")
        exact_match_patterns, ai_needed_patterns = self.validate_patterns_against_golden(pattern_mapping)
        
        # Phase 4: Create optimized batch
        print("\n🚀 PHASE 4: OPTIMIZED BATCH CREATION")
        batch_size = self.create_optimized_batch_queue(ai_needed_patterns)
        
        # Phase 5: Process batch (if needed)
        if batch_size > 0:
            print("\n⚡ PHASE 5: BATCH PROCESSING")
            if not self.process_optimized_batch(wait_for_completion=True):
                print("⚠️  Batch processing failed, continuing with available results")
        
        # Phase 6: Apply results to all products
        print("\n🔄 PHASE 6: RESULT APPLICATION")
        validation_df = self.apply_results_to_all_products(steele_df, ai_needed_patterns)
        
        # Phase 7: Complete transformation pipeline
        print("\n🔄 PHASE 7: TRANSFORMATION PIPELINE")
        standard_products = self.transform_to_standard_format(steele_df, validation_df)
        enhanced_products = self.enhance_with_templates(standard_products)
        final_df = self.transform_to_formatted_shopify_import(enhanced_products)
        
        # Show final statistics
        print("\n" + "=" * 80)
        print("✅ ULTRA-OPTIMIZED PIPELINE COMPLETE")
        print(f"   📊 Total products: {len(steele_df):,}")
        print(f"   📊 Unique patterns: {unique_patterns:,}")
        print(f"   📊 Deduplication: {self.optimization_stats['deduplication_factor']:.1f}x")
        print(f"   📊 AI calls saved: {self.optimization_stats['ai_calls_saved']:,}")
        print(f"   📊 Final products: {len(final_df):,}")
        print(f"   💰 Cost savings: 50% batch API + deduplication")
        print("=" * 80)
        
        # Show AI cost summary if any calls were made
        if self.batch_ai_matcher.api_calls_made > 0:
            print("\n")
            self.batch_ai_matcher.print_cost_report()
        
        return final_df
    
    def _validate_input_data(self, df: pd.DataFrame) -> None:
        """Validate input data structure and quality."""
        if len(df) == 0:
            raise ValueError("Input data is empty")
        
        required_columns = [
            'StockCode', 'Product Name', 'Description', 'MAP', 'Dealer Price',
            'Year', 'Make', 'Model', 'PartNumber'
        ]
        
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Validate critical fields are not all null
        critical_fields = ['StockCode', 'Product Name', 'Year', 'Make', 'Model']
        for field in critical_fields:
            if df[field].isnull().all():
                raise ValueError(f"Critical field '{field}' is completely empty")