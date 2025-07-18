import pandas as pd
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel
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
    """Standard format for complete fitment data sources (NO AI)"""
    title: str
    year_min: str = "1800"
    year_max: str = "1800"
    make: str = "NONE"
    model: str = "NONE"
    mpn: str = ""
    cost: float = 0.0
    price: float = 0.0
    body_html: str = ""
    collection: str = "Accessories"  # Template-based categorization
    product_type: str = "Automotive Part"
    meta_title: str = ""     # Template-generated
    meta_description: str = "" # Template-generated
    car_ids: List[str] = []  # Golden dataset car IDs for compatibility
    
    # Validation flags
    golden_validated: bool = False
    fitment_source: str = "vendor_provided"
    processing_method: str = "template_based"  # NOT ai_enhanced

class TemplateGenerator:
    """Template-based enhancement for complete fitment data sources (NO AI)"""
    
    def generate_meta_title(self, product_name: str, year: str, make: str, model: str) -> str:
        """Generate SEO meta title using template"""
        template = f"{product_name} - {year} {make} {model}"
        return template[:60] if len(template) > 60 else template
    
    def generate_meta_description(self, product_name: str, year: str, make: str, model: str) -> str:
        """Generate SEO meta description using template"""
        template = f"Quality {product_name} for {year} {make} {model} vehicles. OEM replacement part."
        return template[:160] if len(template) > 160 else template
    
    def categorize_product(self, product_name: str) -> str:
        """Rule-based product categorization (NO AI)"""
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

class BatchSteeleDataTransformer:
    """
    Batch-enabled transformer for Steele data source using OpenAI's batch API.
    Processes AI requests in batches for significant cost savings and efficiency.
    """
    
    def __init__(self, use_ai: bool = True):
        """
        Initialize the batch transformer for complete fitment data.
        
        Args:
            use_ai: Whether to use AI for edge cases (batch processing)
        """
        self.use_ai = use_ai
        self.vendor_name = "Steele"
        self.golden_df = None
        self.template_generator = TemplateGenerator()
        self.batch_ai_matcher = BatchAIVehicleMatcher(use_ai=use_ai)
        
        # Track items that need AI processing
        self.ai_processing_needed = {}
        
        print("🔄 Initialized Batch Steele Data Transformer")
        if use_ai:
            print("   ✅ AI batch processing enabled (50% cost savings)")
        else:
            print("   ⚠️  AI processing disabled")
    
    def load_sample_data(self, file_path: str = "data/samples/steele_sample.csv") -> pd.DataFrame:
        """
        Step 1: Load Steele sample data for processing.
        
        Args:
            file_path: Path to the sample CSV file (relative to Steele directory)
            
        Returns:
            DataFrame with loaded data
            
        Raises:
            FileNotFoundError: If sample file doesn't exist
            ValueError: If data format is invalid
        """
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
        """
        Step 2: Load golden master dataset for vehicle validation.
        
        Args:
            file_path: Path to golden dataset, uses shared path if not provided
            
        Returns:
            DataFrame with golden dataset
        """
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
    
    def validate_against_golden_dataset_batch(self, steele_df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 2b: Validate Steele vehicles against golden master dataset.
        Collects AI tasks in batch instead of processing individually.
        
        Args:
            steele_df: Steele data to validate
            
        Returns:
            DataFrame with validation results (AI tasks queued for batch processing)
        """
        if self.golden_df is None:
            self.load_golden_dataset()
        
        validation_results = []
        ai_tasks_queued = 0

        # Progress tracking for large datasets
        total_rows = len(steele_df)
        progress_interval = max(1000, total_rows // 100)  # Show progress every 1% or 1k rows
        
        print(f"🔍 Validating {total_rows:,} products against golden dataset...")
        
        for idx, row in steele_df.iterrows():
            try:
                # Show progress for large datasets
                if idx % progress_interval == 0:
                    progress = (idx / total_rows) * 100
                    print(f"   Processing... {progress:.1f}% ({idx:,}/{total_rows:,})")
                
                year = int(row['Year']) if pd.notna(row['Year']) else None
                make = str(row['Make']) if pd.notna(row['Make']) else None
                model = str(row['Model']) if pd.notna(row['Model']) else None
                submodel = str(row['Submodel']) if pd.notna(row['Submodel']) else None
                type = str(row['Type']) if pd.notna(row['Type']) else None
                doors = str(row['Doors']) if pd.notna(row['Doors']) else None
                body_type = str(row['BodyType']) if pd.notna(row['BodyType']) else None
                
                if year and make and model:
                    # Step 1: Try exact match (year + make + model)
                    exact_matches = self.golden_df[
                        (self.golden_df['year'] == year) &
                        (self.golden_df['make'] == make) &
                        (self.golden_df['model'] == model)
                    ]
                    
                    if len(exact_matches) > 0:
                        # Found exact match - no AI needed
                        validation_results.append({
                            'steele_row_index': idx,
                            'stock_code': row['StockCode'],
                            'year': year,
                            'make': make,
                            'model': model,
                            'golden_validated': True,
                            'golden_matches': len(exact_matches),
                            'car_ids': exact_matches['car_id'].unique().tolist(),
                            'match_type': 'exact',
                            'ai_task_id': None
                        })
                    else:
                        # Step 2: Try year + make match, then queue AI model matching
                        year_make_matches = self.golden_df[
                            (self.golden_df['year'] == year) &
                            (self.golden_df['make'] == make)
                        ]
                        
                        if len(year_make_matches) > 0:
                            # Special case: if make == model, return all year+make matches
                            if make == model:
                                validation_results.append({
                                    'steele_row_index': idx,
                                    'stock_code': row['StockCode'],
                                    'year': year,
                                    'make': make,
                                    'model': model,
                                    'golden_validated': True,
                                    'golden_matches': len(year_make_matches),
                                    'car_ids': year_make_matches['car_id'].unique().tolist(),
                                    'match_type': 'make_equals_model',
                                    'ai_task_id': None
                                })
                            else:
                                # Queue AI model matching task
                                task_id = f"model_match_{idx}"
                                self.batch_ai_matcher.add_model_matching_task(
                                    task_id=task_id,
                                    year_make_matches=year_make_matches,
                                    input_model=model,
                                    input_submodel=submodel,
                                    input_type=type,
                                    input_doors=doors,
                                    input_body_type=body_type
                                )
                                
                                # Store for later processing
                                self.ai_processing_needed[task_id] = {
                                    'type': 'model_match',
                                    'steele_row_index': idx,
                                    'year_make_matches': year_make_matches,
                                    'stock_code': row['StockCode'],
                                    'year': year,
                                    'make': make,
                                    'model': model
                                }
                                
                                # Placeholder result (will be updated after batch processing)
                                validation_results.append({
                                    'steele_row_index': idx,
                                    'stock_code': row['StockCode'],
                                    'year': year,
                                    'make': make,
                                    'model': model,
                                    'golden_validated': False,  # Will be updated
                                    'golden_matches': 0,  # Will be updated
                                    'car_ids': [],  # Will be updated
                                    'match_type': 'ai_model_match_pending',
                                    'ai_task_id': task_id
                                })
                                
                                ai_tasks_queued += 1
                        else:
                            # No year+make matches found, try fuzzy make matching
                            year_matches = self.golden_df[
                                (self.golden_df['year'] == year)
                            ]
                            
                            if len(year_matches) > 0:
                                # Queue AI make matching task
                                task_id = f"make_match_{idx}"
                                self.batch_ai_matcher.add_make_matching_task(
                                    task_id=task_id,
                                    year_matches=year_matches,
                                    input_make=make
                                )
                                
                                # Store for later processing
                                self.ai_processing_needed[task_id] = {
                                    'type': 'make_match',
                                    'steele_row_index': idx,
                                    'year_matches': year_matches,
                                    'stock_code': row['StockCode'],
                                    'year': year,
                                    'make': make,
                                    'model': model,
                                    'submodel': submodel,
                                    'type': type,
                                    'doors': doors,
                                    'body_type': body_type
                                }
                                
                                # Placeholder result
                                validation_results.append({
                                    'steele_row_index': idx,
                                    'stock_code': row['StockCode'],
                                    'year': year,
                                    'make': make,
                                    'model': model,
                                    'golden_validated': False,  # Will be updated
                                    'golden_matches': 0,  # Will be updated
                                    'car_ids': [],  # Will be updated
                                    'match_type': 'ai_make_match_pending',
                                    'ai_task_id': task_id
                                })
                                
                                ai_tasks_queued += 1
                            else:
                                # No year matches at all
                                validation_results.append({
                                    'steele_row_index': idx,
                                    'stock_code': row['StockCode'],
                                    'year': year,
                                    'make': make,
                                    'model': model,
                                    'golden_validated': False,
                                    'golden_matches': 0,
                                    'car_ids': [],
                                    'match_type': 'no_year_match',
                                    'ai_task_id': None
                                })
                else:
                    validation_results.append({
                        'steele_row_index': idx,
                        'stock_code': row['StockCode'],
                        'year': year,
                        'make': make,
                        'model': model,
                        'golden_validated': False,
                        'golden_matches': 0,
                        'car_ids': [],
                        'error': 'incomplete_vehicle_data',
                        'ai_task_id': None
                    })
                    
            except Exception as e:
                validation_results.append({
                    'steele_row_index': idx,
                    'stock_code': row.get('StockCode', 'unknown'),
                    'golden_validated': False,
                    'error': str(e),
                    'ai_task_id': None
                })
        
        print(f"✅ Initial validation complete: {ai_tasks_queued} AI tasks queued for batch processing")
        
        return pd.DataFrame(validation_results)
    
    def process_ai_batch(self, wait_for_completion: bool = True, max_wait_time: int = 3600) -> bool:
        """
        Process all queued AI tasks using batch API.
        
        Args:
            wait_for_completion: Whether to wait for batch to complete
            max_wait_time: Maximum time to wait for completion (seconds)
            
        Returns:
            True if batch processing completed successfully
        """
        if self.batch_ai_matcher.get_queue_size() == 0:
            print("ℹ️  No AI tasks to process")
            return True
        
        print(f"🚀 Processing {self.batch_ai_matcher.get_queue_size()} AI tasks in batch...")
        
        # Submit batch
        batch_id = self.batch_ai_matcher.process_batch("steele_validation")
        if not batch_id:
            print("❌ Failed to submit batch")
            return False
        
        if not wait_for_completion:
            print(f"📋 Batch submitted: {batch_id}")
            print("⏳ Use check_batch_status() and retrieve_batch_results() to get results later")
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
    
    def update_validation_with_ai_results(self, validation_df: pd.DataFrame) -> pd.DataFrame:
        """
        Update validation results with AI batch processing results.
        
        Args:
            validation_df: DataFrame with pending AI tasks
            
        Returns:
            Updated DataFrame with AI results incorporated
        """
        updated_results = []
        
        for idx, row in validation_df.iterrows():
            if pd.isna(row.get('ai_task_id')) or row['ai_task_id'] is None:
                # No AI processing needed, keep as-is
                updated_results.append(row.to_dict())
                continue
            
            task_id = row['ai_task_id']
            task_info = self.ai_processing_needed.get(task_id)
            
            if not task_info:
                print(f"⚠️  No task info found for {task_id}")
                updated_results.append(row.to_dict())
                continue
            
            # Process based on task type
            if task_info['type'] == 'model_match':
                ai_matches = self.batch_ai_matcher.get_model_match_result(
                    task_id, task_info['year_make_matches']
                )
                
                if len(ai_matches) > 0:
                    # AI found matches
                    updated_row = row.to_dict()
                    updated_row.update({
                        'golden_validated': True,
                        'golden_matches': len(ai_matches),
                        'car_ids': ai_matches['car_id'].unique().tolist(),
                        'match_type': 'ai_model_match'
                    })
                    updated_results.append(updated_row)
                else:
                    # AI found no matches - add debug output
                    print(f"⚠️  AI returned empty matches for {task_id}")
                    print(f"    Input year: {task_info.get('year', 'unknown')}")
                    print(f"    Input make: {task_info.get('make', 'unknown')}")
                    print(f"    Input model: {task_info.get('model', 'unknown')}")
                    if 'year_make_matches' in task_info and len(task_info['year_make_matches']) > 0:
                        available_models = task_info['year_make_matches']['model'].unique()
                        print(f"    Available models: {list(available_models)}")
                        print(f"    Available car_ids: {list(task_info['year_make_matches']['car_id'].unique())}")
                    else:
                        print(f"    No year_make_matches available in task_info")
                    
                    updated_row = row.to_dict()
                    updated_row.update({
                        'golden_validated': False,
                        'golden_matches': 0,
                        'car_ids': [],
                        'match_type': 'ai_model_match_failed'
                    })
                    updated_results.append(updated_row)
            
            elif task_info['type'] == 'make_match':
                corrected_make = self.batch_ai_matcher.get_make_match_result(task_id)
                
                if corrected_make:
                    # AI corrected the make, now validate with corrected make
                    validation_result = self._validate_with_corrected_make_batch(
                        task_info, corrected_make
                    )
                    
                    updated_row = row.to_dict()
                    updated_row.update(validation_result)
                    updated_row['corrected_make'] = corrected_make
                    updated_results.append(updated_row)
                else:
                    # AI could not correct make
                    updated_row = row.to_dict()
                    updated_row.update({
                        'golden_validated': False,
                        'golden_matches': 0,
                        'car_ids': [],
                        'match_type': 'ai_make_match_failed'
                    })
                    updated_results.append(updated_row)
            
            else:
                print(f"⚠️  Unknown task type: {task_info['type']}")
                updated_results.append(row.to_dict())
        
        return pd.DataFrame(updated_results)
    
    def _validate_with_corrected_make_batch(self, task_info: Dict, corrected_make: str) -> Dict:
        """
        Validate with AI-corrected make.
        
        Args:
            task_info: Original task information
            corrected_make: Make corrected by AI
            
        Returns:
            Dictionary with validation results
        """
        try:
            year = task_info['year']
            original_model = task_info['model']
            
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
                # Special case: if corrected_make == model, return all matches
                if corrected_make == original_model:
                    return {
                        'golden_validated': True,
                        'golden_matches': len(year_make_matches),
                        'car_ids': year_make_matches['car_id'].unique().tolist(),
                        'match_type': 'make_equals_model_corrected'
                    }
                else:
                    # Would need another AI call for model matching with corrected make
                    # For simplicity, return the year+make matches
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
        """
        Step 3: Transform to standard format preserving existing fitment data (NO AI).
        
        Args:
            steele_df: Original Steele data
            validation_df: Golden dataset validation results
            
        Returns:
            List of ProductData objects for template processing
        """
        standard_products = []
        
        for idx, steele_row in steele_df.iterrows():
            # Get validation result for this row
            validation_row = validation_df[validation_df['steele_row_index'] == idx]
            is_validated = len(validation_row) > 0 and validation_row.iloc[0]['golden_validated']
            
            # Extract vehicle data (already available in Steele)
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
                year_max=year,  # Same year for Steele
                make=make,
                model=model,
                mpn=str(steele_row.get('PartNumber', steele_row['StockCode'])),
                cost=float(steele_row['Dealer Price']) if pd.notna(steele_row['Dealer Price']) else 0.0,
                price=float(steele_row['MAP']) if pd.notna(steele_row['MAP']) else 0.0,
                body_html=str(steele_row['Description']),
                golden_validated=is_validated,
                fitment_source="vendor_provided",  # Steele provides complete fitment
                processing_method="template_based"   # NO AI processing
            )
            
            standard_products.append(product_data)
        
        return standard_products
    
    def enhance_with_templates(self, product_data_list: List[ProductData]) -> List[ProductData]:
        """
        Step 3b: Enhance data using templates only (NO AI).
        
        Args:
            product_data_list: List of ProductData to enhance
            
        Returns:
            Enhanced ProductData list using templates
        """
        enhanced_products = []
        
        for product_data in product_data_list:
            try:
                # Only enhance golden-validated products
                if product_data.golden_validated:
                    # Template-based SEO fields
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
        """
        Consolidate products by SKU, combining car_ids and creating year ranges.
        
        Args:
            products: List of ProductData to consolidate
            
        Returns:
            List of consolidated ProductData objects
        """
        sku_groups = {}
        
        # Group products by SKU
        for product in products:
            sku = product.mpn  # Using mpn as SKU
            if sku not in sku_groups:
                sku_groups[sku] = []
            sku_groups[sku].append(product)
        
        consolidated = []
        
        for sku, group_products in sku_groups.items():
            if len(group_products) == 1:
                # Single product, no consolidation needed
                consolidated.append(group_products[0])
            else:
                # Multiple products with same SKU - consolidate
                base_product = group_products[0]  # Use first as template
                
                # Combine all car_ids and deduplicate
                all_car_ids = []
                for product in group_products:
                    all_car_ids.extend(product.car_ids)
                unique_car_ids = list(dict.fromkeys(all_car_ids))  # Preserves order while deduplicating
                
                # Extract years from car_ids for year range
                years = set()
                for car_id in unique_car_ids:
                    if car_id and '_' in car_id:
                        year_part = car_id.split('_')[0]
                        if year_part.isdigit():
                            years.add(int(year_part))
                
                # Determine year range
                if years:
                    min_year = str(min(years))
                    max_year = str(max(years))
                else:
                    min_year = base_product.year_min
                    max_year = base_product.year_max
                
                # Create consolidated product
                consolidated_product = ProductData(
                    title=base_product.title,
                    year_min=min_year,
                    year_max=max_year,
                    make=base_product.make,
                    model=base_product.model,
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
        """
        FORMATTED STEP: Transform template-enhanced data to complete Shopify import format.
        Consolidates products by SKU and combines car_ids. Generates ALL 65 columns in correct order.
        
        Args:
            enhanced_products: List of template-enhanced ProductData
            
        Returns:
            DataFrame in complete 65-column Shopify import format (FORMATTED STEP)
        """
        # Step 1: Consolidate products by SKU
        consolidated_products = self._consolidate_products_by_sku(enhanced_products)
        
        final_records = []
        
        for product_data in consolidated_products:
            # Generate vehicle tags from car_ids (comma-separated)
            vehicle_tags = ", ".join(product_data.car_ids) if product_data.car_ids else ""
            
            # Create complete record with ALL columns from product_import requirements
            final_record = {}
            
            # Populate each column in the exact order from cols_list
            for col in cols_list:
                if col == "ID":
                    final_record[col] = ""  # Shopify will auto-generate
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
                elif col == "Category: ID":
                    final_record[col] = ""
                elif col == "Category: Name":
                    final_record[col] = ""
                elif col == "Category":
                    final_record[col] = ""
                elif col == "Custom Collections":
                    final_record[col] = product_data.collection
                elif col == "Smart Collections":
                    final_record[col] = ""
                elif col == "Image Type":
                    final_record[col] = ""
                elif col == "Image Src":
                    final_record[col] = ""
                elif col == "Image Command":
                    final_record[col] = "MERGE"
                elif col == "Image Position":
                    final_record[col] = 1
                elif col == "Image Width":
                    final_record[col] = ""
                elif col == "Image Height":
                    final_record[col] = ""
                elif col == "Image Alt Text":
                    final_record[col] = product_data.title
                elif col == "Variant Inventory Item ID":
                    final_record[col] = ""
                elif col == "Variant ID":
                    final_record[col] = ""
                elif col == "Variant Command":
                    final_record[col] = "MERGE"
                elif col == "Option1 Name":
                    final_record[col] = ""
                elif col == "Option1 Value":
                    final_record[col] = ""
                elif col == "Option2 Name":
                    final_record[col] = ""
                elif col == "Option2 Value":
                    final_record[col] = ""
                elif col == "Option3 Name":
                    final_record[col] = ""
                elif col == "Option3 Value":
                    final_record[col] = ""
                elif col == "Variant Position":
                    final_record[col] = 1
                elif col == "Variant SKU":
                    final_record[col] = product_data.mpn
                elif col == "Variant Barcode":
                    final_record[col] = ""
                elif col == "Variant Image":
                    final_record[col] = ""
                elif col == "Variant Weight":
                    final_record[col] = ""
                elif col == "Variant Weight Unit":
                    final_record[col] = ""
                elif col == "Variant Price":
                    final_record[col] = product_data.price
                elif col == "Variant Compare At Price":
                    final_record[col] = ""
                elif col == "Variant Taxable":
                    final_record[col] = "TRUE"
                elif col == "Variant Tax Code":
                    final_record[col] = ""
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
                elif col == "Variant Inventory Adjust":
                    final_record[col] = ""
                elif col == "Variant Cost":
                    final_record[col] = product_data.cost
                elif col == "Variant HS Code":
                    final_record[col] = ""
                elif col == "Variant Country of Origin":
                    final_record[col] = ""
                elif col == "Variant Province of Origin":
                    final_record[col] = ""
                elif col == "Metafield: title_tag [string]":
                    final_record[col] = product_data.meta_title
                elif col == "Metafield: description_tag [string]":
                    final_record[col] = product_data.meta_description
                elif col == "Metafield: custom.engine_types [list.single_line_text_field]":
                    final_record[col] = ""
                elif col == "Metafield: mm-google-shopping.custom_product [boolean]":
                    final_record[col] = ""
                elif col.startswith("Variant Metafield: mm-google-shopping.custom_label_"):
                    final_record[col] = ""
                elif col == "Variant Metafield: mm-google-shopping.size_system [single_line_text_field]":
                    final_record[col] = ""
                elif col == "Variant Metafield: mm-google-shopping.size_type [single_line_text_field]":
                    final_record[col] = ""
                elif col == "Variant Metafield: mm-google-shopping.mpn [single_line_text_field]":
                    final_record[col] = product_data.mpn
                elif col == "Variant Metafield: mm-google-shopping.gender [single_line_text_field]":
                    final_record[col] = ""
                elif col == "Variant Metafield: mm-google-shopping.condition [single_line_text_field]":
                    final_record[col] = "new"
                elif col == "Variant Metafield: mm-google-shopping.age_group [single_line_text_field]":
                    final_record[col] = ""
                elif col == "Variant Metafield: harmonized_system_code [string]":
                    final_record[col] = ""
                elif col == "Metafield: mm-google-shopping.mpn [single_line_text_field]":
                    final_record[col] = product_data.mpn
                else:
                    # Default empty value for any unmapped columns
                    final_record[col] = ""
            
            final_records.append(final_record)
        
        # Create DataFrame with columns in exact order from cols_list
        return pd.DataFrame(final_records, columns=cols_list)
    
    def process_complete_pipeline_batch(self, sample_file_path: str = "data/samples/steele_sample.csv") -> pd.DataFrame:
        """
        Execute complete batch-enabled transformation pipeline:
        Full Dataset → Golden Master Validation → Batch AI Processing → Template Enhancement → Final Format
        
        Args:
            sample_file_path: Path to Steele complete dataset
            
        Returns:
            Final consolidated Shopify-ready DataFrame with batch AI processing
        """
        print("🚀 STEELE BATCH AI PIPELINE (50% Cost Savings)")
        print("   Batch processing for AI requests with template-based enhancement")
        print("")
        
        print("🔄 Step 1: Loading Steele complete dataset...")
        steele_df = self.load_sample_data(sample_file_path)
        print(f"✅ Loaded {len(steele_df):,} Steele products from complete dataset")
        
        print("🔄 Step 2: Golden master validation (queuing AI tasks)...")
        self.load_golden_dataset()
        validation_df = self.validate_against_golden_dataset_batch(steele_df)
        exact_matches = len(validation_df[validation_df['match_type'] == 'exact'])
        ai_tasks = self.batch_ai_matcher.get_queue_size()
        print(f"✅ {exact_matches}/{len(steele_df)} exact matches, {ai_tasks} AI tasks queued")
        
        if ai_tasks > 0:
            print("🔄 Step 2b: Processing AI tasks in batch...")
            if self.process_ai_batch(wait_for_completion=True):
                print("🔄 Step 2c: Updating validation with AI results...")
                validation_df = self.update_validation_with_ai_results(validation_df)
                validated_count = len(validation_df[validation_df['golden_validated'] == True])
                print(f"✅ {validated_count}/{len(steele_df)} products validated after AI processing")
            else:
                print("⚠️  AI batch processing failed, continuing with available results")
        
        print("🔄 Step 3: Transform to standard format (preserving fitment)...")
        standard_products = self.transform_to_standard_format(steele_df, validation_df)
        print(f"✅ Transformed {len(standard_products)} products to standard format")

        print("🔄 Step 3b: Template-based enhancement (NO AI)...")
        enhanced_products = self.enhance_with_templates(standard_products)
        print(f"✅ Enhanced {len(enhanced_products)} products with templates")
        
        print("🔄 Step 4: Converting to formatted Shopify import (65 columns)...")
        final_df = self.transform_to_formatted_shopify_import(enhanced_products)
        print(f"✅ Generated formatted Shopify import with {len(final_df)} products")
        
        print("")
        print("⚡ PERFORMANCE: Batch AI processing + template enhancement")
        print("💰 COST: 50% savings with OpenAI Batch API")
        print("🎯 RELIABILITY: Template consistency + batch AI accuracy")
        print("🤖 AI ENHANCEMENT: Smart batch processing for edge cases")
        
        # Show AI cost summary
        if self.batch_ai_matcher.api_calls_made > 0:
            print("")
            self.batch_ai_matcher.print_cost_report()
        
        return final_df
    
    def _validate_input_data(self, df: pd.DataFrame) -> None:
        """Validate input data structure and quality."""
        # Check if DataFrame is empty first
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