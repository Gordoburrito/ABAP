import pandas as pd
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel
import openai
import json

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

class AIVehicleTagGenerator:
    """AI-powered vehicle tag generator that uses actual master_ultimate_golden data (no hallucination)"""
    
    def __init__(self):
        # Initialize OpenAI client
        try:
            self.openai_client = openai.OpenAI()
            self.ai_available = True
        except Exception as e:
            print(f"⚠️  OpenAI not available: {e}")
            self.ai_available = False
        
        # Load FULL golden master dataset for accurate lookups
        self.golden_df = None
        self._load_golden_master()
    
    def _load_golden_master(self):
        """Load complete golden master dataset for accurate vehicle lookups"""
        try:
            golden_path = project_root / "shared" / "data" / "master_ultimate_golden.csv"
            if golden_path.exists():
                # Load complete dataset for accurate lookups
                self.golden_df = pd.read_csv(golden_path)
                print(f"✅ Loaded {len(self.golden_df)} golden master records for accurate vehicle mapping")
            else:
                print("⚠️  Golden master dataset not found")
        except Exception as e:
            print(f"⚠️  Could not load golden master: {e}")
    
    def generate_accurate_vehicle_tag(self, year: str, make: str, model: str, product_name: str) -> str:
        """Generate accurate vehicle tag using structured approach (no hallucination)"""
        
        if not self.ai_available or self.golden_df is None:
            # Fallback to simple format
            return f"{year}_{make.replace(' ', '_')}_{model.replace(' ', '_')}"
        
        try:
            print(f"🤖 Processing: {year} {make} {model}")
            
            # STEP 1: AI extracts/confirms Year and Make from Steele data
            confirmed_year, confirmed_make = self._extract_year_and_make(year, make, model, product_name)
            print(f"   Step 1 - Confirmed: {confirmed_year} {confirmed_make}")
            
            # STEP 2: Query golden master for actual models for this Year/Make
            real_models = self._get_real_models_for_year_make(confirmed_year, confirmed_make)
            print(f"   Step 2 - Found {len(real_models)} real models in golden master")
            if len(real_models) > 0:
                print(f"           Models: {', '.join(real_models[:10])}{'...' if len(real_models) > 10 else ''}")
            
            if len(real_models) == 0:
                print(f"   ⚠️  No models found for {confirmed_year} {confirmed_make} in golden master")
                return f"{confirmed_year}_{confirmed_make.replace(' ', '_')}_UNKNOWN"
            
            # STEP 3: AI selects model(s) from actual golden master options only
            selected_models = self._select_best_model_from_options(
                confirmed_year, confirmed_make, model, product_name, real_models
            )
            
            if isinstance(selected_models, list):
                print(f"   Step 3 - Selected ALL {len(selected_models)} models (generic model detected)")
                # Generate tags for all models
                all_tags = []
                for selected_model in selected_models:
                    tag = f"{confirmed_year}_{confirmed_make.replace(' ', '_')}_{selected_model.replace(' ', '_')}"
                    all_tags.append(tag)
                final_tags = ", ".join(all_tags)
                print(f"   ✅ Final: {final_tags}")
                return final_tags
            else:
                print(f"   Step 3 - Selected: {selected_models}")
                # Generate single tag
                final_tag = f"{confirmed_year}_{confirmed_make.replace(' ', '_')}_{selected_models.replace(' ', '_')}"
                print(f"   ✅ Final: {final_tag}")
                return final_tag
                
        except Exception as e:
            print(f"⚠️  AI vehicle mapping failed: {e}")
            return f"{year}_{make.replace(' ', '_')}_{model.replace(' ', '_')}"
    
    def _extract_year_and_make(self, year: str, make: str, model: str, product_name: str) -> tuple:
        """STEP 1: AI extracts/confirms Year and Make from Steele data"""
        
        try:
            ai_prompt = f"""
You are a vehicle data expert. Extract the correct Year and Make from this Steele vehicle data.

Steele Data:
- Year: {year}
- Make: {make}
- Model: {model}
- Product: {product_name}

Rules:
1. Extract the most accurate Year (4-digit number)
2. Extract the most accurate Make (manufacturer name)
3. For ambiguous cases like "Stutz/Stutz", the Make is "Stutz"
4. Clean up and standardize the Make name
5. Return in exact format: YEAR|MAKE

Examples:
Input: Year=1928, Make=Stutz, Model=Stutz → Output: 1928|Stutz
Input: Year=1930, Make=Durant, Model=Model 6-14 → Output: 1930|Durant

Return ONLY: YEAR|MAKE
"""

            response = self.openai_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": ai_prompt}],
                temperature=0.1,
                max_tokens=20
            )
            
            ai_response = response.choices[0].message.content.strip()
            
            # Parse the response
            if '|' in ai_response:
                confirmed_year, confirmed_make = ai_response.split('|', 1)
                return confirmed_year.strip(), confirmed_make.strip()
            else:
                # Fallback to original data
                return year, make
                
        except Exception as e:
            print(f"⚠️  Year/Make extraction failed: {e}")
            return year, make
    
    def _get_real_models_for_year_make(self, year: str, make: str) -> list:
        """STEP 2: Query golden master for actual models for this Year/Make combination"""
        
        try:
            year_int = int(year)
            
            # Find exact matches in golden master
            matches = self.golden_df[
                (self.golden_df['Year'] == year_int) &
                (self.golden_df['Make'].str.upper() == make.upper())
            ]
            
            # Get unique models
            real_models = matches['Model'].unique().tolist()
            
            # Clean and sort models
            real_models = [model for model in real_models if pd.notna(model)]
            real_models.sort()
            
            return real_models
            
        except Exception as e:
            print(f"⚠️  Golden master lookup failed: {e}")
            return []
    
    def _select_best_model_from_options(self, year: str, make: str, original_model: str, 
                                      product_name: str, real_models: list):
        """STEP 3: AI selects model(s) from actual golden master options only"""
        
        if len(real_models) == 1:
            return real_models[0]
        
        # Check if original model is generic (same as make or very simple)
        is_generic_model = (
            original_model.lower() == make.lower() or  # "Stutz" model for "Stutz" make
            len(original_model.split()) == 1 and len(real_models) > 1  # Single word model with multiple options
        )
        
        if is_generic_model and len(real_models) > 1:
            print(f"           Generic model '{original_model}' detected with {len(real_models)} specific variants")
            print(f"           Returning ALL models for broad compatibility")
            return real_models  # Return all models as list
        
        # For specific models, use AI to select the best match
        try:
            # Limit to top 10 options for AI prompt efficiency
            model_options = real_models[:10]
            
            ai_prompt = f"""
You are a vehicle data expert. Select the best model from these REAL options from the golden master dataset.

Steele Data:
- Year: {year}
- Make: {make}
- Original Model: {original_model}
- Product: {product_name}

REAL Model Options (from golden master):
{chr(10).join([f"- {model}" for model in model_options])}

Rules:
1. You MUST select from the provided options only (no hallucination)
2. Choose the model that best matches the original Steele model "{original_model}"
3. Consider the product context: "{product_name}"
4. If uncertain, choose the first/most common option

Return ONLY the selected model name exactly as shown in the options.
"""

            response = self.openai_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": ai_prompt}],
                temperature=0.1,
                max_tokens=50
            )
            
            selected_model = response.choices[0].message.content.strip()
            
            # Validate selection is in real options
            if selected_model in real_models:
                return selected_model
            else:
                print(f"⚠️  AI selected invalid model: {selected_model}, using first option")
                return real_models[0]
                
        except Exception as e:
            print(f"⚠️  Model selection failed: {e}")
            return real_models[0]  # Return first option as fallback

class SteeleDataTransformer:
    """
    Main transformer class for Steele data source - COMPLETE FITMENT DATA with AI Vehicle Tag Generation
    Implements: Sample Data → Golden Master Validation → Template Enhancement → AI Vehicle Tags → Final Format
    
    Uses AI for accurate vehicle tag generation to match master_ultimate_golden format
    """
    
    def __init__(self, use_ai: bool = True):
        """
        Initialize the transformer for complete fitment data with AI vehicle tag generation.
        
        Args:
            use_ai: True to use AI for accurate vehicle tag generation (recommended)
        """
        self.use_ai = use_ai
        self.vendor_name = "Steele"
        self.golden_df = None
        self.template_generator = TemplateGenerator()
        
        # Initialize AI vehicle tag generator
        if self.use_ai:
            self.ai_tag_generator = AIVehicleTagGenerator()
            print("🤖 AI vehicle tag generation enabled for accurate master_ultimate_golden mapping")
        else:
            self.ai_tag_generator = None
            print("⚠️  AI disabled - using simple vehicle tag generation")
    
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
        Step 2: Load golden master dataset for vehicle validation (ONLY CRITICAL STEP).
        
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
    
    def validate_against_golden_dataset(self, steele_df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 2b: Validate Steele vehicles against golden master dataset (ONLY CRITICAL STEP).
        
        Args:
            steele_df: Steele data to validate
            
        Returns:
            DataFrame with validation results
        """
        if self.golden_df is None:
            self.load_golden_dataset()
        
        validation_results = []
        
        for idx, row in steele_df.iterrows():
            try:
                year = int(row['Year']) if pd.notna(row['Year']) else None
                make = str(row['Make']) if pd.notna(row['Make']) else None
                model = str(row['Model']) if pd.notna(row['Model']) else None
                
                if year and make and model:
                    # Check if combination exists in golden dataset
                    matches = self.golden_df[
                        (self.golden_df['year'] == year) &
                        (self.golden_df['make'] == make) &
                        (self.golden_df['model'] == model)
                    ]
                    
                    validation_results.append({
                        'steele_row_index': idx,
                        'stock_code': row['StockCode'],
                        'year': year,
                        'make': make,
                        'model': model,
                        'golden_validated': len(matches) > 0,
                        'golden_matches': len(matches),
                        'car_ids': matches['car_id'].unique().tolist() if len(matches) > 0 else []
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
                        'error': 'incomplete_vehicle_data'
                    })
                    
            except Exception as e:
                validation_results.append({
                    'steele_row_index': idx,
                    'stock_code': row.get('StockCode', 'unknown'),
                    'golden_validated': False,
                    'error': str(e)
                })
        
        return pd.DataFrame(validation_results)
    
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
            
            product_data = ProductData(
                title=str(steele_row['Product Name']),
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
    
    def transform_to_formatted_shopify_import(self, enhanced_products: List[ProductData]) -> pd.DataFrame:
        """
        FORMATTED STEP: Transform template-enhanced data to complete Shopify import format.
        Generates ALL 65 columns from product_import-column-requirements.py in correct order.
        
        Args:
            enhanced_products: List of template-enhanced ProductData
            
        Returns:
            DataFrame in complete 65-column Shopify import format (FORMATTED STEP)
        """
        final_records = []
        
        for product_data in enhanced_products:
            # Generate vehicle tags from existing fitment data
            vehicle_tag = self._generate_vehicle_tag(product_data)
            
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
                    final_record[col] = vehicle_tag
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
    
    def process_complete_pipeline_no_ai(self, sample_file_path: str = "data/samples/steele_sample.csv") -> pd.DataFrame:
        """
        Execute complete transformation pipeline with AI vehicle tag generation:
        Sample Data → Golden Master Validation → Template Enhancement → AI Vehicle Tags → Final Format → Consolidation
        
        Args:
            sample_file_path: Path to Steele sample data
            
        Returns:
            Final consolidated Shopify-ready DataFrame with comprehensive vehicle tags
        """
        print("🚀 STEELE PIPELINE with AI VEHICLE TAG GENERATION + CONSOLIDATION")
        print("   Template-based processing + AI-powered accurate vehicle tags + Product consolidation")
        print("")
        
        print("🔄 Step 1: Loading Steele sample data...")
        steele_df = self.load_sample_data(sample_file_path)
        print(f"✅ Loaded {len(steele_df)} Steele products")
        
        print("🔄 Step 2: Golden master validation (ONLY CRITICAL STEP)...")
        self.load_golden_dataset()
        validation_df = self.validate_against_golden_dataset(steele_df)
        validated_count = len(validation_df[validation_df['golden_validated'] == True])
        print(f"✅ {validated_count}/{len(steele_df)} products validated against golden dataset")
        
        print("🔄 Step 3: Transform to standard format (preserving fitment)...")
        standard_products = self.transform_to_standard_format(steele_df, validation_df)
        print(f"✅ Transformed {len(standard_products)} products to standard format")
        
        print("🔄 Step 3b: Template-based enhancement (NO AI)...")
        enhanced_products = self.enhance_with_templates(standard_products)
        print(f"✅ Enhanced {len(enhanced_products)} products with templates")
        
        print("🔄 Step 4: Converting to formatted Shopify import with AI vehicle tags...")
        formatted_df = self.transform_to_formatted_shopify_import(enhanced_products)
        print(f"✅ Generated formatted Shopify import with {len(formatted_df)} products")
        
        print("🔄 Step 5: Consolidating products by unique ID...")
        final_df = self.consolidate_products_by_unique_id(formatted_df)
        print(f"✅ Final consolidated output with {len(final_df)} unique products")
        
        print("")
        if self.use_ai:
            print("🤖 AI VEHICLE TAG GENERATION: Accurate master_ultimate_golden mapping")
        print("📦 PRODUCT CONSOLIDATION: One product per SKU with comprehensive tags")
        print("⚡ PERFORMANCE: Fast template-based processing with AI vehicle tags")
        print("💰 COST: Low (AI only for vehicle tag generation)")
        print("🎯 ACCURACY: High accuracy vehicle tags matching master_ultimate_golden")
        
        return final_df
    
    def _generate_vehicle_tag(self, product_data: ProductData) -> str:
        """Generate accurate vehicle tag using AI to map to master_ultimate_golden format."""
        if (product_data.make != "NONE" and 
            product_data.model != "NONE" and 
            product_data.year_min != "Unknown"):
            
            if self.use_ai and self.ai_tag_generator:
                # Use AI for accurate vehicle tag generation
                return self.ai_tag_generator.generate_accurate_vehicle_tag(
                    year=product_data.year_min,
                    make=product_data.make,
                    model=product_data.model,
                    product_name=product_data.title
                )
            else:
                # Fallback to simple format
                make = product_data.make.replace(' ', '_')
                model = product_data.model.replace(' ', '_')
                return f"{product_data.year_min}_{make}_{model}"
        else:
            return ""
    
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
    
    def validate_output(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Validate transformed output against complete Shopify requirements from product_import.
        
        Args:
            df: Transformed DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'errors': [],
            'warnings': [],
            'info': []
        }
        
        # Check that we have ALL required columns in correct order
        expected_columns = cols_list
        actual_columns = list(df.columns)
        
        if actual_columns != expected_columns:
            validation_results['errors'].append("Column order/set does not match product_import requirements")
            missing_cols = set(expected_columns) - set(actual_columns)
            extra_cols = set(actual_columns) - set(expected_columns)
            if missing_cols:
                validation_results['errors'].append(f"Missing columns: {missing_cols}")
            if extra_cols:
                validation_results['errors'].append(f"Extra columns: {extra_cols}")
        
        # Validate data quality
        if len(df) == 0:
            validation_results['errors'].append("Output DataFrame is empty")
        
        # Check required always fields have data
        for col in ["Title", "Body HTML", "Vendor", "Tags"]:
            if col in df.columns:
                empty_count = df[col].isnull().sum() + (df[col] == "").sum()
                if empty_count > 0:
                    validation_results['warnings'].append(f"{empty_count} rows have empty {col}")
        
        # Check price fields
        if 'Variant Price' in df.columns:
            invalid_prices = df[df['Variant Price'] <= 0]
            if len(invalid_prices) > 0:
                validation_results['warnings'].append(f"{len(invalid_prices)} rows have invalid prices")
        
        # Check meta title lengths
        if 'Metafield: title_tag [string]' in df.columns:
            long_titles = df[df['Metafield: title_tag [string]'].str.len() > 60]
            if len(long_titles) > 0:
                validation_results['warnings'].append(f"{len(long_titles)} meta titles exceed 60 characters")
        
        # Check meta description lengths
        if 'Metafield: description_tag [string]' in df.columns:
            long_descriptions = df[df['Metafield: description_tag [string]'].str.len() > 160]
            if len(long_descriptions) > 0:
                validation_results['warnings'].append(f"{len(long_descriptions)} meta descriptions exceed 160 characters")
        
        validation_results['info'].append(f"Processed {len(df)} products")
        validation_results['info'].append(f"Vendor: {self.vendor_name}")
        validation_results['info'].append(f"Total columns: {len(df.columns)}")
        validation_results['info'].append(f"Matches product_import requirements: {len(validation_results['errors']) == 0}")
        
        return validation_results
    
    def save_output(self, df: pd.DataFrame, output_path: str = "data/transformed/steele_transformed.csv") -> str:
        """
        Save transformed data to file.
        
        Args:
            df: DataFrame to save
            output_path: Output file path (relative to Steele directory)
            
        Returns:
            Path to saved file
        """
        # Convert relative path to absolute path from Steele directory
        if not os.path.isabs(output_path):
            output_path = str(steele_root / output_path)
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        
        return output_path
    
    def consolidate_products_by_unique_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        CONSOLIDATION STEP: Group products by unique ID and combine all vehicle tags.
        Creates one final product per unique StockCode/SKU with comprehensive vehicle tags.
        
        Args:
            df: DataFrame with potentially multiple rows per product (different vehicle tags)
            
        Returns:
            DataFrame with one row per unique product, consolidated tags
        """
        print("🔄 Step 5: Consolidating products by unique ID and combining tags...")
        
        # Group by unique product identifier (Variant SKU contains StockCode/PartNumber)
        sku_column = 'Variant SKU'
        
        if sku_column not in df.columns:
            print("⚠️  No SKU column found, skipping consolidation")
            return df
        
        consolidated_records = []
        
        # Group by SKU to find products with same ID but different vehicle tags
        grouped = df.groupby(sku_column)
        
        for sku, group in grouped:
            if len(group) == 1:
                # Single product, no consolidation needed
                consolidated_records.append(group.iloc[0].to_dict())
            else:
                # Multiple rows for same product - consolidate tags
                print(f"   📦 Consolidating {len(group)} variants for SKU: {sku}")
                
                # Take the first row as base and combine tags
                base_record = group.iloc[0].to_dict()
                
                # Collect all unique tags from all variants
                all_tags = []
                for _, row in group.iterrows():
                    tags = str(row['Tags']).strip()
                    if tags and tags != 'nan':
                        # Split comma-separated tags and add to collection
                        tag_list = [tag.strip() for tag in tags.split(',')]
                        all_tags.extend(tag_list)
                
                # Remove duplicates while preserving order
                unique_tags = []
                seen = set()
                for tag in all_tags:
                    if tag not in seen and tag:
                        unique_tags.append(tag)
                        seen.add(tag)
                
                # Combine into comprehensive tag string
                base_record['Tags'] = ', '.join(unique_tags)
                
                # Optional: Combine other fields that might vary
                # For descriptions, take the longest one
                descriptions = [str(row['Body HTML']) for _, row in group.iterrows() if str(row['Body HTML']) != 'nan']
                if descriptions:
                    base_record['Body HTML'] = max(descriptions, key=len)
                
                # For collections, combine unique values
                collections = [str(row['Custom Collections']) for _, row in group.iterrows() if str(row['Custom Collections']) != 'nan']
                unique_collections = list(set([c for c in collections if c and c != 'nan']))
                if unique_collections:
                    base_record['Custom Collections'] = ', '.join(unique_collections)
                
                consolidated_records.append(base_record)
                print(f"       🏷️  Combined tags: {len(unique_tags)} unique vehicle applications")
        
        # Create new DataFrame with consolidated records
        consolidated_df = pd.DataFrame(consolidated_records, columns=df.columns)
        
        print(f"✅ Consolidated {len(df)} rows → {len(consolidated_df)} unique products")
        print(f"   Saved {len(df) - len(consolidated_df)} duplicate product rows")
        
        return consolidated_df 