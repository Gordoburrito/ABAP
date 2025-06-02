import pandas as pd
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

class ProductData(BaseModel):
    """AI-friendly intermediate format that uses fewer tokens"""
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

class SteeleDataTransformer:
    """
    Main transformer class for Steele data source.
    Implements: Sample Data → Golden Master → AI-Friendly Format → Final Tagged Format
    """
    
    def __init__(self, use_ai: bool = True):
        """
        Initialize the transformer.
        
        Args:
            use_ai: Whether to use OpenAI API for enhancements
        """
        self.use_ai = use_ai
        self.vendor_name = "Steele"
        self.golden_df = None
        
        if use_ai:
            load_dotenv()
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.openai_client = OpenAI(api_key=api_key)
            else:
                print("Warning: OPENAI_API_KEY not found, AI features disabled")
                self.use_ai = False
    
    def load_sample_data(self, file_path: str = "data/samples/steele_sample.csv") -> pd.DataFrame:
        """
        Step 1: Load Steele sample data for processing.
        
        Args:
            file_path: Path to the sample CSV file
            
        Returns:
            DataFrame with loaded data
            
        Raises:
            FileNotFoundError: If sample file doesn't exist
            ValueError: If data format is invalid
        """
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
    
    def validate_against_golden_dataset(self, steele_df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 2b: Validate Steele vehicles against golden master dataset.
        
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
    
    def transform_to_ai_friendly_format(self, steele_df: pd.DataFrame, validation_df: pd.DataFrame) -> List[ProductData]:
        """
        Step 3: Transform to AI-friendly intermediate format with fewer tokens.
        
        Args:
            steele_df: Original Steele data
            validation_df: Golden dataset validation results
            
        Returns:
            List of ProductData objects for AI processing
        """
        ai_friendly_products = []
        
        for idx, steele_row in steele_df.iterrows():
            # Get validation result for this row
            validation_row = validation_df[validation_df['steele_row_index'] == idx]
            is_golden_validated = len(validation_row) > 0 and validation_row.iloc[0]['golden_validated']
            
            # Create AI-friendly format
            product_data = ProductData(
                title=str(steele_row['Product Name']),
                year_min=str(int(steele_row['Year'])) if pd.notna(steele_row['Year']) else "1800",
                year_max=str(int(steele_row['Year'])) if pd.notna(steele_row['Year']) else "1800", 
                make=str(steele_row['Make']) if pd.notna(steele_row['Make']) and is_golden_validated else "NONE",
                model=str(steele_row['Model']) if pd.notna(steele_row['Model']) and is_golden_validated else "NONE",
                mpn=str(steele_row['StockCode']),
                cost=float(steele_row['Dealer Price']) if pd.notna(steele_row['Dealer Price']) else 0.0,
                price=float(steele_row['MAP']) if pd.notna(steele_row['MAP']) else 0.0,
                body_html=str(steele_row['Description'])[:200] + "..." if len(str(steele_row['Description'])) > 200 else str(steele_row['Description']),
                collection="Accessories",  # Will be enhanced by AI
                product_type="Automotive Part"
            )
            
            ai_friendly_products.append(product_data)
        
        return ai_friendly_products
    
    def enhance_with_ai(self, product_data_list: List[ProductData]) -> List[ProductData]:
        """
        Step 3b: Enhance AI-friendly data using OpenAI API.
        
        Args:
            product_data_list: List of ProductData to enhance
            
        Returns:
            Enhanced ProductData list
        """
        if not self.use_ai or not hasattr(self, 'openai_client'):
            print("AI enhancement skipped - using defaults")
            return product_data_list
        
        enhanced_products = []
        
        for product_data in product_data_list:
            try:
                # Prepare concise AI input (fewer tokens)
                ai_input = {
                    'title': product_data.title,
                    'description': product_data.body_html[:100] + "...",  # Truncated for efficiency
                    'year_range': f"{product_data.year_min}-{product_data.year_max}",
                    'vehicle': f"{product_data.make} {product_data.model}" if product_data.make != "NONE" else "Unknown"
                }
                
                # Simple AI enhancement for collection and meta fields
                if self._should_enhance_with_ai(product_data):
                    enhanced_data = self._enhance_single_product_with_ai(ai_input)
                    
                    # Update product data with AI enhancements
                    product_data.collection = enhanced_data.get('collection', product_data.collection)
                    product_data.meta_title = enhanced_data.get('meta_title', self._generate_basic_meta_title(product_data))
                    product_data.meta_description = enhanced_data.get('meta_description', self._generate_basic_meta_description(product_data))
                else:
                    # Use basic generation without AI
                    product_data.meta_title = self._generate_basic_meta_title(product_data)
                    product_data.meta_description = self._generate_basic_meta_description(product_data)
                
                enhanced_products.append(product_data)
                
            except Exception as e:
                print(f"AI enhancement failed for {product_data.title}: {e}")
                # Use fallback generation
                product_data.meta_title = self._generate_basic_meta_title(product_data)
                product_data.meta_description = self._generate_basic_meta_description(product_data)
                enhanced_products.append(product_data)
        
        return enhanced_products
    
    def transform_to_final_tagged_format(self, enhanced_products: List[ProductData]) -> pd.DataFrame:
        """
        Step 4: Transform AI-enhanced data to final Shopify format with tags.
        
        Args:
            enhanced_products: List of AI-enhanced ProductData
            
        Returns:
            DataFrame in final Shopify import format
        """
        final_records = []
        
        for product_data in enhanced_products:
            # Generate vehicle tags
            vehicle_tag = self._generate_vehicle_tag(product_data)
            
            final_record = {
                'Title': product_data.title,
                'Body HTML': product_data.body_html,
                'Vendor': self.vendor_name,
                'Tags': vehicle_tag,
                'Variant SKU': product_data.mpn,
                'Variant Price': product_data.price,
                'Variant Cost': product_data.cost,
                'Variant Barcode': '',  # To be populated if available
                'Command': 'MERGE',
                'Image Src': '',
                'Image Command': 'MERGE', 
                'Image Position': 1,
                'Image Alt Text': product_data.title,
                'Metafield: title_tag [string]': product_data.meta_title,
                'Metafield: description_tag [string]': product_data.meta_description
            }
            
            final_records.append(final_record)
        
        return pd.DataFrame(final_records)
    
    def process_complete_pipeline(self, sample_file_path: str = "data/samples/steele_sample.csv") -> pd.DataFrame:
        """
        Execute complete transformation pipeline:
        Sample Data → Golden Master → AI-Friendly Format → Final Tagged Format
        
        Args:
            sample_file_path: Path to Steele sample data
            
        Returns:
            Final Shopify-ready DataFrame
        """
        print("🔄 Step 1: Loading Steele sample data...")
        steele_df = self.load_sample_data(sample_file_path)
        print(f"✅ Loaded {len(steele_df)} Steele products")
        
        print("🔄 Step 2: Loading and validating against golden dataset...")
        self.load_golden_dataset()
        validation_df = self.validate_against_golden_dataset(steele_df)
        validated_count = len(validation_df[validation_df['golden_validated'] == True])
        print(f"✅ {validated_count}/{len(steele_df)} products validated against golden dataset")
        
        print("🔄 Step 3: Transforming to AI-friendly format...")
        ai_friendly_products = self.transform_to_ai_friendly_format(steele_df, validation_df)
        print(f"✅ Transformed {len(ai_friendly_products)} products to AI-friendly format")
        
        print("🔄 Step 3b: Enhancing with AI...")
        enhanced_products = self.enhance_with_ai(ai_friendly_products)
        print(f"✅ Enhanced {len(enhanced_products)} products with AI")
        
        print("🔄 Step 4: Converting to final tagged format...")
        final_df = self.transform_to_final_tagged_format(enhanced_products)
        print(f"✅ Generated final Shopify format with {len(final_df)} products")
        
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
    
    def _should_enhance_with_ai(self, product_data: ProductData) -> bool:
        """Determine if product should be enhanced with AI."""
        # Only use AI for products with valid vehicle data
        return (product_data.make != "NONE" and 
                product_data.model != "NONE" and 
                product_data.year_min != "1800")
    
    def _enhance_single_product_with_ai(self, ai_input: dict) -> dict:
        """Enhance single product using AI (placeholder for actual implementation)."""
        # This would implement actual AI enhancement
        # For now, return reasonable defaults
        return {
            'collection': 'Engine' if 'engine' in ai_input['title'].lower() else 'Accessories',
            'meta_title': f"{ai_input['title']} - {ai_input['vehicle']}",
            'meta_description': f"Quality {ai_input['title']} for {ai_input['vehicle']} vehicles."
        }
    
    def _generate_vehicle_tag(self, product_data: ProductData) -> str:
        """Generate vehicle compatibility tag."""
        if (product_data.make != "NONE" and 
            product_data.model != "NONE" and 
            product_data.year_min != "1800"):
            
            make = product_data.make.replace(' ', '_')
            model = product_data.model.replace(' ', '_')
            return f"{product_data.year_min}_{make}_{model}"
        else:
            return ""
    
    def _generate_basic_meta_title(self, product_data: ProductData) -> str:
        """Generate basic meta title without AI."""
        if product_data.make != "NONE":
            title = f"{product_data.title} - {product_data.year_min} {product_data.make}"
        else:
            title = product_data.title
        
        return title[:60] if len(title) > 60 else title
    
    def _generate_basic_meta_description(self, product_data: ProductData) -> str:
        """Generate basic meta description without AI."""
        if product_data.make != "NONE":
            desc = f"Quality {product_data.title} for {product_data.year_min} {product_data.make} {product_data.model} vehicles."
        else:
            desc = f"Quality automotive {product_data.title}."
        
        return desc[:160] if len(desc) > 160 else desc
    
    def validate_output(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Validate transformed output against Shopify requirements.
        
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
        
        # Check required columns
        required_columns = [
            'Title', 'Body HTML', 'Vendor', 'Tags', 'Variant Price', 'Variant Cost',
            'Metafield: title_tag [string]', 'Metafield: description_tag [string]'
        ]
        
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            validation_results['errors'].append(f"Missing required columns: {missing_columns}")
        
        # Validate data quality
        if len(df) == 0:
            validation_results['errors'].append("Output DataFrame is empty")
        
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
        
        validation_results['info'].append(f"Processed {len(df)} products")
        validation_results['info'].append(f"Vendor: {self.vendor_name}")
        
        return validation_results
    
    def save_output(self, df: pd.DataFrame, output_path: str = "data/transformed/steele_transformed.csv") -> str:
        """
        Save transformed data to file.
        
        Args:
            df: DataFrame to save
            output_path: Output file path
            
        Returns:
            Path to saved file
        """
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        
        return output_path 