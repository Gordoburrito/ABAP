import json
import re
import time
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from pydantic import BaseModel, Field, field_validator
from openai import OpenAI
import logging
from utils.exceptions import AIExtractionError

logger = logging.getLogger(__name__)


class ProductData(BaseModel):
    """Pydantic model for validated product data"""
    
    title: str = Field(..., min_length=1, max_length=200)
    year_min: int = Field(..., ge=1900, le=2024)
    year_max: int = Field(..., ge=1900, le=2024)
    make: str = Field(..., min_length=1, max_length=50)
    model: str = Field(..., min_length=1, max_length=50)
    mpn: str = Field(..., min_length=1, max_length=50)
    cost: float = Field(..., ge=0)
    price: float = Field(..., ge=0)
    body_html: str = Field(..., min_length=1)
    collection: str = Field(..., min_length=1, max_length=100)
    product_type: str = Field(..., min_length=1, max_length=50)
    meta_title: str = Field(..., min_length=1, max_length=60)
    meta_description: str = Field(..., min_length=1, max_length=160)
    
    @field_validator('year_max')
    @classmethod
    def validate_year_range(cls, v, info):
        """Validate that year_max >= year_min"""
        if 'year_min' in info.data and v < info.data['year_min']:
            raise ValueError('year_max must be >= year_min')
        return v
    
    @field_validator('price')
    @classmethod
    def validate_price_vs_cost(cls, v, info):
        """Validate that price >= cost"""
        if 'cost' in info.data and v < info.data['cost']:
            raise ValueError('price must be >= cost')
        return v


class VehicleCompatibility(BaseModel):
    """Single vehicle compatibility entry"""
    year_min: int = Field(..., ge=1900, le=2024)
    year_max: int = Field(..., ge=1900, le=2024)
    make: str = Field(..., min_length=1, max_length=50)
    model: str = Field(..., min_length=1, max_length=100)
    
    @field_validator('year_max')
    @classmethod
    def validate_year_range(cls, v, info):
        """Validate that year_max >= year_min"""
        if 'year_min' in info.data and v < info.data['year_min']:
            raise ValueError('year_max must be >= year_min')
        return v

class MultiVehicleData(BaseModel):
    """Pydantic model for Pass 1 multi-vehicle extraction"""
    
    vehicles: List[VehicleCompatibility] = Field(..., min_items=1, max_items=10)
    primary_vehicle_index: int = Field(default=0, ge=0)
    
    @field_validator('primary_vehicle_index')
    @classmethod
    def validate_primary_index(cls, v, info):
        """Validate primary index is within vehicles list"""
        if 'vehicles' in info.data and v >= len(info.data['vehicles']):
            return 0  # Default to first vehicle
        return v

class RefinedMultiVehicleData(BaseModel):
    """Pydantic model for Pass 2 refined multi-vehicle data"""
    
    vehicles: List[VehicleCompatibility] = Field(..., min_items=1, max_items=20)
    title: str = Field(..., min_length=1, max_length=200)
    mpn: str = Field(..., min_length=1, max_length=50)
    cost: float = Field(..., ge=0)
    price: float = Field(..., ge=0)
    body_html: str = Field(..., min_length=1)
    collection: str = Field(..., min_length=1, max_length=100)
    product_type: str = Field(..., min_length=1, max_length=50)
    meta_title: str = Field(..., min_length=1, max_length=60)
    meta_description: str = Field(..., min_length=1, max_length=160)
    
    @field_validator('price')
    @classmethod
    def validate_price_vs_cost(cls, v, info):
        """Validate that price >= cost"""
        if 'cost' in info.data and v < info.data['cost']:
            raise ValueError('price must be >= cost')
        return v


class TwoPassAIEngine:
    """Implements two-pass AI strategy with golden master filtering"""
    
    def __init__(self, client: OpenAI, golden_df: pd.DataFrame):
        self.client = client
        # Normalize column names to lowercase for consistency
        self.golden_df = golden_df.copy()
        self.golden_df.columns = [col.lower().replace(' ', '_') for col in self.golden_df.columns]
        self.logger = logging.getLogger(__name__)
    
    def extract_initial_vehicle_info(self, product_info: str) -> Dict[str, Any]:
        """
        Pass 1: Extract ALL vehicle compatibility information from product description
        
        Args:
            product_info: Raw product information string
            
        Returns:
            Dict: Initial vehicle data with all compatible vehicles
        """
        # Get broad context from golden master
        valid_makes = sorted([x for x in self.golden_df['make'].unique() if pd.notna(x)][:50])  # Limit for token efficiency
        valid_models = sorted([x for x in self.golden_df['model'].unique() if pd.notna(x)][:100])  # Limit for token efficiency
        
        prompt = f"""
Extract ALL vehicle compatibility information from this automotive parts product description.
CRITICAL: You MUST extract EVERY SINGLE model mentioned - never skip any models in lists.

Product Information:
{product_info}

AVAILABLE GOLDEN MASTER DATA FOR REFERENCE (use these EXACT model names):
Valid Makes: {', '.join(valid_makes[:25])}
Valid Models: {', '.join(valid_models[:40])}

IMPORTANT: When you see generic terms like "Commander" or "President", look through the valid models above for specific variants like "6-7A Commander", "State President", etc. and use those instead.

INSTRUCTIONS - FOLLOW EXACTLY:
1. Extract ALL vehicle combinations mentioned in the description
2. Pay special attention to comma-separated model lists:
   - "Model K, KA, KB and Series K" = 4 separate models: Model K, KA, KB, Series K  
   - "810, 812" = 2 separate models: 810, 812
   - "Commander, 6-8A Commander, 8-4C State President, 8-5C State President, 9A Commander" = 5 separate models
3. For EACH MAKE, create separate vehicle entries for EVERY model mentioned
4. Handle complex patterns like:
   - "1931-1939 Lincoln Model K, KA, KB and Series K models" → 4 Lincoln entries: "Model K", "Model KA", "Model KB", "Series K" (preserve "Model" prefix for KA/KB)
   - "1938-1939 Studebaker 6-7A Commander, 6-8A Commander, 8-4C State President, 8-5C State President, 9A Commander" → 5 Studebaker entries (1938-1939 each)
   - "1936-1937 Cord: 810, 812; 1940-1941 Graham Hollywood and Hupmobile Skylark" → 4 entries total
5. CRITICAL MODEL MATCHING - ALWAYS use golden master data to refine model names:
   - If you extract "K" but golden master shows "Model K", use "Model K"
   - If you extract "Commander" but golden master shows "6-7A Commander, 6-8A Commander", extract ALL Commander variants: "6-7A Commander", "6-8A Commander" 
   - If you extract "President" but golden master shows "State President", use "State President"
   - If golden master shows multiple variants (like "6-7A Commander, 6-8A Commander"), extract ALL variants as separate vehicles
   - NEVER use generic names if specific variants exist in golden master - always prefer the EXACT model names from golden master data
6. For Lincoln models like "Model K, KA, KB", preserve "Model" prefix: extract as "Model K", "Model KA", "Model KB" NOT "Model K", "KA", "KB"
7. NEVER combine models - each model gets its own vehicle entry
8. For universal parts, use ONE entry: year_min: 1900, year_max: 2024, make: "Universal", model: "Universal"
9. Set primary_vehicle_index to the most prominent/first vehicle (usually index 0)

EXAMPLES OF CORRECT EXTRACTION WITH GOLDEN MASTER MATCHING:
Input: "1931-1939 Lincoln Model K, KA, KB and Series K models"
Golden Master Models: Model K, Model KA, Model KB, Series K, Zephyr
Output: 4 vehicles - Model K, Model KA, Model KB, Series K (all 1931-1939 Lincoln) - NOTE: KA becomes "Model KA", KB becomes "Model KB"

Input: "1938-1939 Studebaker Commander and President models"
Golden Master Models: 6-7A Commander, 6-8A Commander, State President, Dictator
Output: 3 vehicles - 6-7A Commander, 6-8A Commander, State President (refined from generic "Commander" and "President" to ALL specific variants in golden master)

Input: "Ford K models"  
Golden Master Models: Model A, Model K, Model T
Output: 1 vehicle - Model K (refined from "K" using golden master)

Return ONLY a JSON object with this structure:
{{
    "vehicles": [
        {{"year_min": 1931, "year_max": 1939, "make": "Lincoln", "model": "Model K"}},
        {{"year_min": 1931, "year_max": 1939, "make": "Lincoln", "model": "Model KA"}},
        {{"year_min": 1931, "year_max": 1939, "make": "Lincoln", "model": "Model KB"}},
        {{"year_min": 1931, "year_max": 1939, "make": "Lincoln", "model": "Series K"}}
    ],
    "primary_vehicle_index": 0
}}
"""
        
        try:
            response = self.client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert automotive parts cataloger. Extract EVERY SINGLE vehicle mentioned, then refine model names using the golden master data provided. Match short names like 'K' to full names like 'Model K' when available."},
                    {"role": "user", "content": prompt}
                ],
                response_format=MultiVehicleData,
                temperature=0.1
            )
            
            multi_data = response.choices[0].message.parsed
            vehicles_list = multi_data.model_dump()['vehicles']
            primary_index = multi_data.primary_vehicle_index
            
            # For backward compatibility, return primary vehicle data in old format
            # but also include all vehicles for later processing
            primary_vehicle = vehicles_list[primary_index] if vehicles_list else {
                'year_min': 1900, 'year_max': 2024, 'make': 'Universal', 'model': 'Universal'
            }
            
            result = {
                'year_min': primary_vehicle['year_min'],
                'year_max': primary_vehicle['year_max'],
                'make': primary_vehicle['make'],
                'model': primary_vehicle['model'],
                'all_vehicles': vehicles_list,  # Include all vehicles for multi-vehicle processing
                'primary_vehicle_index': primary_index
            }
            
            # Add title and body_html for debugging
            result['title'] = f"AI Generated Title for {result['make']} {result['model']}"
            result['body_html'] = f"<p>AI Generated description for {result['make']} {result['model']} ({result['year_min']}-{result['year_max']})</p>"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Pass 1 multi-vehicle extraction failed: {str(e)}")
            # Fallback to universal
            return {
                'year_min': 1900,
                'year_max': 2024,
                'make': 'Universal',
                'model': 'Universal',
                'title': 'Universal Product',
                'body_html': '<p>Universal automotive part</p>',
                'all_vehicles': [{'year_min': 1900, 'year_max': 2024, 'make': 'Universal', 'model': 'Universal'}],
                'primary_vehicle_index': 0
            }
    
    def filter_golden_master_context(self, year_min: int, year_max: int, make: str) -> pd.DataFrame:
        """
        Filter golden master data to provide accurate context for Pass 2
        
        Args:
            year_min: Minimum year from Pass 1
            year_max: Maximum year from Pass 1
            make: Make from Pass 1
            
        Returns:
            pd.DataFrame: Filtered golden master data
        """
        try:
            # Clean the golden_df year column
            golden_clean = self.golden_df.dropna(subset=['year', 'make', 'model'])
            
            if make.lower() in ['universal', 'all', 'unknown']:
                # For universal parts, provide broader context but still filter by year
                filtered_golden = golden_clean[
                    (golden_clean["year"].astype(int) >= year_min) & 
                    (golden_clean["year"].astype(int) <= year_max)
                ]
            else:
                # Filter by year range and make
                filtered_golden = golden_clean[
                    (golden_clean["year"].astype(int) >= year_min) & 
                    (golden_clean["year"].astype(int) <= year_max) &
                    (golden_clean["make"] == make)
                ]
            
            return filtered_golden
            
        except Exception as e:
            self.logger.error(f"Golden master filtering failed: {str(e)}")
            return self.golden_df.head(100)  # Fallback to sample data
    
    def get_valid_models_for_context(self, filtered_df: pd.DataFrame) -> List[str]:
        """
        Extract valid models from filtered golden master for AI context
        
        Args:
            filtered_df: Filtered golden master DataFrame
            
        Returns:
            List[str]: Valid models for AI context
        """
        return sorted(filtered_df["model"].unique().tolist())
    
    def refine_with_golden_master(self, initial_data: Dict, product_info: str) -> Dict[str, Any]:
        """
        Pass 2: Generate comprehensive tags using Pass 1 vehicle context and golden master data
        
        Args:
            initial_data: Results from Pass 1 with vehicle information
            product_info: Original product information
            
        Returns:
            Dict: Complete product data with refined tags and vehicle information
        """
        # Get vehicle context from Pass 1
        all_vehicles = initial_data.get('all_vehicles', [])
        primary_vehicle = {
            'year_min': initial_data['year_min'],
            'year_max': initial_data['year_max'],
            'make': initial_data['make'],
            'model': initial_data['model']
        }
        
        # Get golden master context for tag generation
        try:
            golden_clean = self.golden_df.dropna(subset=['year', 'make', 'model'])
            
            # Filter golden master for relevant vehicle compatibility tags
            if primary_vehicle['make'].lower() not in ['universal', 'unknown']:
                # Filter by make and year range for specific vehicles
                filtered_golden = golden_clean[
                    (golden_clean["year"].astype(int) >= primary_vehicle['year_min']) & 
                    (golden_clean["year"].astype(int) <= primary_vehicle['year_max']) &
                    (golden_clean["make"].str.lower() == primary_vehicle['make'].lower())
                ]
            else:
                # For universal parts, get broader context
                filtered_golden = golden_clean.head(200)  # Sample for context
            
        except Exception as e:
            self.logger.error(f"Golden master filtering failed: {str(e)}")
            filtered_golden = self.golden_df.head(100)
        
        # Get sample vehicle compatibility tags from golden master
        sample_vehicle_tags = []
        if not filtered_golden.empty:
            for _, row in filtered_golden.head(10).iterrows():
                try:
                    year = int(row['year'])
                    make = str(row['make'])
                    model = str(row['model'])
                    sample_vehicle_tags.append(f"{year} {make} {model}")
                except:
                    continue
        
        # Format Pass 1 vehicle context for AI
        vehicle_context = f"{primary_vehicle['year_min']}-{primary_vehicle['year_max']} {primary_vehicle['make']} {primary_vehicle['model']}"
        
        prompt = f"""
Generate comprehensive product tags using Pass 1 vehicle context and golden master data.

Product Information:
{product_info}

PASS 1 VEHICLE CONTEXT:
Primary Vehicle: {vehicle_context}

GOLDEN MASTER VEHICLE COMPATIBILITY EXAMPLES:
{', '.join(sample_vehicle_tags[:20])}

INSTRUCTIONS FOR TAG GENERATION:
1. Generate COMPREHENSIVE vehicle compatibility tags using the exact vehicle information from Pass 1
2. Include specific year/make/model combinations that match the golden master format
3. Generate category tags (e.g., "Weatherstrip", "Door Parts", "Seals")
4. Generate brand/manufacturer tags
5. Generate fitment tags (e.g., "Front Door", "Rear Window", "Universal Fit")
6. Generate SEO tags for searchability
7. Generate material/type tags if mentioned (e.g., "Rubber", "Steel", "Chrome")
8. DO NOT generate basic product info - focus on TAGS that will help customers find this product

EXAMPLES OF GOOD TAGS:
- Vehicle: "1939 Hupmobile Skylark", "1940 Hupmobile Skylark", "1941 Hupmobile Skylark"
- Category: "Door Weatherstrip", "Window Seals", "Automotive Weatherstripping"
- Fitment: "Front Door", "Vent Window", "4 Door Sedan"
- Brand: "Steele Rubber", "OEM Replacement"
- SEO: "Classic Car Parts", "Restoration Parts", "Vintage Automotive"

Return ONLY a JSON object with this structure:
{{
    "vehicle_compatibility_tags": ["1939 Hupmobile Skylark", "1940 Hupmobile Skylark", "1941 Hupmobile Skylark"],
    "category_tags": ["Door Weatherstrip", "Window Seals", "Automotive Weatherstripping"],
    "fitment_tags": ["Front Door", "Vent Window", "4 Door Sedan"],
    "brand_tags": ["Steele Rubber", "OEM Replacement"],
    "seo_tags": ["Classic Car Parts", "Restoration Parts", "Vintage Automotive"],
    "material_tags": ["Rubber", "Weather Resistant"],
    "title": "Product title",
    "mpn": "Part number",
    "cost": 25.99,
    "price": 39.99,
    "body_html": "<p>Description</p>",
    "collection": "Collection",
    "product_type": "Part type",
    "meta_title": "SEO title",
    "meta_description": "SEO description"
}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Generate comprehensive product tags using Pass 1 vehicle context and golden master data. Focus on creating detailed, searchable tags that help customers find products."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            # Parse the JSON response
            import json
            import re
            response_text = response.choices[0].message.content
            
            # Extract JSON from response
            try:
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    tag_result = json.loads(json_str)
                else:
                    tag_result = json.loads(response_text)
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse tag generation JSON: {str(e)}")
                # Fallback with basic tags
                tag_result = {
                    'vehicle_compatibility_tags': [f"{primary_vehicle['year_min']}-{primary_vehicle['year_max']} {primary_vehicle['make']} {primary_vehicle['model']}"],
                    'category_tags': ['Automotive Parts'],
                    'fitment_tags': ['OEM Replacement'],
                    'brand_tags': ['Steele Rubber'],
                    'seo_tags': ['Classic Car Parts'],
                    'material_tags': ['Rubber'],
                    'title': f"{primary_vehicle['make']} {primary_vehicle['model']} Part",
                    'mpn': 'Unknown',
                    'cost': 25.99,
                    'price': 39.99,
                    'body_html': '<p>Automotive replacement part</p>',
                    'collection': 'Automotive Parts',
                    'product_type': 'Replacement Part',
                    'meta_title': f"{primary_vehicle['make']} {primary_vehicle['model']} Replacement Part",
                    'meta_description': f"High-quality replacement part for {primary_vehicle['year_min']}-{primary_vehicle['year_max']} {primary_vehicle['make']} {primary_vehicle['model']}"
                }
            
            # Combine all tags into a single tags list for Shopify
            all_tags = []
            all_tags.extend(tag_result.get('vehicle_compatibility_tags', []))
            all_tags.extend(tag_result.get('category_tags', []))
            all_tags.extend(tag_result.get('fitment_tags', []))
            all_tags.extend(tag_result.get('brand_tags', []))
            all_tags.extend(tag_result.get('seo_tags', []))
            all_tags.extend(tag_result.get('material_tags', []))
            
            # Remove duplicates and empty tags
            all_tags = list(dict.fromkeys([tag.strip() for tag in all_tags if tag and tag.strip()]))
            
            # Return product data with comprehensive tags
            return {
                'title': tag_result.get('title', f"{primary_vehicle['make']} {primary_vehicle['model']} Part"),
                'year_min': primary_vehicle['year_min'],  # Keep original vehicle info from Pass 1
                'year_max': primary_vehicle['year_max'],
                'make': primary_vehicle['make'],
                'model': primary_vehicle['model'],
                'mpn': tag_result.get('mpn', 'Unknown'),
                'cost': tag_result.get('cost', 25.99),
                'price': tag_result.get('price', 39.99),
                'body_html': tag_result.get('body_html', '<p>Automotive replacement part</p>'),
                'collection': tag_result.get('collection', 'Automotive Parts'),
                'product_type': tag_result.get('product_type', 'Replacement Part'),
                'meta_title': tag_result.get('meta_title', f"{primary_vehicle['make']} {primary_vehicle['model']} Part"),
                'meta_description': tag_result.get('meta_description', f"Replacement part for {primary_vehicle['make']} {primary_vehicle['model']}"),
                'tags': ', '.join(all_tags),  # Combined tags for Shopify
                'vehicle_compatibility_tags': tag_result.get('vehicle_compatibility_tags', []),
                'category_tags': tag_result.get('category_tags', []),
                'fitment_tags': tag_result.get('fitment_tags', []),
                'brand_tags': tag_result.get('brand_tags', []),
                'seo_tags': tag_result.get('seo_tags', []),
                'material_tags': tag_result.get('material_tags', []),
                'all_vehicles': all_vehicles  # Keep original vehicles from Pass 1
            }
            
        except Exception as e:
            self.logger.error(f"Pass 2 tag generation failed: {str(e)}")
            # Return Pass 1 data with basic tags as fallback
            basic_tags = [f"{primary_vehicle['year_min']}-{primary_vehicle['year_max']} {primary_vehicle['make']} {primary_vehicle['model']}", "Automotive Parts", "Steele Rubber"]
            return {
                'title': initial_data.get('title', f"{primary_vehicle['make']} {primary_vehicle['model']} Part"),
                'year_min': primary_vehicle['year_min'],
                'year_max': primary_vehicle['year_max'],
                'make': primary_vehicle['make'],
                'model': primary_vehicle['model'],
                'mpn': 'Unknown',
                'cost': 25.99,
                'price': 39.99,
                'body_html': initial_data.get('body_html', '<p>Automotive replacement part</p>'),
                'collection': 'Automotive Parts',
                'product_type': 'Replacement Part',
                'meta_title': f"{primary_vehicle['make']} {primary_vehicle['model']} Part",
                'meta_description': f"Replacement part for {primary_vehicle['make']} {primary_vehicle['model']}",
                'tags': ', '.join(basic_tags),
                'all_vehicles': all_vehicles
            }


class AIProductExtractor:
    """Handles AI-powered product data extraction from raw product information using two-pass strategy"""
    
    def __init__(self, api_key: str, model: str = "gpt-4.1-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.two_pass_engine = None  # Will be initialized when golden_df is available
    
    def initialize_two_pass_engine(self, golden_df: pd.DataFrame):
        """Initialize the two-pass AI engine with golden master data"""
        self.two_pass_engine = TwoPassAIEngine(self.client, golden_df)
    
    def extract_product_data(self, product_info: str, golden_df: pd.DataFrame = None) -> ProductData:
        """
        Extract structured product data using two-pass AI strategy
        
        Args:
            product_info: Raw product information string
            golden_df: Golden master DataFrame for validation
            
        Returns:
            ProductData: Validated product data
        """
        if golden_df is not None and self.two_pass_engine is None:
            self.initialize_two_pass_engine(golden_df)
        
        if self.two_pass_engine is not None:
            # Use two-pass AI strategy
            try:
                logger.info("Starting two-pass AI extraction")
                
                # Pass 1: Extract initial vehicle information
                initial_data = self.two_pass_engine.extract_initial_vehicle_info(product_info)
                logger.info(f"Pass 1 completed: {initial_data}")
                
                # Pass 2: Refine with golden master context
                refined_data = self.two_pass_engine.refine_with_golden_master(initial_data, product_info)
                logger.info(f"Pass 2 completed: make={refined_data.get('make')}, model={refined_data.get('model')}")
                
                return ProductData(**refined_data)
                
            except Exception as e:
                logger.error(f"Two-pass AI extraction failed, falling back to single-pass: {str(e)}")
                # Fall back to single-pass if two-pass fails
                return self._extract_single_pass(product_info, golden_df)
        else:
            # Fall back to single-pass extraction
            return self._extract_single_pass(product_info, golden_df)
    
    def _extract_single_pass(self, product_info: str, golden_df: pd.DataFrame = None) -> ProductData:
        """
        Extract structured product data using AI
        
        Args:
            product_info: Raw product information string
            golden_df: Golden master DataFrame for validation
            
        Returns:
            ProductData: Validated product data
            
        Raises:
            AIExtractionError: If extraction fails
        """
        try:
            # Get valid options from golden master
            valid_options = self.get_valid_options_from_golden_master(golden_df)
            
            # Create extraction prompt
            prompt = self.create_extraction_prompt(product_info, valid_options)
            
            # Make API call with retry logic
            for attempt in range(self.max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are an expert automotive parts data extractor."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.1,
                        max_tokens=1000
                    )
                    
                    # Parse response
                    response_text = response.choices[0].message.content
                    
                    # Extract JSON from response
                    try:
                        # Try to find JSON in response
                        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(0)
                            extracted_data = json.loads(json_str)
                        else:
                            # Try to parse entire response as JSON
                            extracted_data = json.loads(response_text)
                    except json.JSONDecodeError:
                        if attempt < self.max_retries - 1:
                            time.sleep(self.retry_delay * (2 ** attempt))
                            continue
                        else:
                            raise AIExtractionError(f"Failed to parse JSON from AI response: {response_text}")
                    
                    # Validate and create ProductData
                    try:
                        product_data = ProductData(**extracted_data)
                        logger.info(f"Successfully extracted product data for {product_data.mpn}")
                        return product_data
                    except Exception as e:
                        if attempt < self.max_retries - 1:
                            time.sleep(self.retry_delay * (2 ** attempt))
                            continue
                        else:
                            raise AIExtractionError(f"Failed to validate extracted data: {str(e)}")
                    
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        logger.warning(f"API call failed (attempt {attempt + 1}): {str(e)}")
                        time.sleep(self.retry_delay * (2 ** attempt))
                        continue
                    else:
                        raise AIExtractionError(f"AI extraction failed after {self.max_retries} attempts: {str(e)}")
            
        except Exception as e:
            logger.error(f"Product extraction failed: {str(e)}")
            return self.handle_ai_errors(e)
    
    def create_extraction_prompt(self, product_info: str, valid_options: Dict) -> str:
        """
        Create structured prompt for AI extraction
        
        Args:
            product_info: Raw product information
            valid_options: Valid makes, models, years from golden master
            
        Returns:
            str: Formatted extraction prompt
        """
        prompt = f"""
Extract structured automotive product data from the following information:

{product_info}

Valid options from our database:
- Makes: {', '.join(valid_options['makes'][:20])}...
- Models: {', '.join(valid_options['models'][:20])}...
- Years: {min(valid_options['years'])}-{max(valid_options['years'])}

IMPORTANT INSTRUCTIONS:
1. Extract year range from descriptions like "1965-1970", "34/64" (means 1934-1964), "65-70" (means 1965-1970)
2. PRIORITIZE SPECIFIC VEHICLE COMPATIBILITY over universal - look for ANY mention of makes, models, years in product name/description
3. For truly universal parts only, use year_min: 1900, year_max: 2024, make: "Universal", model: "Universal"
4. Extract specific make and model from product names or descriptions (e.g., "Ford Mustang", "Chevrolet Camaro")
5. Generate SEO-optimized content
6. Use realistic pricing based on automotive parts market
7. Create appropriate product categorization
8. VEHICLE COMPATIBILITY TAGS will be generated from your year_min, year_max, make, and model fields - be as specific as possible

Return ONLY a JSON object with this exact structure:
{{
    "title": "Product title (max 200 chars)",
    "year_min": 1965,
    "year_max": 1970,
    "make": "Ford",
    "model": "Mustang",
    "mpn": "Part number from StockCode",
    "cost": 43.76,
    "price": 75.49,
    "body_html": "<p>SEO-optimized HTML description</p>",
    "collection": "Brand or category collection",
    "product_type": "Specific part type",
    "meta_title": "SEO title (max 60 chars)",
    "meta_description": "SEO description (max 160 chars)"
}}

Ensure all fields are present and correctly formatted.
"""
        return prompt
    
    def get_valid_options_from_golden_master(self, golden_df: pd.DataFrame) -> Dict[str, List]:
        """
        Extract valid options from golden master dataset
        
        Args:
            golden_df: Golden master DataFrame
            
        Returns:
            Dict: Valid makes, models, years
        """
        valid_options = {
            'makes': [],
            'models': [],
            'years': []
        }
        
        if not golden_df.empty:
            if 'make' in golden_df.columns:
                valid_options['makes'] = sorted(golden_df['make'].dropna().unique().tolist())
            
            if 'model' in golden_df.columns:
                valid_options['models'] = sorted(golden_df['model'].dropna().unique().tolist())
            
            if 'year' in golden_df.columns:
                valid_options['years'] = sorted(golden_df['year'].dropna().unique().tolist())
        
        # Fallback defaults if golden master is empty
        if not valid_options['makes']:
            valid_options['makes'] = ['Ford', 'Chevrolet', 'Dodge', 'Plymouth', 'Chrysler']
        
        if not valid_options['models']:
            valid_options['models'] = ['Mustang', 'Camaro', 'Challenger', 'Charger', 'Corvette']
        
        if not valid_options['years']:
            valid_options['years'] = list(range(1900, 2025))
        
        return valid_options
    
    def extract_year_range(self, description: str) -> Tuple[int, int]:
        """
        Extract year range from product description
        
        Args:
            description: Product description
            
        Returns:
            Tuple[int, int]: (year_min, year_max)
        """
        if not description:
            return (1900, 2024)
        
        # Check for universal compatibility indicators first
        universal_keywords = ['universal', 'all', 'any', 'fits all']
        if any(keyword in description.lower() for keyword in universal_keywords):
            return (1900, 2024)
        
        # Pattern matching for various year formats
        patterns = [
            r'(\d{4})-(\d{4})',  # 1965-1970
            r'(\d{4})\s*-\s*(\d{4})',  # 1965 - 1970
            r'(\d{4})\s*to\s*(\d{4})',  # 1965 to 1970
            r'(\d{2})/(\d{2})',  # 34/64
            r'(\d{2})-(\d{2})',  # 65-70
            r'(\d{4})',  # Single year
        ]
        
        for pattern in patterns:
            match = re.search(pattern, description)
            if match:
                groups = match.groups()
                if len(groups) == 2:
                    year1, year2 = groups
                    # Handle 2-digit years - special logic for automotive years
                    if len(year1) == 2:
                        year1 = int(year1)
                        # For automotive context, assume 1900s for most 2-digit years
                        year1 = 1900 + year1
                    if len(year2) == 2:
                        year2 = int(year2)
                        # For automotive context, assume 1900s for most 2-digit years
                        year2 = 1900 + year2
                    
                    year1, year2 = int(year1), int(year2)
                    return (min(year1, year2), max(year1, year2))
                else:
                    # Single year
                    year = int(groups[0])
                    if len(groups[0]) == 2:
                        # For automotive context, assume 1900s for most 2-digit years
                        year = 1900 + year
                    return (year, year)
        
        # Default fallback
        return (1900, 2024)
    
    def extract_make_model(self, description: str) -> Tuple[str, str]:
        """
        Extract make and model from product description
        
        Args:
            description: Product description
            
        Returns:
            Tuple[str, str]: (make, model)
        """
        if not description:
            return ("Unknown", "Unknown")
        
        # Common automotive makes and their models
        make_model_patterns = {
            'Ford': ['Mustang', 'Thunderbird', 'Galaxie', 'Falcon', 'Torino', 'Bronco'],
            'Chevrolet': ['Camaro', 'Corvette', 'Chevelle', 'Nova', 'Impala', 'Bel Air'],
            'Dodge': ['Challenger', 'Charger', 'Dart', 'Coronet', 'Super Bee'],
            'Plymouth': ['Barracuda', 'Duster', 'Fury', 'Satellite', 'Road Runner'],
            'Chrysler': ['300', 'Newport', 'New Yorker', 'Imperial']
        }
        
        description_lower = description.lower()
        
        # Search for make/model combinations
        for make, models in make_model_patterns.items():
            if make.lower() in description_lower:
                for model in models:
                    if model.lower() in description_lower:
                        return (make, model)
                # Found make but not specific model
                return (make, "Unknown")
        
        # Check for universal compatibility
        universal_keywords = ['universal', 'all', 'any', 'fits all']
        if any(keyword in description_lower for keyword in universal_keywords):
            return ("Universal", "All")
        
        # Default fallback
        return ("Unknown", "Unknown")
    
    def generate_seo_content(self, product_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate SEO-optimized content for product
        
        Args:
            product_data: Product data dictionary
            
        Returns:
            Dict: SEO content (meta_title, meta_description, body_html)
        """
        title = product_data.get('title', 'Automotive Part')
        make = product_data.get('make', 'Unknown')
        model = product_data.get('model', 'Unknown')
        year_min = product_data.get('year_min', 1900)
        year_max = product_data.get('year_max', 2024)
        mpn = product_data.get('mpn', 'Unknown')
        
        # Generate year range string
        if year_min == year_max:
            year_str = str(year_min)
        elif year_min == 1900 and year_max == 2024:
            year_str = "Universal"
        else:
            year_str = f"{year_min}-{year_max}"
        
        # Generate meta title (max 60 chars)
        if make != "Unknown" and model != "Unknown":
            meta_title = f"{make} {model} {title} {year_str}"
        else:
            meta_title = f"{title} {year_str}"
        
        # Truncate if too long
        if len(meta_title) > 60:
            meta_title = meta_title[:57] + "..."
        
        # Generate meta description (max 160 chars)
        if make != "Unknown" and model != "Unknown":
            meta_description = f"High-quality {title.lower()} for {year_str} {make} {model}. Part #{mpn}. Fast shipping and great prices."
        else:
            meta_description = f"High-quality {title.lower()} for {year_str} vehicles. Part #{mpn}. Fast shipping and great prices."
        
        # Truncate if too long
        if len(meta_description) > 160:
            meta_description = meta_description[:157] + "..."
        
        # Generate HTML body
        body_html = f"""<p><strong>{title}</strong></p>
<p>Compatible with: {year_str} {make} {model}</p>
<p>Part Number: {mpn}</p>
<p>High-quality automotive part designed for reliable performance and durability.</p>
<ul>
<li>Direct fit replacement</li>
<li>Quality tested for performance</li>
<li>Backed by manufacturer warranty</li>
</ul>"""
        
        return {
            'meta_title': meta_title,
            'meta_description': meta_description,
            'body_html': body_html
        }
    
    def handle_ai_errors(self, error: Exception) -> ProductData:
        """
        Handle AI extraction errors with fallback data
        
        Args:
            error: The exception that occurred
            
        Returns:
            ProductData: Fallback product data
        """
        logger.error(f"AI extraction failed: {str(error)}")
        
        # Create fallback product data
        fallback_data = ProductData(
            title="Error: Processing Failed",
            year_min=1900,
            year_max=2024,
            make="Unknown",
            model="Unknown",
            mpn="ERROR-001",
            cost=0.0,
            price=0.0,
            body_html="<p>Error processing this product. Please review manually.</p>",
            collection="Error Items",
            product_type="Unknown",
            meta_title="Processing Error",
            meta_description="This product requires manual review due to processing error."
        )
        
        return fallback_data


class BatchExtractor:
    """Handles batch processing of AI extractions"""
    
    def __init__(self, extractor: AIProductExtractor, batch_size: int = 50):
        self.extractor = extractor
        self.batch_size = batch_size
        self.results = []
        self.errors = []
    
    def process_batch(self, product_infos: List[str], golden_df: pd.DataFrame) -> List[ProductData]:
        """
        Process a batch of product information
        
        Args:
            product_infos: List of product information strings
            golden_df: Golden master DataFrame
            
        Returns:
            List[ProductData]: Extracted product data
        """
        results = []
        
        for i, product_info in enumerate(product_infos):
            try:
                logger.info(f"Processing item {i+1}/{len(product_infos)}")
                result = self.extractor.extract_product_data(product_info, golden_df)
                results.append(result)
                
                # Add delay to respect rate limits
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Failed to process item {i+1}: {str(e)}")
                fallback_result = self.extractor.handle_ai_errors(e)
                results.append(fallback_result)
                self.errors.append({'index': i, 'error': str(e), 'product_info': product_info})
        
        return results
    
    def get_error_report(self) -> Dict[str, Any]:
        """
        Get error report for batch processing
        
        Returns:
            Dict: Error report
        """
        return {
            'total_errors': len(self.errors),
            'error_rate': len(self.errors) / len(self.results) if self.results else 0,
            'errors': self.errors
        }