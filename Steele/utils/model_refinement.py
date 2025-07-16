import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any, Optional, Union, Tuple
from openai import OpenAI
import logging
import time
from pathlib import Path
from utils.exceptions import GoldenMasterValidationError

logger = logging.getLogger(__name__)


class BodyTypeProcessor:
    """Handles processing of body type specifications and logic"""
    
    def __init__(self):
        self.body_type_patterns = {
            'door_patterns': [
                r'ALL\s*\((\d+)-Door\)',  # ALL (2-Door)
                r'(\d+)-Door',  # 2-Door
                r'(\d+)\s*&\s*(\d+)-Door',  # 2 & 4-Door
            ],
            'chassis_patterns': [
                r'ALL\s*\(([A-Z]-Body)\)',  # ALL (A-Body)
                r'([A-Z]-Body)',  # A-Body
            ],
            'body_style_patterns': [
                r'(Coupe|Sedan|Convertible|Hardtop|Fastback|Wagon)',
                r'(Coupe|Sedan|Convertible|Hardtop|Fastback|Wagon)\s*models',
            ]
        }
    
    def process_body_type_specification(self, description: str, golden_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Process body type specification from product description
        
        Args:
            description: Product description
            golden_df: Golden master DataFrame
            
        Returns:
            Dict: Body type processing result
        """
        if not description:
            return {
                'body_type_filter': None,
                'applicable_models': [],
                'processing_method': 'none'
            }
        
        # Extract body type information
        body_type_filter = self.extract_body_type_from_description(description)
        
        # Determine applicable models based on body type
        applicable_models = self.get_models_for_body_type(body_type_filter, golden_df)
        
        # Determine processing method
        processing_method = self.determine_processing_method(body_type_filter, description)
        
        return {
            'body_type_filter': body_type_filter,
            'applicable_models': applicable_models,
            'processing_method': processing_method,
            'description': description
        }
    
    def extract_body_type_from_description(self, description: str) -> Union[str, List[str], None]:
        """
        Extract body type specification from description
        
        Args:
            description: Product description
            
        Returns:
            Union[str, List[str], None]: Body type specification
        """
        if not description:
            return None
        
        description = description.strip()
        
        # Check for combined door patterns first (e.g., "2 & 4-Door")
        combined_door_match = re.search(r'(\d+)\s*&\s*(\d+)-Door', description, re.IGNORECASE)
        if combined_door_match:
            return [f"{combined_door_match.group(1)}-Door", f"{combined_door_match.group(2)}-Door"]
        
        # Check for combined body style patterns (e.g., "Convertible and Hardtop")
        combined_style_match = re.search(r'(Coupe|Sedan|Convertible|Hardtop|Fastback|Wagon)\s+and\s+(Coupe|Sedan|Convertible|Hardtop|Fastback|Wagon)', description, re.IGNORECASE)
        if combined_style_match:
            return [combined_style_match.group(1), combined_style_match.group(2)]
        
        # Check for door patterns
        for pattern in self.body_type_patterns['door_patterns']:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                groups = match.groups()
                if len(groups) == 1:
                    return f"{groups[0]}-Door"
                elif len(groups) == 2:
                    return [f"{groups[0]}-Door", f"{groups[1]}-Door"]
        
        # Check for chassis patterns
        for pattern in self.body_type_patterns['chassis_patterns']:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # Check for body style patterns
        for pattern in self.body_type_patterns['body_style_patterns']:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def get_models_for_body_type(self, body_type_filter: Union[str, List[str], None], 
                                golden_df: pd.DataFrame) -> List[str]:
        """
        Get models that match the body type filter
        
        Args:
            body_type_filter: Body type specification
            golden_df: Golden master DataFrame
            
        Returns:
            List[str]: Matching models
        """
        if body_type_filter is None or golden_df.empty:
            return []
        
        if 'body_type' not in golden_df.columns:
            return []
        
        applicable_models = []
        
        if isinstance(body_type_filter, list):
            # Handle multiple body types
            for body_type in body_type_filter:
                models = self._get_models_for_single_body_type(body_type, golden_df)
                applicable_models.extend(models)
        else:
            # Handle single body type
            applicable_models = self._get_models_for_single_body_type(body_type_filter, golden_df)
        
        return list(set(applicable_models))
    
    def _get_models_for_single_body_type(self, body_type: str, golden_df: pd.DataFrame) -> List[str]:
        """Get models for a single body type"""
        if not body_type:
            return []
        
        # Normalize body type for matching
        body_type_normalized = body_type.lower().strip()
        
        # Check if body_type_normalized column exists
        if 'body_type_normalized' in golden_df.columns:
            mask = golden_df['body_type_normalized'].str.contains(body_type_normalized, na=False)
        else:
            mask = golden_df['body_type'].str.lower().str.contains(body_type_normalized, na=False)
        
        return golden_df[mask]['model'].dropna().unique().tolist()
    
    def determine_processing_method(self, body_type_filter: Union[str, List[str], None], 
                                   description: str) -> str:
        """
        Determine the processing method based on body type specification
        
        Args:
            body_type_filter: Body type specification
            description: Product description
            
        Returns:
            str: Processing method
        """
        if body_type_filter is None:
            return 'standard'
        
        if isinstance(body_type_filter, list):
            return 'multi_body_type'
        
        if 'ALL' in description.upper():
            return 'universal_with_filter'
        
        if any(pattern in body_type_filter for pattern in ['A-Body', 'B-Body', 'C-Body']):
            return 'chassis_specific'
        
        if any(pattern in body_type_filter for pattern in ['2-Door', '4-Door']):
            return 'door_specific'
        
        return 'body_style_specific'
    
    def validate_body_type_for_model(self, year: int, make: str, model: str, 
                                   body_type: str, golden_df: pd.DataFrame) -> bool:
        """
        Validate body type for specific model and year
        
        Args:
            year: Vehicle year
            make: Vehicle make
            model: Vehicle model
            body_type: Body type to validate
            golden_df: Golden master DataFrame
            
        Returns:
            bool: True if body type is valid for model
        """
        if golden_df.empty or 'body_type' not in golden_df.columns:
            return True  # Can't validate without body type data
        
        # Normalize inputs
        make_normalized = make.lower().strip()
        model_normalized = model.lower().strip()
        body_type_normalized = body_type.lower().strip()
        
        # Check if combination exists
        mask = (
            (golden_df['year'] == year) &
            (golden_df['make_normalized'] == make_normalized) &
            (golden_df['model_normalized'] == model_normalized)
        )
        
        if 'body_type_normalized' in golden_df.columns:
            mask = mask & (golden_df['body_type_normalized'] == body_type_normalized)
        else:
            mask = mask & (golden_df['body_type'].str.lower().str.strip() == body_type_normalized)
        
        return mask.any()
    
    def get_available_body_types_for_model(self, year: int, make: str, model: str, 
                                         golden_df: pd.DataFrame) -> List[str]:
        """
        Get available body types for specific model and year
        
        Args:
            year: Vehicle year
            make: Vehicle make
            model: Vehicle model
            golden_df: Golden master DataFrame
            
        Returns:
            List[str]: Available body types
        """
        if golden_df.empty or 'body_type' not in golden_df.columns:
            return []
        
        # Normalize inputs
        make_normalized = make.lower().strip()
        model_normalized = model.lower().strip()
        
        # Filter data
        mask = (
            (golden_df['year'] == year) &
            (golden_df['make_normalized'] == make_normalized) &
            (golden_df['model_normalized'] == model_normalized)
        )
        
        return golden_df[mask]['body_type'].dropna().unique().tolist()
    
    def filter_models_by_body_type(self, golden_df: pd.DataFrame, make: str, 
                                  year: int, body_type: str) -> List[str]:
        """
        Filter models by body type specification
        
        Args:
            golden_df: Golden master DataFrame
            make: Vehicle make
            year: Vehicle year
            body_type: Body type filter
            
        Returns:
            List[str]: Filtered models
        """
        if golden_df.empty:
            return []
        
        # Normalize inputs
        make_normalized = make.lower().strip()
        
        # Base filter
        mask = (
            (golden_df['year'] == year) &
            (golden_df['make_normalized'] == make_normalized)
        )
        
        # Add body type filter if available
        if body_type and 'body_type' in golden_df.columns:
            body_type_normalized = body_type.lower().strip()
            if 'body_type_normalized' in golden_df.columns:
                mask = mask & (golden_df['body_type_normalized'] == body_type_normalized)
            else:
                mask = mask & (golden_df['body_type'].str.lower().str.strip() == body_type_normalized)
        
        return golden_df[mask]['model'].dropna().unique().tolist()
    
    def expand_all_specification(self, golden_df: pd.DataFrame, make: str, 
                                year: int, body_type_filter: Optional[str]) -> List[str]:
        """
        Expand 'ALL' specification to specific models
        
        Args:
            golden_df: Golden master DataFrame
            make: Vehicle make
            year: Vehicle year
            body_type_filter: Optional body type filter
            
        Returns:
            List[str]: Expanded models
        """
        if golden_df.empty:
            return []
        
        # Get all models for make and year
        base_models = self.filter_models_by_body_type(golden_df, make, year, None)
        
        # Apply body type filter if specified
        if body_type_filter:
            filtered_models = self.filter_models_by_body_type(golden_df, make, year, body_type_filter)
            return filtered_models
        
        return base_models


class ModelRefinementEngine:
    """Handles AI-powered model refinement with body type logic"""
    
    def __init__(self, client: OpenAI, golden_df: pd.DataFrame):
        self.client = client
        self.golden_df = golden_df
        self.body_type_processor = BodyTypeProcessor()
        self.max_retries = 3
        self.retry_delay = 1.0
        
    def refine_models_with_ai(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Refine models using AI with body type logic
        
        Args:
            df: DataFrame with product data
            
        Returns:
            pd.DataFrame: DataFrame with refined models
        """
        refined_df = df.copy()
        refined_df['refined_models'] = ''
        refined_df['refinement_confidence'] = 0.0
        refined_df['refinement_method'] = 'none'
        
        for index, row in df.iterrows():
            try:
                # Handle universal compatibility
                if self._is_universal_compatibility(row):
                    refined_df.loc[index, 'refined_models'] = self.handle_universal_compatibility(row)
                    refined_df.loc[index, 'refinement_method'] = 'universal'
                    refined_df.loc[index, 'refinement_confidence'] = 1.0
                    continue
                
                # Get refined models from AI
                refined_models = self.get_refined_models_from_ai(
                    row.get('title', ''),
                    row.get('year_min', 1900),
                    row.get('make', ''),
                    row.get('model', '')
                )
                
                # Validate refined models
                if self.validate_model_selection(
                    refined_models.split(', '), 
                    row.get('year_min', 1900), 
                    row.get('make', '')
                ):
                    refined_df.loc[index, 'refined_models'] = refined_models
                    refined_df.loc[index, 'refinement_method'] = 'ai'
                    refined_df.loc[index, 'refinement_confidence'] = 0.8
                else:
                    # Fallback to original model
                    refined_df.loc[index, 'refined_models'] = row.get('model', '')
                    refined_df.loc[index, 'refinement_method'] = 'fallback'
                    refined_df.loc[index, 'refinement_confidence'] = 0.5
                    
            except Exception as e:
                logger.error(f"Model refinement failed for row {index}: {str(e)}")
                # Fallback to original model
                refined_df.loc[index, 'refined_models'] = row.get('model', '')
                refined_df.loc[index, 'refinement_method'] = 'error'
                refined_df.loc[index, 'refinement_confidence'] = 0.0
        
        return refined_df
    
    def get_refined_models_from_ai(self, title: str, year: int, make: str, models_str: str) -> str:
        """
        Get refined models from AI
        
        Args:
            title: Product title
            year: Vehicle year
            make: Vehicle make
            models_str: Current models string
            
        Returns:
            str: Refined models string
        """
        try:
            prompt = self.create_refinement_prompt(title, year, make, models_str)
            
            for attempt in range(self.max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model="gpt-4.1-mini",
                        messages=[
                            {"role": "system", "content": "You are an expert automotive historian and parts specialist."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.1,
                        max_tokens=200
                    )
                    
                    refined_models = response.choices[0].message.content.strip()
                    
                    # Clean up the response
                    refined_models = self._clean_ai_response(refined_models)
                    
                    return refined_models
                    
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        logger.warning(f"AI refinement failed (attempt {attempt + 1}): {str(e)}")
                        time.sleep(self.retry_delay * (2 ** attempt))
                        continue
                    else:
                        raise
                        
        except Exception as e:
            logger.error(f"AI model refinement failed: {str(e)}")
            # Return original models as fallback
            return models_str
    
    def create_refinement_prompt(self, title: str, year: int, make: str, models_str: str) -> str:
        """
        Create AI refinement prompt
        
        Args:
            title: Product title
            year: Vehicle year
            make: Vehicle make
            models_str: Current models string
            
        Returns:
            str: Refinement prompt
        """
        # Get available models from golden master
        available_models = self._get_available_models_for_make_year(make, year)
        
        prompt = f"""
Refine the model list for this automotive part based on historical accuracy and compatibility:

Product: {title}
Year: {year}
Make: {make}
Current Models: {models_str}

Available models for {make} in {year}: {', '.join(available_models[:20])}

INSTRUCTIONS:
1. Analyze the product title for specific model indicators
2. Consider historical availability of models in {year}
3. Remove any models that didn't exist in {year}
4. Add any missing models that are compatible
5. Consider body type specifications if mentioned in the title

Return ONLY a comma-separated list of refined models, nothing else.
Example: "Mustang, Fastback, Coupe"
"""
        return prompt
    
    def _get_available_models_for_make_year(self, make: str, year: int) -> List[str]:
        """Get available models for make and year from golden master"""
        if self.golden_df.empty:
            return []
        
        make_normalized = make.lower().strip()
        
        mask = (
            (self.golden_df['year'] == year) &
            (self.golden_df['make_normalized'] == make_normalized)
        )
        
        return self.golden_df[mask]['model'].dropna().unique().tolist()
    
    def _clean_ai_response(self, response: str) -> str:
        """Clean AI response to extract model list"""
        # Remove common prefixes/suffixes
        response = response.strip()
        
        # Remove quotes
        response = response.replace('"', '').replace("'", '')
        
        # Remove explanatory text
        if ':' in response:
            response = response.split(':')[-1].strip()
        
        # Remove periods and other punctuation at the end
        response = response.rstrip('.')
        
        return response
    
    def expand_models_to_car_ids(self, row: pd.Series) -> List[str]:
        """
        Expand models to car IDs
        
        Args:
            row: DataFrame row with model information
            
        Returns:
            List[str]: Car IDs
        """
        if self.golden_df.empty or 'car_id' not in self.golden_df.columns:
            return []
        
        car_ids = []
        
        # Get refined models
        refined_models = row.get('refined_models', '').split(', ')
        make = row.get('make', '')
        year_min = row.get('year_min', 1900)
        year_max = row.get('year_max', 2024)
        
        # Normalize make
        make_normalized = make.lower().strip()
        
        for model in refined_models:
            if not model.strip():
                continue
                
            model_normalized = model.strip().lower()
            
            # Get car IDs for this model
            for year in range(year_min, year_max + 1):
                mask = (
                    (self.golden_df['year'] == year) &
                    (self.golden_df['make_normalized'] == make_normalized) &
                    (self.golden_df['model_normalized'] == model_normalized)
                )
                
                model_car_ids = self.golden_df[mask]['car_id'].dropna().unique().tolist()
                car_ids.extend(model_car_ids)
        
        return list(set(car_ids))
    
    def validate_model_selection(self, models: List[str], year: int, make: str) -> bool:
        """
        Validate model selection against golden master
        
        Args:
            models: List of models to validate
            year: Vehicle year
            make: Vehicle make
            
        Returns:
            bool: True if models are valid
        """
        if not models or self.golden_df.empty:
            return False
        
        make_normalized = make.lower().strip()
        
        # Check each model
        for model in models:
            model_normalized = model.strip().lower()
            
            mask = (
                (self.golden_df['year'] == year) &
                (self.golden_df['make_normalized'] == make_normalized) &
                (self.golden_df['model_normalized'] == model_normalized)
            )
            
            if not mask.any():
                return False
        
        return True
    
    def handle_universal_compatibility(self, row: pd.Series) -> str:
        """
        Handle universal compatibility products
        
        Args:
            row: DataFrame row
            
        Returns:
            str: Universal compatibility string
        """
        return "Universal"
    
    def _is_universal_compatibility(self, row: pd.Series) -> bool:
        """Check if product has universal compatibility"""
        make = str(row.get('make', '')).lower()
        model = str(row.get('model', '')).lower()
        year_min = row.get('year_min', 1900)
        year_max = row.get('year_max', 2024)
        
        universal_indicators = ['universal', 'all', 'any']
        
        return (
            any(indicator in make for indicator in universal_indicators) or
            any(indicator in model for indicator in universal_indicators) or
            (year_min <= 1900 and year_max >= 2024)
        )
    
    def get_refined_models_with_confidence(self, title: str, year: int, make: str, models_str: str) -> Dict[str, Any]:
        """
        Get refined models with confidence scoring
        
        Args:
            title: Product title
            year: Vehicle year
            make: Vehicle make
            models_str: Current models string
            
        Returns:
            Dict: Refined models with confidence
        """
        try:
            refined_models = self.get_refined_models_from_ai(title, year, make, models_str)
            
            # Calculate confidence based on validation
            models_list = refined_models.split(', ')
            is_valid = self.validate_model_selection(models_list, year, make)
            
            confidence = 0.8 if is_valid else 0.5
            
            return {
                'models': refined_models,
                'confidence': confidence,
                'is_valid': is_valid
            }
            
        except Exception as e:
            logger.error(f"Model refinement with confidence failed: {str(e)}")
            return {
                'models': models_str,
                'confidence': 0.0,
                'is_valid': False
            }
    
    def validate_historical_accuracy(self, year: int, make: str, model: str) -> bool:
        """
        Validate historical accuracy of year/make/model combination
        
        Args:
            year: Vehicle year
            make: Vehicle make
            model: Vehicle model
            
        Returns:
            bool: True if historically accurate
        """
        if self.golden_df.empty:
            return False
        
        make_normalized = make.lower().strip()
        model_normalized = model.lower().strip()
        
        mask = (
            (self.golden_df['year'] == year) &
            (self.golden_df['make_normalized'] == make_normalized) &
            (self.golden_df['model_normalized'] == model_normalized)
        )
        
        return mask.any()
    
    def process_batch_refinement(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process batch refinement with progress tracking
        
        Args:
            df: DataFrame to process
            
        Returns:
            pd.DataFrame: Processed DataFrame
        """
        total_items = len(df)
        logger.info(f"Starting batch refinement for {total_items} items")
        
        refined_df = self.refine_models_with_ai(df)
        
        # Add processing statistics
        refined_df['processing_time'] = pd.Timestamp.now()
        
        # Calculate success rate
        success_rate = (refined_df['refinement_confidence'] > 0.5).mean()
        logger.info(f"Batch refinement completed with {success_rate:.1%} success rate")
        
        return refined_df