import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from collections import Counter
import difflib
from utils.exceptions import GoldenMasterValidationError

logger = logging.getLogger(__name__)


class GoldenMasterValidator:
    """Handles validation of vehicle compatibility against golden master dataset"""
    
    def __init__(self, golden_master_path: str):
        self.golden_master_path = Path(golden_master_path)
        self.golden_df = None
        self._validation_cache = {}
        self._options_cache = {}
        
    def load_golden_master(self) -> pd.DataFrame:
        """
        Load and preprocess golden master dataset
        
        Returns:
            pd.DataFrame: Processed golden master data
            
        Raises:
            FileNotFoundError: If golden master file doesn't exist
            GoldenMasterValidationError: If data processing fails
        """
        try:
            if not self.golden_master_path.exists():
                raise FileNotFoundError(f"Golden master file not found: {self.golden_master_path}")
            
            # Load with optimized settings for large files
            df = pd.read_csv(self.golden_master_path, low_memory=False)
            
            if df.empty:
                raise GoldenMasterValidationError("Golden master dataset is empty")
            
            # Standardize column names
            df = self._standardize_columns(df)
            
            # Clean and preprocess data
            df = self._clean_data(df)
            
            # Create indices for fast lookup
            df = self._create_indices(df)
            
            self.golden_df = df
            logger.info(f"Loaded {len(df)} records from golden master")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load golden master: {str(e)}")
            raise GoldenMasterValidationError(f"Failed to load golden master: {str(e)}")
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names for consistent access"""
        column_mapping = {
            'Year': 'year',
            'Make': 'make', 
            'Model': 'model',
            'Car ID': 'car_id',
            'Body': 'body_type',
            'Engine': 'engine',
            'Submodel': 'submodel',
            'Trim': 'trim'
        }
        
        # Apply mapping where columns exist
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})
        
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize data for validation"""
        df = df.copy()  # Avoid SettingWithCopyWarning
        
        # Convert year to numeric
        if 'year' in df.columns:
            df.loc[:, 'year'] = pd.to_numeric(df['year'], errors='coerce')
            df = df.dropna(subset=['year'])
            df.loc[:, 'year'] = df['year'].astype(int)
        
        # Normalize text columns
        text_columns = ['make', 'model', 'body_type', 'engine', 'submodel', 'trim']
        for col in text_columns:
            if col in df.columns:
                df.loc[:, col] = df[col].astype(str).str.strip()
                df.loc[:, col] = df[col].replace('nan', pd.NA)
        
        # Filter out invalid years
        if 'year' in df.columns:
            df = df[(df['year'] >= 1885) & (df['year'] <= 2030)]
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        return df
    
    def _create_indices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create indices for fast lookups"""
        df = df.copy()  # Avoid SettingWithCopyWarning
        
        # Create normalized columns for case-insensitive lookups
        if 'make' in df.columns:
            df.loc[:, 'make_normalized'] = df['make'].str.lower().str.strip()
        if 'model' in df.columns:
            df.loc[:, 'model_normalized'] = df['model'].str.lower().str.strip()
        if 'body_type' in df.columns:
            df.loc[:, 'body_type_normalized'] = df['body_type'].str.lower().str.strip()
        
        return df
    
    def validate_combination(self, year: int, make: str, model: str) -> bool:
        """
        Validate a single year/make/model combination
        
        Args:
            year: Vehicle year
            make: Vehicle make
            model: Vehicle model
            
        Returns:
            bool: True if combination is valid
        """
        if self.golden_df is None:
            self.load_golden_master()
        
        # Create cache key
        cache_key = (year, make.lower().strip(), model.lower().strip())
        
        if cache_key in self._validation_cache:
            return self._validation_cache[cache_key]
        
        # Normalize inputs
        make_normalized = make.lower().strip()
        model_normalized = model.lower().strip()
        
        # Check if combination exists
        mask = (
            (self.golden_df['year'] == year) &
            (self.golden_df['make_normalized'] == make_normalized) &
            (self.golden_df['model_normalized'] == model_normalized)
        )
        
        is_valid = mask.any()
        
        # Cache result
        self._validation_cache[cache_key] = is_valid
        
        return is_valid
    
    def validate_year_range(self, year_min: int, year_max: int, make: str, model: str) -> Union[bool, Dict[str, Any]]:
        """
        Validate a year range for make/model combination
        
        Args:
            year_min: Minimum year
            year_max: Maximum year
            make: Vehicle make
            model: Vehicle model
            
        Returns:
            Union[bool, Dict]: True if all years valid, or detailed info
        """
        if self.golden_df is None:
            self.load_golden_master()
        
        # Normalize inputs
        make_normalized = make.lower().strip()
        model_normalized = model.lower().strip()
        
        # Get all years for this make/model
        mask = (
            (self.golden_df['make_normalized'] == make_normalized) &
            (self.golden_df['model_normalized'] == model_normalized)
        )
        
        available_years = set(self.golden_df[mask]['year'].unique())
        requested_years = set(range(year_min, year_max + 1))
        
        valid_years = available_years.intersection(requested_years)
        invalid_years = requested_years - available_years
        
        if not invalid_years:
            return True
        
        # Return detailed information
        return {
            'is_valid': len(invalid_years) == 0,
            'valid_years': sorted(valid_years),
            'invalid_years': sorted(invalid_years),
            'available_years': sorted(available_years),
            'coverage': len(valid_years) / len(requested_years)
        }
    
    def get_valid_options(self, year_range: Optional[Tuple[int, int]] = None, 
                         make: Optional[str] = None) -> Dict[str, List]:
        """
        Get valid options from golden master
        
        Args:
            year_range: Optional year range filter
            make: Optional make filter
            
        Returns:
            Dict: Valid makes, models, years
        """
        if self.golden_df is None:
            self.load_golden_master()
        
        # Create cache key
        cache_key = (year_range, make.lower().strip() if make else None)
        
        if cache_key in self._options_cache:
            return self._options_cache[cache_key]
        
        # Apply filters
        df = self.golden_df.copy()
        
        if year_range:
            year_min, year_max = year_range
            df = df[(df['year'] >= year_min) & (df['year'] <= year_max)]
        
        if make:
            make_normalized = make.lower().strip()
            df = df[df['make_normalized'] == make_normalized]
        
        # Get unique options
        options = {
            'makes': sorted(df['make'].dropna().unique().tolist()),
            'models': sorted(df['model'].dropna().unique().tolist()),
            'years': sorted(df['year'].dropna().unique().tolist())
        }
        
        # Cache result
        self._options_cache[cache_key] = options
        
        return options
    
    def get_valid_models_for_make_and_year(self, make: str, year: int) -> List[str]:
        """
        Get valid models for specific make and year
        
        Args:
            make: Vehicle make
            year: Vehicle year
            
        Returns:
            List[str]: Valid models
        """
        if self.golden_df is None:
            self.load_golden_master()
        
        make_normalized = make.lower().strip()
        
        mask = (
            (self.golden_df['make_normalized'] == make_normalized) &
            (self.golden_df['year'] == year)
        )
        
        return sorted(self.golden_df[mask]['model'].dropna().unique().tolist())
    
    def get_valid_car_ids(self, year: int, make: str, model: str) -> List[str]:
        """
        Get valid car IDs for year/make/model combination
        
        Args:
            year: Vehicle year
            make: Vehicle make
            model: Vehicle model
            
        Returns:
            List[str]: Valid car IDs
        """
        if self.golden_df is None:
            self.load_golden_master()
        
        make_normalized = make.lower().strip()
        model_normalized = model.lower().strip()
        
        mask = (
            (self.golden_df['year'] == year) &
            (self.golden_df['make_normalized'] == make_normalized) &
            (self.golden_df['model_normalized'] == model_normalized)
        )
        
        if 'car_id' in self.golden_df.columns:
            return self.golden_df[mask]['car_id'].dropna().unique().tolist()
        else:
            return []
    
    def generate_validation_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate validation report for batch of products
        
        Args:
            df: DataFrame with products to validate
            
        Returns:
            Dict: Validation report
        """
        if self.golden_df is None:
            self.load_golden_master()
        
        valid_items = 0
        invalid_items = 0
        invalid_details = []
        
        for index, row in df.iterrows():
            try:
                # Handle different column naming conventions
                year_min = row.get('year_min', row.get('year', None))
                year_max = row.get('year_max', year_min)
                make = row.get('make', '')
                model = row.get('model', '')
                
                if pd.isna(year_min) or pd.isna(make) or pd.isna(model):
                    invalid_items += 1
                    invalid_details.append({
                        'index': index,
                        'reason': 'Missing required fields',
                        'year_min': year_min,
                        'year_max': year_max,
                        'make': make,
                        'model': model
                    })
                    continue
                
                # Validate year range
                if year_min == year_max:
                    is_valid = self.validate_combination(int(year_min), str(make), str(model))
                else:
                    validation_result = self.validate_year_range(int(year_min), int(year_max), str(make), str(model))
                    is_valid = validation_result is True or (isinstance(validation_result, dict) and validation_result.get('coverage', 0) > 0.8)
                
                if is_valid:
                    valid_items += 1
                else:
                    invalid_items += 1
                    invalid_details.append({
                        'index': index,
                        'reason': 'Invalid year/make/model combination',
                        'year_min': year_min,
                        'year_max': year_max,
                        'make': make,
                        'model': model,
                        'suggestions': self.suggest_alternatives(str(make), str(model), int(year_min))
                    })
                    
            except Exception as e:
                invalid_items += 1
                invalid_details.append({
                    'index': index,
                    'reason': f'Validation error: {str(e)}',
                    'year_min': row.get('year_min'),
                    'year_max': row.get('year_max'),
                    'make': row.get('make'),
                    'model': row.get('model')
                })
        
        total_items = len(df)
        
        return {
            'total_items': total_items,
            'valid_items': valid_items,
            'invalid_items': invalid_items,
            'validation_rate': valid_items / total_items if total_items > 0 else 0,
            'invalid_details': invalid_details
        }
    
    def suggest_alternatives(self, make: str, model: str, year: int) -> Dict[str, Any]:
        """
        Suggest alternatives for invalid combinations
        
        Args:
            make: Vehicle make
            model: Vehicle model
            year: Vehicle year
            
        Returns:
            Dict: Suggested alternatives
        """
        if self.golden_df is None:
            self.load_golden_master()
        
        suggestions = {
            'valid_models_for_make': [],
            'similar_models': [],
            'valid_years_for_make': [],
            'similar_makes': []
        }
        
        make_normalized = make.lower().strip()
        model_normalized = model.lower().strip()
        
        # Get valid models for this make
        make_mask = self.golden_df['make_normalized'] == make_normalized
        if make_mask.any():
            suggestions['valid_models_for_make'] = sorted(
                self.golden_df[make_mask]['model'].dropna().unique().tolist()
            )
            
            # Get valid years for this make
            suggestions['valid_years_for_make'] = sorted(
                self.golden_df[make_mask]['year'].dropna().unique().tolist()
            )
        
        # Find similar models using fuzzy matching
        all_models = self.golden_df['model'].dropna().unique().tolist()
        similar_models = difflib.get_close_matches(model, all_models, n=5, cutoff=0.6)
        suggestions['similar_models'] = similar_models
        
        # Find similar makes
        all_makes = self.golden_df['make'].dropna().unique().tolist()
        similar_makes = difflib.get_close_matches(make, all_makes, n=5, cutoff=0.6)
        suggestions['similar_makes'] = similar_makes
        
        return suggestions
    
    def validate_body_type(self, year: int, make: str, model: str, body_type: str) -> bool:
        """
        Validate body type for year/make/model combination
        
        Args:
            year: Vehicle year
            make: Vehicle make
            model: Vehicle model
            body_type: Body type to validate
            
        Returns:
            bool: True if body type is valid
        """
        if self.golden_df is None:
            self.load_golden_master()
        
        if 'body_type' not in self.golden_df.columns:
            return True  # Can't validate without body type data
        
        make_normalized = make.lower().strip()
        model_normalized = model.lower().strip()
        body_type_normalized = body_type.lower().strip()
        
        mask = (
            (self.golden_df['year'] == year) &
            (self.golden_df['make_normalized'] == make_normalized) &
            (self.golden_df['model_normalized'] == model_normalized) &
            (self.golden_df['body_type_normalized'] == body_type_normalized)
        )
        
        return mask.any()
    
    def get_available_body_types(self, year: int, make: str, model: str) -> List[str]:
        """
        Get available body types for year/make/model combination
        
        Args:
            year: Vehicle year
            make: Vehicle make
            model: Vehicle model
            
        Returns:
            List[str]: Available body types
        """
        if self.golden_df is None:
            self.load_golden_master()
        
        if 'body_type' not in self.golden_df.columns:
            return []
        
        make_normalized = make.lower().strip()
        model_normalized = model.lower().strip()
        
        mask = (
            (self.golden_df['year'] == year) &
            (self.golden_df['make_normalized'] == make_normalized) &
            (self.golden_df['model_normalized'] == model_normalized)
        )
        
        return sorted(self.golden_df[mask]['body_type'].dropna().unique().tolist())
    
    def validate_universal_compatibility(self, year_min: int, year_max: int, 
                                       make: str, model: str) -> Dict[str, Any]:
        """
        Validate universal compatibility claims
        
        Args:
            year_min: Minimum year
            year_max: Maximum year
            make: Vehicle make
            model: Vehicle model
            
        Returns:
            Dict: Universal compatibility validation
        """
        if self.golden_df is None:
            self.load_golden_master()
        
        # Check if this is a universal compatibility claim
        is_universal = (
            make.lower() in ['universal', 'all', 'any'] or
            model.lower() in ['universal', 'all', 'any'] or
            (year_min <= 1900 and year_max >= 2024)
        )
        
        if is_universal:
            # Count total applicable combinations
            year_mask = (self.golden_df['year'] >= max(year_min, 1900)) & (self.golden_df['year'] <= min(year_max, 2024))
            applicable_count = len(self.golden_df[year_mask])
            
            return {
                'is_universal': True,
                'applicable_count': applicable_count,
                'year_range': (max(year_min, 1900), min(year_max, 2024))
            }
        else:
            return {
                'is_universal': False,
                'applicable_count': 0,
                'year_range': (year_min, year_max)
            }
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the golden master dataset
        
        Returns:
            Dict: Dataset statistics
        """
        if self.golden_df is None:
            self.load_golden_master()
        
        stats = {
            'total_combinations': len(self.golden_df),
            'unique_makes': len(self.golden_df['make'].dropna().unique()),
            'unique_models': len(self.golden_df['model'].dropna().unique()),
            'year_range': (
                int(self.golden_df['year'].min()),
                int(self.golden_df['year'].max())
            ),
            'most_common_makes': self.golden_df['make'].value_counts().head(10).to_dict(),
            'most_common_models': self.golden_df['model'].value_counts().head(10).to_dict()
        }
        
        return stats
    
    def validate_with_confidence(self, year: int, make: str, model: str) -> Dict[str, Any]:
        """
        Validate with confidence scoring
        
        Args:
            year: Vehicle year
            make: Vehicle make
            model: Vehicle model
            
        Returns:
            Dict: Validation result with confidence score
        """
        if self.golden_df is None:
            self.load_golden_master()
        
        # Exact match check
        is_valid = self.validate_combination(year, make, model)
        
        if is_valid:
            confidence = 1.0
        else:
            # Calculate confidence based on fuzzy matching
            confidence = 0.0
            
            # Check for similar makes
            all_makes = self.golden_df['make'].dropna().unique().tolist()
            similar_makes = difflib.get_close_matches(make, all_makes, n=1, cutoff=0.8)
            if similar_makes:
                confidence += 0.3
            
            # Check for similar models
            all_models = self.golden_df['model'].dropna().unique().tolist()
            similar_models = difflib.get_close_matches(model, all_models, n=1, cutoff=0.8)
            if similar_models:
                confidence += 0.3
            
            # Check for nearby years
            make_normalized = make.lower().strip()
            model_normalized = model.lower().strip()
            
            make_mask = self.golden_df['make_normalized'] == make_normalized
            model_mask = self.golden_df['model_normalized'] == model_normalized
            
            if make_mask.any() and model_mask.any():
                available_years = self.golden_df[make_mask & model_mask]['year'].unique()
                if len(available_years) > 0:
                    min_year_diff = min(abs(year - y) for y in available_years)
                    if min_year_diff <= 2:
                        confidence += 0.2
        
        return {
            'is_valid': is_valid,
            'confidence': confidence,
            'year': year,
            'make': make,
            'model': model
        }