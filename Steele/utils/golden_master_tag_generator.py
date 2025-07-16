import pandas as pd
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class GoldenMasterTagGenerator:
    """Generates vehicle-specific tags using golden master car_id field"""
    
    def __init__(self, golden_df: pd.DataFrame):
        # Normalize column names to lowercase for consistency
        self.golden_df = golden_df.copy()
        self.golden_df.columns = [col.lower().replace(' ', '_') for col in self.golden_df.columns]
        
        # Drop rows with missing critical data
        critical_cols = ['year', 'make', 'model', 'car_id']
        available_cols = [col for col in critical_cols if col in self.golden_df.columns]
        self.golden_df = self.golden_df.dropna(subset=available_cols)
        
        self.logger = logging.getLogger(__name__)
        
        # Clean year column to ensure it's integer
        if 'year' in self.golden_df.columns:
            self.golden_df['year'] = pd.to_numeric(self.golden_df['year'], errors='coerce')
            self.golden_df = self.golden_df.dropna(subset=['year'])
            self.golden_df['year'] = self.golden_df['year'].astype(int)
            
        logger.info(f"Initialized GoldenMasterTagGenerator with {len(self.golden_df)} records")
        
    def generate_vehicle_tags_from_car_ids(self, year_min: int, year_max: int, make: str, models: List[str]) -> List[str]:
        """
        Generate YEAR_MAKE_MODEL format tags from golden master car_id field
        
        Args:
            year_min: Minimum year
            year_max: Maximum year
            make: Vehicle make
            models: List of vehicle models
            
        Returns:
            List[str]: Vehicle compatibility tags in YEAR_MAKE_MODEL format
        """
        try:
            car_ids = []
            
            # Handle universal/unknown cases
            if make.lower() in ['universal', 'unknown', 'all']:
                self.logger.info("Universal make detected, returning empty tags")
                return []
            
            if not models or any(model.lower() in ['universal', 'unknown', 'all'] for model in models):
                self.logger.info("Universal models detected, returning empty tags")
                return []
            
            # Extract car_ids for each year in range
            for year in range(int(year_min), int(year_max) + 1):
                if year > 2024:  # Don't go beyond current year
                    break
                    
                # Filter golden master by year and make
                year_mask = self.golden_df['year'] == year
                make_mask = self.golden_df['make'] == make
                model_mask = self.golden_df['model'].isin(models)
                
                # Debug: Show filtering results
                year_matches = self.golden_df[year_mask]
                make_matches = self.golden_df[year_mask & make_mask]
                model_matches = self.golden_df[year_mask & make_mask & model_mask]
                
                self.logger.info(f"🔍 Golden Master Search - Year {year}:")
                self.logger.info(f"   - Records with year {year}: {len(year_matches)}")
                self.logger.info(f"   - Records with year {year} + make '{make}': {len(make_matches)}")
                self.logger.info(f"   - Records with year {year} + make '{make}' + models {models}: {len(model_matches)}")
                
                # Get matching car_ids (already in YEAR_MAKE_MODEL format)
                filtered_df = self.golden_df[year_mask & make_mask & model_mask]
                year_car_ids = filtered_df['car_id'].unique().tolist()
                
                # Filter out any invalid car_ids
                valid_car_ids = [car_id for car_id in year_car_ids if car_id and isinstance(car_id, str)]
                car_ids.extend(valid_car_ids)
                
                if valid_car_ids:
                    self.logger.info(f"✅ Year {year}: Found {len(valid_car_ids)} car_ids: {valid_car_ids}")
                else:
                    self.logger.info(f"⚠️ Year {year}: No car_ids found")
                    
                    # Show available makes for this year for debugging
                    if len(year_matches) > 0:
                        available_makes = year_matches['make'].unique()[:10]  # Show first 10
                        self.logger.info(f"   Available makes for year {year}: {list(available_makes)}")
                    
                    # Show available models for this year+make for debugging
                    if len(make_matches) > 0:
                        available_models = make_matches['model'].unique()[:10]  # Show first 10
                        self.logger.info(f"   Available models for year {year} + make '{make}': {list(available_models)}")
            
            # Remove duplicates while preserving order
            unique_car_ids = list(dict.fromkeys(car_ids))
            
            self.logger.info(f"Generated {len(unique_car_ids)} vehicle tags for {make} {models}")
            return unique_car_ids
            
        except Exception as e:
            self.logger.error(f"Failed to generate vehicle tags: {str(e)}")
            return []
    
    def extract_car_ids_from_golden_master(self, year: int, make: str, models: List[str]) -> List[str]:
        """
        Extract car_ids for a specific year/make/model combination
        
        Args:
            year: Vehicle year
            make: Vehicle make
            models: List of vehicle models
            
        Returns:
            List[str]: Car IDs for the specific combination
        """
        try:
            # Filter golden master
            year_mask = self.golden_df['year'] == year
            make_mask = self.golden_df['make'] == make
            model_mask = self.golden_df['model'].isin(models)
            
            filtered_df = self.golden_df[year_mask & make_mask & model_mask]
            car_ids = filtered_df['car_id'].unique().tolist()
            
            # Filter out invalid entries
            valid_car_ids = [car_id for car_id in car_ids if car_id and isinstance(car_id, str)]
            
            return valid_car_ids
            
        except Exception as e:
            self.logger.error(f"Failed to extract car_ids: {str(e)}")
            return []
    
    def validate_tags_against_golden_master(self, tags: List[str]) -> bool:
        """
        Validate that all tags exist in golden master car_id field
        
        Args:
            tags: List of YEAR_MAKE_MODEL tags to validate
            
        Returns:
            bool: True if all tags are valid
        """
        try:
            if not tags:
                return True  # Empty list is valid
            
            all_car_ids = set(self.golden_df['car_id'].unique())
            
            for tag in tags:
                if tag not in all_car_ids:
                    self.logger.warning(f"Tag '{tag}' not found in golden master")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to validate tags: {str(e)}")
            return False
    
    def get_car_id_format_tags(self, filtered_df: pd.DataFrame) -> List[str]:
        """
        Get car_id format tags from a filtered DataFrame
        
        Args:
            filtered_df: Pre-filtered golden master DataFrame
            
        Returns:
            List[str]: Car ID tags in YEAR_MAKE_MODEL format
        """
        try:
            car_ids = filtered_df['car_id'].unique().tolist()
            valid_car_ids = [car_id for car_id in car_ids if car_id and isinstance(car_id, str)]
            
            return valid_car_ids
            
        except Exception as e:
            self.logger.error(f"Failed to get car_id format tags: {str(e)}")
            return []
    
    def get_compatible_models_for_refinement(self, year_min: int, year_max: int, make: str, model_spec: str) -> List[str]:
        """
        Get compatible models for AI refinement based on body type specifications
        
        Args:
            year_min: Minimum year
            year_max: Maximum year
            make: Vehicle make
            model_spec: Model specification (e.g., "ALL (4-Door)", "Mustang", etc.)
            
        Returns:
            List[str]: Compatible models for the specification
        """
        try:
            # Filter golden master by year range and make
            year_mask = (self.golden_df['year'] >= year_min) & (self.golden_df['year'] <= year_max)
            make_mask = self.golden_df['make'] == make
            
            filtered_df = self.golden_df[year_mask & make_mask]
            
            if filtered_df.empty:
                return []
            
            # Get all available models for the year/make combination
            available_models = filtered_df['model'].unique().tolist()
            
            # For now, return all available models
            # TODO: Implement body type filtering based on model_spec
            return available_models
            
        except Exception as e:
            self.logger.error(f"Failed to get compatible models: {str(e)}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the golden master dataset
        
        Returns:
            Dict: Statistics about the dataset
        """
        try:
            return {
                'total_records': len(self.golden_df),
                'unique_car_ids': len(self.golden_df['car_id'].unique()),
                'unique_makes': len(self.golden_df['make'].unique()),
                'unique_models': len(self.golden_df['model'].unique()),
                'year_range': f"{self.golden_df['year'].min()}-{self.golden_df['year'].max()}",
                'sample_car_ids': self.golden_df['car_id'].unique()[:10].tolist()
            }
        except Exception as e:
            self.logger.error(f"Failed to get statistics: {str(e)}")
            return {}