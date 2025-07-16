#!/usr/bin/env python3
"""
Multi-vehicle tag generation utilities
"""

from typing import List, Dict, Any
from utils.golden_master_tag_generator import GoldenMasterTagGenerator
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class MultiVehicleTagGenerator:
    """Generate tags for products that fit multiple vehicle types"""
    
    def __init__(self, tag_generator: GoldenMasterTagGenerator):
        self.tag_generator = tag_generator
    
    def generate_all_vehicle_tags(self, extraction_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate tags for ALL vehicles that a product fits
        
        Args:
            extraction_result: Result from AI extraction with 'all_vehicles' field
            
        Returns:
            Dict with combined tags and vehicle breakdown
        """
        all_vehicles = extraction_result.get('all_vehicles', [])
        if not all_vehicles:
            # Fallback to single vehicle
            all_vehicles = [{
                'year_min': extraction_result.get('year_min', 1900),
                'year_max': extraction_result.get('year_max', 2024),
                'make': extraction_result.get('make', 'Universal'),
                'model': extraction_result.get('model', 'Universal')
            }]
        
        all_tags = []
        vehicle_breakdown = []
        
        for idx, vehicle in enumerate(all_vehicles):
            # Skip universal vehicles unless it's the only one
            if vehicle['make'].lower() in ['universal', 'unknown'] and len(all_vehicles) > 1:
                continue
                
            # Generate tags for this vehicle
            vehicle_tags = self._generate_vehicle_tags(vehicle)
            
            vehicle_info = {
                'index': idx,
                'make': vehicle['make'],
                'model': vehicle['model'],
                'year_range': f"{vehicle['year_min']}-{vehicle['year_max']}",
                'tags': vehicle_tags,
                'tag_count': len(vehicle_tags)
            }
            
            vehicle_breakdown.append(vehicle_info)
            all_tags.extend(vehicle_tags)
            
            logger.info(f"Vehicle {idx+1}: {vehicle['make']} {vehicle['model']} ({vehicle['year_min']}-{vehicle['year_max']}) -> {len(vehicle_tags)} tags")
        
        # Remove duplicates while preserving order
        unique_tags = []
        seen = set()
        for tag in all_tags:
            if tag not in seen:
                unique_tags.append(tag)
                seen.add(tag)
        
        return {
            'all_tags': unique_tags,
            'total_tag_count': len(unique_tags),
            'vehicle_count': len(vehicle_breakdown),
            'vehicle_breakdown': vehicle_breakdown,
            'combined_tags_string': ', '.join(unique_tags) if unique_tags else ''
        }
    
    def _generate_vehicle_tags(self, vehicle: Dict[str, Any]) -> List[str]:
        """Generate tags for a single vehicle"""
        try:
            # Parse models intelligently
            model_str = vehicle['model'].replace(' and ', ', ').replace('/', ', ').replace('|', ', ')
            models = [m.strip() for m in model_str.split(',') if m.strip()]
            
            # Try exact model match first
            vehicle_tags = self.tag_generator.generate_vehicle_tags_from_car_ids(
                vehicle['year_min'], 
                vehicle['year_max'], 
                vehicle['make'], 
                models
            )
            
            # If no tags found, try intelligent model matching
            if not vehicle_tags:
                vehicle_tags = self._try_intelligent_model_matching(vehicle, models)
            
            return vehicle_tags if vehicle_tags else []
            
        except Exception as e:
            logger.error(f"Failed to generate tags for {vehicle['make']} {vehicle['model']}: {str(e)}")
            return []
    
    def _try_intelligent_model_matching(self, vehicle: Dict[str, Any], original_models: List[str]) -> List[str]:
        """Try intelligent model matching for cases where exact match fails"""
        try:
            # Get available models for this make/year from golden master
            golden_df = self.tag_generator.golden_df
            
            # Filter for matching make and year range
            filtered_golden = golden_df[
                (golden_df['make'] == vehicle['make']) &
                (pd.to_numeric(golden_df['year'], errors='coerce') >= vehicle['year_min']) &
                (pd.to_numeric(golden_df['year'], errors='coerce') <= vehicle['year_max'])
            ].dropna(subset=['year'])
            
            if filtered_golden.empty:
                return []
            
            available_models = list(filtered_golden['model'].dropna().unique())
            
            # Try intelligent matching for each original model
            matched_models = []
            for original_model in original_models:
                best_match = self._find_best_model_match(original_model, available_models)
                if best_match and best_match not in matched_models:
                    matched_models.append(best_match)
            
            # Generate tags with matched models
            if matched_models:
                logger.info(f"Intelligent matching: {original_models} -> {matched_models}")
                return self.tag_generator.generate_vehicle_tags_from_car_ids(
                    vehicle['year_min'], 
                    vehicle['year_max'], 
                    vehicle['make'], 
                    matched_models
                )
            
            return []
            
        except Exception as e:
            logger.error(f"Intelligent model matching failed: {str(e)}")
            return []
    
    def _find_best_model_match(self, target_model: str, available_models: List[str]) -> str:
        """Find best matching model from available options"""
        target_lower = target_model.lower()
        
        # Exact match
        for model in available_models:
            if model.lower() == target_lower:
                return model
        
        # Common automotive model matching patterns
        # For models like "6-7A Commander" -> "Commander"
        if 'commander' in target_lower:
            for model in available_models:
                if 'commander' in model.lower():
                    return model
        
        # For models like "8-4C State President" -> "State President"  
        if 'state president' in target_lower:
            for model in available_models:
                if 'state president' in model.lower():
                    return model
        
        # For models like "State President" -> "President"
        if 'president' in target_lower:
            for model in available_models:
                if 'president' in model.lower():
                    return model
        
        # Partial word matching (last resort)
        target_words = set(word for word in target_lower.split() if len(word) > 2)
        best_score = 0
        best_match = None
        
        for model in available_models:
            model_words = set(word for word in model.lower().split() if len(word) > 2)
            score = len(target_words & model_words)
            if score > best_score and score > 0:
                best_score = score
                best_match = model
        
        return best_match
    
    def format_vehicle_summary(self, result: Dict[str, Any]) -> str:
        """Create a human-readable summary of vehicle compatibility"""
        breakdown = result.get('vehicle_breakdown', [])
        if not breakdown:
            return "No vehicles found"
        
        summaries = []
        for vehicle in breakdown:
            if vehicle['tag_count'] > 0:
                summaries.append(f"{vehicle['make']} {vehicle['model']} ({vehicle['year_range']}) - {vehicle['tag_count']} tags")
            else:
                summaries.append(f"{vehicle['make']} {vehicle['model']} ({vehicle['year_range']}) - no tags")
        
        return "; ".join(summaries)