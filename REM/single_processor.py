"""
Single item processor for REM data transformations.
Handles processing of individual items rather than batches.
"""

import sys
import os
from typing import Dict, Any, Optional

# Add shared utilities to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared_utils'))

from base_processor import BaseProcessor
from config_utils import ConfigManager
from logging_utils import log_processing_start, log_processing_end, log_error_with_context


class REMSingleProcessor(BaseProcessor):
    """Process single items for REM data transformations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the REM single processor.
        
        Args:
            config: Configuration dictionary for processing parameters
        """
        super().__init__("REM", config)
        self.config_manager = ConfigManager("REM")
        
        # Load configuration if not provided
        if config is None:
            self.config = self.config_manager.load_config()
    
    def process_item(self, item_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single REM item.
        
        Args:
            item_data: Dictionary containing REM item data to process
            
        Returns:
            Dictionary containing processed REM item data
        """
        item_id = item_data.get('id', 'unknown')
        
        try:
            log_processing_start(self.logger, self.data_source, "single_item", item_id=item_id)
            
            # Preprocess using base class functionality
            processed_data = self.preprocess_item(item_data)
            
            # REM-specific processing logic
            processed_data = self._apply_data_source_logic(processed_data)
            
            # Postprocess using base class functionality
            final_data = self.postprocess_item(processed_data)
            
            log_processing_end(
                self.logger, 
                self.data_source, 
                "single_item", 
                item_id=item_id,
                success=True
            )
            
            return final_data
            
        except Exception as e:
            log_error_with_context(
                self.logger, 
                e, 
                {"operation": "single_item", "item_id": item_id, "data_source": self.data_source}
            )
            raise
    
    def _apply_data_source_logic(self, item_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply REM-specific processing logic.
        
        Args:
            item_data: Preprocessed item data
            
        Returns:
            Data after applying REM-specific transformations
        """
        # REM-specific transformation - uppercase titles
        if 'title' in item_data:
            item_data['rem_processed_title'] = item_data['title'].strip().upper()
        
        if 'description' in item_data:
            item_data['rem_cleaned_description'] = self._clean_description(item_data['description'])
        
        # Add REM-specific processing metadata
        item_data['rem_processing_version'] = '1.0'
        
        return item_data
    
    def _clean_description(self, description: str) -> str:
        """
        Clean description text for REM.
        
        Args:
            description: Raw description text
            
        Returns:
            Cleaned description text
        """
        # REM-specific cleaning logic
        cleaned = description.strip()
        cleaned = ' '.join(cleaned.split())  # Normalize whitespace
        
        # REM might want to remove certain patterns
        # Add any REM-specific cleaning rules here
        
        return cleaned


def main():
    """Main function for command line usage."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Process single REM item')
    parser.add_argument('--input', required=True, help='Input file path')
    parser.add_argument('--output', required=True, help='Output file path')
    parser.add_argument('--config', help='Configuration file path')
    
    args = parser.parse_args()
    
    # Initialize processor
    config = None
    if args.config:
        config_manager = ConfigManager("REM")
        config = config_manager.load_config(args.config)
    
    processor = REMSingleProcessor(config)
    
    # Load input data
    with open(args.input, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    # Process the item
    result = processor.process_item(input_data)
    
    # Save the result
    processor.save_result(result, args.output)


if __name__ == "__main__":
    main() 