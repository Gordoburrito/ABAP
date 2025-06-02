"""
Single item processor for Ford data transformations.
Handles processing of individual items rather than batches.

TEMPLATE FILE: Replace Ford with your actual data source name.
"""

import sys
import os
from typing import Dict, Any, Optional

# Add shared utilities to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared_utils'))

from base_processor import BaseProcessor
from config_utils import ConfigManager
from logging_utils import log_processing_start, log_processing_end, log_error_with_context


class FordSingleProcessor(BaseProcessor):
    """Process single items for Ford data transformations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Ford single processor.
        
        Args:
            config: Configuration dictionary for processing parameters
        """
        super().__init__("Ford", config)
        self.config_manager = ConfigManager("Ford")
        
        # Load configuration if not provided
        if config is None:
            self.config = self.config_manager.load_config()
    
    def process_item(self, item_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single Ford item.
        
        Args:
            item_data: Dictionary containing Ford item data to process
            
        Returns:
            Dictionary containing processed Ford item data
        """
        item_id = item_data.get('id', 'unknown')
        
        try:
            log_processing_start(self.logger, self.data_source, "single_item", item_id=item_id)
            
            # Preprocess using base class functionality
            processed_data = self.preprocess_item(item_data)
            
            # Ford-specific processing logic goes here
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
        Apply Ford-specific processing logic.
        
        CUSTOMIZE THIS METHOD for your data source!
        
        Args:
            item_data: Preprocessed item data
            
        Returns:
            Data after applying Ford-specific transformations
        """
        # Example transformation - customize for your data source
        if 'title' in item_data:
            # Ford might want title in uppercase
            item_data['ford_processed_title'] = item_data['title'].strip().upper()
        
        if 'description' in item_data:
            # Ford might want to clean descriptions
            item_data['ford_cleaned_description'] = self._clean_description(item_data['description'])
        
        # Add any other Ford-specific processing here
        item_data['ford_processing_version'] = '1.0'
        
        return item_data
    
    def _clean_description(self, description: str) -> str:
        """
        Clean description text for Ford.
        
        CUSTOMIZE THIS METHOD for your data source needs!
        
        Args:
            description: Raw description text
            
        Returns:
            Cleaned description text
        """
        # Example cleaning logic - customize as needed
        cleaned = description.strip()
        cleaned = ' '.join(cleaned.split())  # Normalize whitespace
        
        # Add any Ford-specific cleaning rules here
        
        return cleaned


def main():
    """Main function for command line usage."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Process single Ford item')
    parser.add_argument('--input', required=True, help='Input file path')
    parser.add_argument('--output', required=True, help='Output file path')
    parser.add_argument('--config', help='Configuration file path')
    
    args = parser.parse_args()
    
    # Initialize processor
    config = None
    if args.config:
        config_manager = ConfigManager("Ford")
        config = config_manager.load_config(args.config)
    
    processor = FordSingleProcessor(config)
    
    # Load input data
    with open(args.input, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    # Process the item
    result = processor.process_item(input_data)
    
    # Save the result
    processor.save_result(result, args.output)


if __name__ == "__main__":
    main() 