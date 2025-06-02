"""
Batch processor for {{DATA_SOURCE}} data transformations.
Handles processing of multiple items in batches.

TEMPLATE FILE: Replace {{DATA_SOURCE}} with your actual data source name.
"""

import sys
import os
from typing import Dict, Any, Optional, List

# Add shared utilities to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared_utils'))

from base_processor import BaseBatchProcessor
from config_utils import ConfigManager
from logging_utils import log_processing_start, log_processing_end, log_error_with_context


class {{DATA_SOURCE}}BatchProcessor(BaseBatchProcessor):
    """Process batches of items for {{DATA_SOURCE}} data transformations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the {{DATA_SOURCE}} batch processor.
        
        Args:
            config: Configuration dictionary for processing parameters
        """
        super().__init__("{{DATA_SOURCE}}", config)
        self.config_manager = ConfigManager("{{DATA_SOURCE}}")
        
        # Load configuration if not provided
        if config is None:
            self.config = self.config_manager.load_config()
        
        # Get batch processing configuration
        self.batch_config = self.config_manager.get_processing_config(self.config)
        self.batch_size = self.batch_config.get('batch_size', 100)
    
    def process_batch(self, batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of {{DATA_SOURCE}} items.
        
        Args:
            batch_data: List of dictionaries containing {{DATA_SOURCE}} batch data
            
        Returns:
            List of dictionaries containing processed {{DATA_SOURCE}} batch data
        """
        batch_id = self._generate_batch_id()
        
        try:
            log_processing_start(
                self.logger, 
                self.data_source, 
                "batch_processing", 
                batch_id=batch_id,
                batch_size=len(batch_data)
            )
            
            processed_items = []
            
            for i, item in enumerate(batch_data):
                try:
                    # Add batch context to item
                    item_with_context = item.copy()
                    item_with_context['batch_id'] = batch_id
                    item_with_context['batch_position'] = i
                    
                    # Process individual item using {{DATA_SOURCE}}-specific logic
                    processed_item = self._process_single_item_in_batch(item_with_context)
                    processed_items.append(processed_item)
                    
                except Exception as e:
                    self.logger.error(f"Error processing item {i} in batch {batch_id}: {e}")
                    # Add failed item with error info
                    failed_item = item.copy()
                    failed_item['processing_error'] = str(e)
                    failed_item['processing_status'] = 'failed'
                    processed_items.append(failed_item)
            
            log_processing_end(
                self.logger,
                self.data_source,
                "batch_processing",
                batch_id=batch_id,
                items_processed=len(processed_items),
                success=True
            )
            
            return processed_items
            
        except Exception as e:
            log_error_with_context(
                self.logger,
                e,
                {"operation": "batch_processing", "batch_id": batch_id, "data_source": self.data_source}
            )
            raise
    
    def _process_single_item_in_batch(self, item_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single item within a batch context.
        
        Args:
            item_data: Individual item data with batch context
            
        Returns:
            Processed item data
        """
        # Apply {{DATA_SOURCE}}-specific batch processing logic
        processed_item = self._apply_batch_logic(item_data)
        
        # Add batch completion metadata
        processed_item['processing_status'] = 'completed'
        processed_item['data_source'] = self.data_source
        
        return processed_item
    
    def _apply_batch_logic(self, item_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply {{DATA_SOURCE}}-specific batch processing logic.
        
        CUSTOMIZE THIS METHOD for your data source!
        
        Args:
            item_data: Item data with batch context
            
        Returns:
            Data after applying {{DATA_SOURCE}}-specific batch transformations
        """
        processed_data = item_data.copy()
        
        # Example batch-specific transformations - customize for your data source
        if 'title' in processed_data:
            # {{DATA_SOURCE}} batch processing might standardize titles differently
            processed_data['{{DATA_SOURCE_LOWER}}_batch_title'] = self._standardize_title(processed_data['title'])
        
        if 'category' in processed_data:
            # {{DATA_SOURCE}} might have specific category mappings
            processed_data['{{DATA_SOURCE_LOWER}}_normalized_category'] = self._normalize_category(processed_data['category'])
        
        # Add batch-specific metadata
        processed_data['{{DATA_SOURCE_LOWER}}_batch_version'] = '1.0'
        processed_data['processing_timestamp'] = self._get_timestamp()
        
        return processed_data
    
    def _standardize_title(self, title: str) -> str:
        """
        Standardize title for {{DATA_SOURCE}} batch processing.
        
        CUSTOMIZE THIS METHOD for your data source needs!
        
        Args:
            title: Raw title text
            
        Returns:
            Standardized title
        """
        # Example standardization - customize as needed
        standardized = title.strip().title()
        
        # Add any {{DATA_SOURCE}}-specific title rules here
        
        return standardized
    
    def _normalize_category(self, category: str) -> str:
        """
        Normalize category for {{DATA_SOURCE}}.
        
        CUSTOMIZE THIS METHOD for your data source needs!
        
        Args:
            category: Raw category
            
        Returns:
            Normalized category
        """
        # Example normalization - customize as needed
        normalized = category.strip().lower()
        
        # Add any {{DATA_SOURCE}}-specific category mappings here
        category_mappings = {
            # Add your mappings here
            # 'old_category': 'new_category'
        }
        
        return category_mappings.get(normalized, normalized)
    
    def _generate_batch_id(self) -> str:
        """Generate unique batch ID."""
        import uuid
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"{{DATA_SOURCE_LOWER}}_batch_{timestamp}_{unique_id}"
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        from datetime import datetime
        return datetime.now().isoformat()


def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process {{DATA_SOURCE}} batch')
    parser.add_argument('--input', required=True, help='Input batch file path')
    parser.add_argument('--output', required=True, help='Output results file path')
    parser.add_argument('--config', help='Configuration file path')
    
    args = parser.parse_args()
    
    # Initialize processor
    config = None
    if args.config:
        config_manager = ConfigManager("{{DATA_SOURCE}}")
        config = config_manager.load_config(args.config)
    
    processor = {{DATA_SOURCE}}BatchProcessor(config)
    
    # Load batch data
    batch_data = processor.load_batch(args.input)
    
    # Process the batch
    results = processor.process_batch(batch_data)
    
    # Save the results
    processor.save_batch_results(results, args.output)


if __name__ == "__main__":
    main() 