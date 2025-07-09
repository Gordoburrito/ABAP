"""
Single item processor for Steele data transformations.
Handles processing of individual items rather than batches.
"""

import os
import sys
import logging
from typing import Dict, Any, Optional

# Add the utils directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

class SteeleSingleProcessor:
    """Process single items for Steele data transformations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Steele single processor.
        
        Args:
            config: Configuration dictionary for processing parameters
        """
        self.config = config or {}
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def process_item(self, item_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single Steele item.
        
        Args:
            item_data: Dictionary containing Steele item data to process
            
        Returns:
            Dictionary containing processed Steele item data
        """
        self.logger.info(f"Processing Steele single item: {item_data.get('id', 'unknown')}")
        
        # Add your Steele-specific single item processing logic here
        processed_data = item_data.copy()
        
        # Example Steele transformation (customize as needed)
        if 'title' in processed_data:
            processed_data['steele_processed_title'] = processed_data['title'].strip().title()
            processed_data['data_source'] = 'Steele'
        
        self.logger.info("Steele single item processing completed")
        return processed_data
    
    def save_result(self, result: Dict[str, Any], output_path: str) -> None:
        """
        Save Steele processing result to file.
        
        Args:
            result: Processed Steele data to save
            output_path: Path where to save the result
        """
        import json
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Steele result saved to: {output_path}")


def main():
    """Main function for command line usage."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Process single Steele item')
    parser.add_argument('--input', required=True, help='Input file path')
    parser.add_argument('--output', required=True, help='Output file path')
    parser.add_argument('--config', help='Configuration file path')
    
    args = parser.parse_args()
    
    # Load configuration if provided
    config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Load input data
    with open(args.input, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    # Process the item
    processor = SteeleSingleProcessor(config)
    result = processor.process_item(input_data)
    
    # Save the result
    processor.save_result(result, args.output)


if __name__ == "__main__":
    main() 