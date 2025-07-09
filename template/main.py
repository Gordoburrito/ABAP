"""
Main entry point for {{DATA_SOURCE}} data processing.
Orchestrates the processing pipeline.

TEMPLATE FILE: Replace {{DATA_SOURCE}} with your actual data source name.
"""

import sys
import os
import argparse
from typing import Dict, Any, Optional

# Add shared utilities to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared_utils'))

from config_utils import ConfigManager
from logging_utils import setup_logging, log_processing_start, log_processing_end

# Import data source specific processors
from single_processor import {{DATA_SOURCE}}SingleProcessor
from batch_processor import {{DATA_SOURCE}}BatchProcessor


class {{DATA_SOURCE}}Pipeline:
    """Main pipeline orchestrator for {{DATA_SOURCE}} data processing."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the {{DATA_SOURCE}} processing pipeline.
        
        Args:
            config_file: Optional path to configuration file
        """
        self.data_source = "{{DATA_SOURCE}}"
        self.config_manager = ConfigManager(self.data_source)
        self.config = self.config_manager.load_config(config_file)
        
        # Setup logging
        self.logger = setup_logging(
            f"{self.data_source}Pipeline",
            log_file=f"logs/{self.data_source.lower()}/pipeline.log"
        )
        
        # Initialize processors
        self.single_processor = {{DATA_SOURCE}}SingleProcessor(self.config)
        self.batch_processor = {{DATA_SOURCE}}BatchProcessor(self.config)
    
    def run_single_processing(self, input_path: str, output_path: str) -> None:
        """
        Run single item processing.
        
        Args:
            input_path: Path to input data file
            output_path: Path to save processed output
        """
        log_processing_start(
            self.logger,
            self.data_source,
            "single_processing_pipeline",
            input_path=input_path
        )
        
        try:
            # Load input data
            import json
            with open(input_path, 'r', encoding='utf-8') as f:
                input_data = json.load(f)
            
            # Process the item
            result = self.single_processor.process_item(input_data)
            
            # Save result
            self.single_processor.save_result(result, output_path)
            
            log_processing_end(
                self.logger,
                self.data_source,
                "single_processing_pipeline",
                output_path=output_path,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Single processing pipeline failed: {e}", exc_info=True)
            raise
    
    def run_batch_processing(self, input_path: str, output_path: str) -> None:
        """
        Run batch processing.
        
        Args:
            input_path: Path to batch input data file
            output_path: Path to save batch results
        """
        log_processing_start(
            self.logger,
            self.data_source,
            "batch_processing_pipeline",
            input_path=input_path
        )
        
        try:
            # Load batch data
            batch_data = self.batch_processor.load_batch(input_path)
            
            # Process the batch
            results = self.batch_processor.process_batch(batch_data)
            
            # Save results
            self.batch_processor.save_batch_results(results, output_path)
            
            log_processing_end(
                self.logger,
                self.data_source,
                "batch_processing_pipeline",
                output_path=output_path,
                items_processed=len(results),
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Batch processing pipeline failed: {e}", exc_info=True)
            raise
    
    def run_full_pipeline(self, input_directory: str, output_directory: str) -> None:
        """
        Run the full processing pipeline for all data in a directory.
        
        CUSTOMIZE THIS METHOD for your {{DATA_SOURCE}} needs!
        
        Args:
            input_directory: Directory containing input data files
            output_directory: Directory to save processed results
        """
        log_processing_start(
            self.logger,
            self.data_source,
            "full_pipeline",
            input_directory=input_directory
        )
        
        try:
            from pathlib import Path
            
            input_path = Path(input_directory)
            output_path = Path(output_directory)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Process all JSON files in the input directory
            json_files = list(input_path.glob("*.json"))
            self.logger.info(f"Found {len(json_files)} JSON files to process")
            
            processed_count = 0
            for json_file in json_files:
                try:
                    output_file = output_path / f"processed_{json_file.name}"
                    
                    # Determine if this should be single or batch processing
                    # Customize this logic for your {{DATA_SOURCE}}!
                    if self._is_batch_file(json_file):
                        self.run_batch_processing(str(json_file), str(output_file))
                    else:
                        self.run_single_processing(str(json_file), str(output_file))
                    
                    processed_count += 1
                    
                except Exception as e:
                    self.logger.error(f"Failed to process {json_file}: {e}")
            
            log_processing_end(
                self.logger,
                self.data_source,
                "full_pipeline",
                output_directory=output_directory,
                files_processed=processed_count,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Full pipeline failed: {e}", exc_info=True)
            raise
    
    def _is_batch_file(self, file_path) -> bool:
        """
        Determine if a file should be processed as a batch.
        
        CUSTOMIZE THIS METHOD for your {{DATA_SOURCE}} file naming conventions!
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file should be processed as batch, False for single processing
        """
        # Example logic - customize for your naming conventions
        filename = file_path.name.lower()
        
        # Files with 'batch' in the name are batch files
        if 'batch' in filename:
            return True
        
        # Files with 'bulk' in the name are batch files
        if 'bulk' in filename:
            return True
        
        # Check file size - large files might be batches
        file_size = file_path.stat().st_size
        if file_size > 1024 * 1024:  # 1MB threshold
            return True
        
        # Default to single processing
        return False


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description='{{DATA_SOURCE}} Data Processing Pipeline')
    parser.add_argument('--config', help='Configuration file path')
    
    subparsers = parser.add_subparsers(dest='command', help='Processing commands')
    
    # Single processing command
    single_parser = subparsers.add_parser('single', help='Process single item')
    single_parser.add_argument('--input', required=True, help='Input file path')
    single_parser.add_argument('--output', required=True, help='Output file path')
    
    # Batch processing command
    batch_parser = subparsers.add_parser('batch', help='Process batch')
    batch_parser.add_argument('--input', required=True, help='Input batch file path')
    batch_parser.add_argument('--output', required=True, help='Output results file path')
    
    # Full pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', help='Run full pipeline')
    pipeline_parser.add_argument('--input-dir', required=True, help='Input directory path')
    pipeline_parser.add_argument('--output-dir', required=True, help='Output directory path')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize pipeline
    pipeline = {{DATA_SOURCE}}Pipeline(args.config)
    
    # Execute requested command
    try:
        if args.command == 'single':
            pipeline.run_single_processing(args.input, args.output)
        elif args.command == 'batch':
            pipeline.run_batch_processing(args.input, args.output)
        elif args.command == 'pipeline':
            pipeline.run_full_pipeline(args.input_dir, args.output_dir)
            
        print(f"{{DATA_SOURCE}} {args.command} processing completed successfully!")
        
    except Exception as e:
        print(f"{{DATA_SOURCE}} {args.command} processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 