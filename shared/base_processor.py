"""
Base processor classes for all data sources.
Provides common functionality and interface consistency.
"""

import os
import sys
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import json

from .logging_utils import setup_logging
from .file_utils import FileManager


class BaseProcessor(ABC):
    """Abstract base class for single item processors."""
    
    def __init__(self, data_source: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base processor.
        
        Args:
            data_source: Name of the data source (e.g., 'REM', 'Steele')
            config: Configuration dictionary for processing parameters
        """
        self.data_source = data_source
        self.config = config or {}
        self.logger = setup_logging(f"{data_source}Processor")
        self.file_manager = FileManager(data_source)
    
    @abstractmethod
    def process_item(self, item_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single item. Must be implemented by subclasses.
        
        Args:
            item_data: Dictionary containing item data to process
            
        Returns:
            Dictionary containing processed item data
        """
        pass
    
    def preprocess_item(self, item_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Common preprocessing steps for all data sources.
        
        Args:
            item_data: Raw item data
            
        Returns:
            Preprocessed item data
        """
        processed_data = item_data.copy()
        processed_data['data_source'] = self.data_source
        processed_data['processing_timestamp'] = self._get_timestamp()
        return processed_data
    
    def postprocess_item(self, item_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Common postprocessing steps for all data sources.
        
        Args:
            item_data: Processed item data
            
        Returns:
            Final processed item data
        """
        item_data['processing_completed'] = True
        return item_data
    
    def save_result(self, result: Dict[str, Any], output_path: str) -> None:
        """
        Save processing result to file.
        
        Args:
            result: Processed data to save
            output_path: Path where to save the result
        """
        self.file_manager.save_json(result, output_path)
        self.logger.info(f"{self.data_source} result saved to: {output_path}")
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        from datetime import datetime
        return datetime.now().isoformat()


class BaseBatchProcessor(ABC):
    """Abstract base class for batch processors."""
    
    def __init__(self, data_source: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base batch processor.
        
        Args:
            data_source: Name of the data source (e.g., 'REM', 'Steele')
            config: Configuration dictionary for processing parameters
        """
        self.data_source = data_source
        self.config = config or {}
        self.logger = setup_logging(f"{data_source}BatchProcessor")
        self.file_manager = FileManager(data_source)
    
    @abstractmethod
    def process_batch(self, batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of items. Must be implemented by subclasses.
        
        Args:
            batch_data: List of dictionaries containing batch data
            
        Returns:
            List of dictionaries containing processed batch data
        """
        pass
    
    def load_batch(self, batch_path: str) -> List[Dict[str, Any]]:
        """
        Load batch data from file.
        
        Args:
            batch_path: Path to the batch file
            
        Returns:
            List of batch items
        """
        return self.file_manager.load_json(batch_path)
    
    def save_batch_results(self, results: List[Dict[str, Any]], output_path: str) -> None:
        """
        Save batch processing results.
        
        Args:
            results: List of processed items
            output_path: Path where to save the results
        """
        batch_summary = {
            'data_source': self.data_source,
            'total_items': len(results),
            'processing_timestamp': self._get_timestamp(),
            'results': results
        }
        
        self.file_manager.save_json(batch_summary, output_path)
        self.logger.info(f"{self.data_source} batch results saved to: {output_path}")
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        from datetime import datetime
        return datetime.now().isoformat() 