"""
File management utilities shared across all data sources.
Provides consistent file I/O operations.
"""

import os
import json
import csv
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path


class FileManager:
    """Manages file operations for data sources."""
    
    def __init__(self, data_source: str):
        """
        Initialize file manager for a specific data source.
        
        Args:
            data_source: Name of the data source (e.g., 'REM', 'Steele')
        """
        self.data_source = data_source
        self.base_path = Path(data_source)
        self.logger = logging.getLogger(f"{data_source}FileManager")
    
    def ensure_directory(self, path: str) -> None:
        """
        Ensure directory exists, create if it doesn't.
        
        Args:
            path: Directory path to ensure exists
        """
        Path(path).mkdir(parents=True, exist_ok=True)
    
    def get_data_path(self, subdirectory: str) -> Path:
        """
        Get path to data subdirectory.
        
        Args:
            subdirectory: Name of subdirectory (raw, processed, results, samples)
            
        Returns:
            Path object to the subdirectory
        """
        return self.base_path / "data" / subdirectory
    
    def save_json(self, data: Dict[str, Any] | List[Dict[str, Any]], filepath: str) -> None:
        """
        Save data as JSON file.
        
        Args:
            data: Data to save
            filepath: Path where to save the file
        """
        self.ensure_directory(os.path.dirname(filepath))
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"JSON data saved to: {filepath}")
    
    def load_json(self, filepath: str) -> Dict[str, Any] | List[Dict[str, Any]]:
        """
        Load data from JSON file.
        
        Args:
            filepath: Path to the JSON file
            
        Returns:
            Loaded data
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.logger.info(f"JSON data loaded from: {filepath}")
        return data
    
    def save_csv(self, data: List[Dict[str, Any]], filepath: str, fieldnames: Optional[List[str]] = None) -> None:
        """
        Save data as CSV file.
        
        Args:
            data: List of dictionaries to save
            filepath: Path where to save the file
            fieldnames: Optional list of field names for CSV headers
        """
        if not data:
            self.logger.warning("No data to save to CSV")
            return
        
        self.ensure_directory(os.path.dirname(filepath))
        
        if fieldnames is None:
            fieldnames = list(data[0].keys())
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        
        self.logger.info(f"CSV data saved to: {filepath}")
    
    def load_csv(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Load data from CSV file.
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            List of dictionaries from CSV
        """
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            data = list(reader)
        
        self.logger.info(f"CSV data loaded from: {filepath} ({len(data)} rows)")
        return data
    
    def list_files(self, directory: str, extension: str = "") -> List[str]:
        """
        List files in a directory with optional extension filter.
        
        Args:
            directory: Directory to list files from
            extension: File extension filter (e.g., '.json', '.csv')
            
        Returns:
            List of file paths
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            self.logger.warning(f"Directory does not exist: {directory}")
            return []
        
        if extension:
            files = list(dir_path.glob(f"*{extension}"))
        else:
            files = [f for f in dir_path.iterdir() if f.is_file()]
        
        file_paths = [str(f) for f in files]
        self.logger.info(f"Found {len(file_paths)} files in {directory}")
        return file_paths
    
    def get_latest_file(self, directory: str, pattern: str = "*") -> Optional[str]:
        """
        Get the most recently modified file in a directory.
        
        Args:
            directory: Directory to search
            pattern: File pattern to match (default: all files)
            
        Returns:
            Path to the latest file, or None if no files found
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            return None
        
        files = list(dir_path.glob(pattern))
        if not files:
            return None
        
        latest_file = max(files, key=lambda f: f.stat().st_mtime)
        self.logger.info(f"Latest file in {directory}: {latest_file}")
        return str(latest_file) 