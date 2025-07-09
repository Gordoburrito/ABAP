"""
Configuration management utilities shared across all data sources.
Provides consistent configuration loading and validation.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging


class ConfigManager:
    """Manages configuration for data sources."""
    
    def __init__(self, data_source: str):
        """
        Initialize configuration manager for a data source.
        
        Args:
            data_source: Name of the data source (e.g., 'REM', 'Steele')
        """
        self.data_source = data_source
        self.logger = logging.getLogger(f"{data_source}Config")
        
        # Configuration search paths (in order of preference)
        self.config_paths = [
            f"{data_source}/config.json",              # Data source specific
            f"config/{data_source.lower()}.json",      # Shared config directory
            "config/default.json",                     # Default config
        ]
    
    def load_config(self, config_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from file with fallback hierarchy.
        
        Args:
            config_file: Optional specific config file path
            
        Returns:
            Configuration dictionary
        """
        if config_file and os.path.exists(config_file):
            return self._load_config_file(config_file)
        
        # Try each config path in order
        for config_path in self.config_paths:
            if os.path.exists(config_path):
                self.logger.info(f"Loading config from: {config_path}")
                return self._load_config_file(config_path)
        
        # Return default config if no files found
        self.logger.warning("No config file found, using defaults")
        return self._get_default_config()
    
    def _load_config_file(self, filepath: str) -> Dict[str, Any]:
        """Load configuration from a specific file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Validate basic structure
            self._validate_config(config)
            return config
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in config file {filepath}: {e}")
            return self._get_default_config()
        except Exception as e:
            self.logger.error(f"Error loading config file {filepath}: {e}")
            return self._get_default_config()
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate configuration structure.
        
        Args:
            config: Configuration dictionary to validate
        """
        required_sections = ['processing', 'logging', 'paths']
        
        for section in required_sections:
            if section not in config:
                self.logger.warning(f"Missing config section: {section}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "data_source": self.data_source,
            "processing": {
                "batch_size": 100,
                "max_retries": 3,
                "timeout_seconds": 300
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "paths": {
                "raw_data": f"{self.data_source}/data/raw",
                "processed_data": f"{self.data_source}/data/processed",
                "results": f"{self.data_source}/data/results",
                "samples": f"{self.data_source}/data/samples"
            },
            "features": {
                "enable_preprocessing": True,
                "enable_validation": True,
                "enable_postprocessing": True
            }
        }
    
    def save_config(self, config: Dict[str, Any], filepath: str) -> None:
        """
        Save configuration to file.
        
        Args:
            config: Configuration dictionary to save
            filepath: Path where to save the config
        """
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Configuration saved to: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving config to {filepath}: {e}")
    
    def get_data_paths(self, config: Dict[str, Any]) -> Dict[str, str]:
        """
        Get data paths from configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Dictionary of data paths
        """
        return config.get('paths', {})
    
    def get_processing_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get processing configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Processing configuration section
        """
        return config.get('processing', {})
    
    def merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two configuration dictionaries.
        
        Args:
            base_config: Base configuration
            override_config: Configuration to override with
            
        Returns:
            Merged configuration
        """
        merged = base_config.copy()
        
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self.merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged 