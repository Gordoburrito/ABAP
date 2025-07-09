"""
Logging utilities shared across all data sources.
Provides consistent logging configuration and formatting.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration for a component.
    
    Args:
        name: Logger name (e.g., 'REMProcessor', 'SteeleProcessor')
        level: Logging level (default: INFO)
        log_file: Optional path to log file
        format_string: Optional custom format string
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Default format
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        # Ensure log directory exists
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_data_source_logger(data_source: str, component: str = "") -> logging.Logger:
    """
    Get a logger specific to a data source and component.
    
    Args:
        data_source: Name of the data source (e.g., 'REM', 'Steele')
        component: Optional component name (e.g., 'Processor', 'BatchProcessor')
        
    Returns:
        Logger instance with appropriate naming
    """
    logger_name = f"{data_source}{component}" if component else data_source
    log_file = f"logs/{data_source.lower()}/{component.lower() if component else 'general'}.log"
    
    return setup_logging(
        name=logger_name,
        log_file=log_file
    )


def log_processing_start(logger: logging.Logger, data_source: str, operation: str, **kwargs) -> None:
    """
    Log the start of a processing operation with context.
    
    Args:
        logger: Logger instance
        data_source: Data source name
        operation: Operation being performed
        **kwargs: Additional context to log
    """
    context = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
    logger.info(f"START {data_source} {operation} | {context}")


def log_processing_end(logger: logging.Logger, data_source: str, operation: str, **kwargs) -> None:
    """
    Log the end of a processing operation with context.
    
    Args:
        logger: Logger instance
        data_source: Data source name
        operation: Operation being performed
        **kwargs: Additional context to log (e.g., items_processed, duration)
    """
    context = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
    logger.info(f"END {data_source} {operation} | {context}")


def log_error_with_context(logger: logging.Logger, error: Exception, context: dict) -> None:
    """
    Log an error with additional context information.
    
    Args:
        logger: Logger instance
        error: Exception that occurred
        context: Dictionary of context information
    """
    context_str = " | ".join([f"{k}={v}" for k, v in context.items()])
    logger.error(f"ERROR: {str(error)} | Context: {context_str}", exc_info=True) 