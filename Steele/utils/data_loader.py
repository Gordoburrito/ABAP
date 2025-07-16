import pandas as pd
import numpy as np
import tiktoken
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DataQualityMetrics:
    """Data quality metrics for loaded data"""
    total_rows: int
    total_columns: int
    completeness: Dict[str, float]
    duplicates: Dict[str, Any]
    data_types: Dict[str, str]


class RawDataLoader:
    """Handles loading and validation of raw vendor data files"""
    
    def __init__(self):
        self.required_columns = {
            'steele': ['StockCode', 'Product Name', 'Description', 'MAP', 'Dealer Price']
        }
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load raw data from various file formats
        
        Args:
            file_path: Path to the data file
            
        Returns:
            pd.DataFrame: Loaded data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        try:
            if file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            elif file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            logger.info(f"Successfully loaded {len(df)} rows from {file_path}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data from {file_path}: {str(e)}")
            raise
    
    def validate_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate the structure of loaded data
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dict containing validation results
        """
        if df.empty:
            return {
                'is_valid': False,
                'missing_columns': [],
                'data_types': {},
                'column_count': 0,
                'error': 'DataFrame is empty'
            }
        
        # Check for required columns (using steele as default)
        required_cols = self.required_columns.get('steele', [])
        missing_columns = [col for col in required_cols if col not in df.columns]
        
        validation_result = {
            'is_valid': len(missing_columns) == 0,
            'missing_columns': missing_columns,
            'data_types': {col: str(df[col].dtype) for col in df.columns},
            'column_count': len(df.columns),
            'available_columns': list(df.columns)
        }
        
        if missing_columns:
            validation_result['error'] = f"Missing required columns: {missing_columns}"
        
        return validation_result
    
    def generate_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive data quality report
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dict containing quality metrics
        """
        if df.empty:
            return {
                'total_rows': 0,
                'total_columns': 0,
                'completeness': {},
                'duplicates': {'duplicate_count': 0, 'duplicate_percentage': 0},
                'data_types': {}
            }
        
        # Calculate completeness for each column
        completeness = {}
        for col in df.columns:
            non_null_count = df[col].notna().sum()
            completeness[col] = (non_null_count / len(df)) * 100
        
        # Calculate duplicates
        duplicate_count = df.duplicated().sum()
        duplicate_percentage = (duplicate_count / len(df)) * 100
        
        # Analyze data types
        data_types = {col: str(df[col].dtype) for col in df.columns}
        
        quality_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'completeness': completeness,
            'duplicates': {
                'duplicate_count': int(duplicate_count),
                'duplicate_percentage': float(duplicate_percentage)
            },
            'data_types': data_types,
            'memory_usage': float(df.memory_usage(deep=True).sum() / 1024 / 1024)  # MB
        }
        
        return quality_report


class AIFriendlyConverter:
    """Converts raw data to AI-friendly format optimized for token usage"""
    
    def __init__(self, model_name: str = "gpt-4.1-mini"):
        self.model_name = model_name
        self.tokenizer = tiktoken.encoding_for_model(model_name)
        
        # Token costs per 1M tokens (as of current pricing)
        self.token_costs = {
            'gpt-4.1-mini': {'input': 0.150, 'output': 0.600},
            'gpt-4o': {'input': 2.50, 'output': 10.00}
        }
    
    def convert_to_ai_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert raw data to AI-friendly format
        
        Args:
            df: Raw DataFrame
            
        Returns:
            pd.DataFrame: AI-optimized DataFrame
        """
        if df.empty:
            return pd.DataFrame()
        
        ai_df = pd.DataFrame()
        
        # Create product info string for AI processing
        ai_df['product_info'] = df.apply(self.create_product_info_string, axis=1)
        
        # Keep essential fields for processing
        if 'StockCode' in df.columns:
            ai_df['stock_code'] = df['StockCode']
        
        if 'MAP' in df.columns:
            ai_df['price'] = pd.to_numeric(df['MAP'], errors='coerce')
        
        if 'Dealer Price' in df.columns:
            ai_df['cost'] = pd.to_numeric(df['Dealer Price'], errors='coerce')
        
        # Add token count for each product
        ai_df['estimated_tokens'] = ai_df['product_info'].apply(self.estimate_tokens)
        
        logger.info(f"Converted {len(df)} products to AI-friendly format")
        return ai_df
    
    def create_product_info_string(self, row: pd.Series) -> str:
        """
        Create optimized product information string for AI processing
        
        Args:
            row: DataFrame row
            
        Returns:
            str: Optimized product information
        """
        info_parts = []
        
        # Add stock code
        if 'StockCode' in row and pd.notna(row['StockCode']):
            info_parts.append(f"SKU: {row['StockCode']}")
        
        # Add product name
        if 'Product Name' in row and pd.notna(row['Product Name']):
            info_parts.append(f"Product: {row['Product Name']}")
        
        # Add description (most important for AI processing)
        if 'Description' in row and pd.notna(row['Description']):
            info_parts.append(f"Description: {row['Description']}")
        
        # Add UOM if available
        if 'StockUom' in row and pd.notna(row['StockUom']):
            info_parts.append(f"Unit: {row['StockUom']}")
        
        return " | ".join(info_parts)
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text
        
        Args:
            text: Text to count tokens for
            
        Returns:
            int: Estimated token count
        """
        if not text or pd.isna(text):
            return 0
        
        try:
            tokens = self.tokenizer.encode(str(text))
            return len(tokens)
        except Exception:
            # Fallback estimation: approximately 4 characters per token
            return len(str(text)) // 4
    
    def estimate_batch_cost(self, df: pd.DataFrame, model: str = None) -> Dict[str, float]:
        """
        Estimate processing cost for batch of products
        
        Args:
            df: AI-friendly DataFrame
            model: Model to use for cost estimation
            
        Returns:
            Dict containing cost estimates
        """
        if df.empty:
            return {
                'total_input_tokens': 0,
                'estimated_output_tokens': 0,
                'total_cost': 0.0,
                'cost_per_item': 0.0,
                'model': model or self.model_name
            }
        
        model = model or self.model_name
        
        # Calculate input tokens
        if 'estimated_tokens' in df.columns:
            total_input_tokens = df['estimated_tokens'].sum()
        else:
            total_input_tokens = df['product_info'].apply(self.estimate_tokens).sum()
        
        # Estimate output tokens (typical AI response is 200-400 tokens per product)
        estimated_output_tokens = len(df) * 300
        
        # Calculate cost based on model pricing
        if model in self.token_costs:
            input_cost = (total_input_tokens / 1_000_000) * self.token_costs[model]['input']
            output_cost = (estimated_output_tokens / 1_000_000) * self.token_costs[model]['output']
            total_cost = input_cost + output_cost
        else:
            # Default to gpt-4.1-mini pricing
            input_cost = (total_input_tokens / 1_000_000) * self.token_costs['gpt-4.1-mini']['input']
            output_cost = (estimated_output_tokens / 1_000_000) * self.token_costs['gpt-4.1-mini']['output']
            total_cost = input_cost + output_cost
        
        cost_per_item = total_cost / len(df) if len(df) > 0 else 0
        
        return {
            'total_input_tokens': int(total_input_tokens),
            'estimated_output_tokens': int(estimated_output_tokens),
            'input_cost': float(input_cost),
            'output_cost': float(output_cost),
            'total_cost': float(total_cost),
            'cost_per_item': float(cost_per_item),
            'model': model
        }


class TokenOptimizer:
    """Optimize data for minimal token usage"""
    
    @staticmethod
    def optimize_product_description(description: str) -> str:
        """
        Optimize product description for token efficiency
        
        Args:
            description: Raw product description
            
        Returns:
            str: Optimized description
        """
        if not description or pd.isna(description):
            return ""
        
        # Remove redundant words and phrases
        optimizations = [
            ("For use with", "For"),
            ("Compatible with", "For"),
            ("Fits the following vehicles:", "Fits:"),
            ("This product is designed for", "For"),
            ("Replacement part for", "For"),
        ]
        
        optimized = str(description)
        for old, new in optimizations:
            optimized = optimized.replace(old, new)
        
        return optimized.strip()
    
    @staticmethod
    def extract_year_range(description: str) -> Optional[str]:
        """
        Extract year range from description for optimization
        
        Args:
            description: Product description
            
        Returns:
            Optional[str]: Extracted year range
        """
        import re
        
        if not description:
            return None
        
        # Common year range patterns
        patterns = [
            r'(\d{4})-(\d{4})',  # 1965-1970
            r'(\d{2})/(\d{2})',  # 65/70
            r'(\d{4})\s*-\s*(\d{4})',  # 1965 - 1970
        ]
        
        for pattern in patterns:
            match = re.search(pattern, description)
            if match:
                return match.group(0)
        
        return None