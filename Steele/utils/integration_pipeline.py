import pandas as pd
import numpy as np
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from utils.data_loader import RawDataLoader, AIFriendlyConverter
from utils.ai_extraction import AIProductExtractor
from utils.golden_master_validation import GoldenMasterValidator
from utils.model_refinement import ModelRefinementEngine
from utils.shopify_format import ShopifyFormatGenerator
from utils.exceptions import DataProcessingError
from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class ProcessingStats:
    """Data class for processing statistics"""
    total_items: int = 0
    processed_items: int = 0
    failed_items: int = 0
    processing_time: float = 0.0
    total_cost: float = 0.0
    average_processing_time: float = 0.0
    success_rate: float = 0.0


class IncompleteDataPipeline:
    """Complete pipeline for processing incomplete fitment data using AI"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.batch_size = config.get('batch_size', 50)
        self.model = config.get('model', 'gpt-4.1-mini')
        self.enable_ai = config.get('enable_ai', True)
        self.max_retries = config.get('max_retries', 3)
        
        # Initialize processing statistics
        self.processing_stats = ProcessingStats()
        
        # Initialize components
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize all pipeline components"""
        try:
            # Data loading components
            self.data_loader = RawDataLoader()
            self.ai_converter = AIFriendlyConverter(self.model)
            
            # AI processing components (if enabled)
            if self.enable_ai:
                api_key = self.config.get('openai_api_key')
                if not api_key:
                    raise DataProcessingError("OpenAI API key required for AI processing")
                self.ai_extractor = AIProductExtractor(api_key, self.model)
                self.openai_client = OpenAI(api_key=api_key)
            else:
                self.ai_extractor = None
                self.openai_client = None
            
            # Validation components
            golden_master_path = self.config.get('golden_master_path')
            if not golden_master_path:
                raise DataProcessingError("Golden master path required")
            self.golden_validator = GoldenMasterValidator(golden_master_path)
            
            # Model refinement (if AI enabled)
            if self.enable_ai and self.openai_client:
                golden_df = self.golden_validator.load_golden_master()
                self.model_refiner = ModelRefinementEngine(self.openai_client, golden_df)
            else:
                self.model_refiner = None
            
            # Output formatting
            column_requirements_path = self.config.get('column_requirements_path')
            if not column_requirements_path:
                raise DataProcessingError("Column requirements path required")
            self.shopify_generator = ShopifyFormatGenerator(column_requirements_path)
            
            logger.info("Pipeline components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline components: {str(e)}")
            raise DataProcessingError(f"Pipeline initialization failed: {str(e)}")
    
    def process_complete_pipeline(self, input_path: str, output_path: str) -> Dict[str, Any]:
        """
        Process complete pipeline from raw input to Shopify format output
        
        Args:
            input_path: Path to input data file
            output_path: Path to output Shopify format file
            
        Returns:
            Dict: Processing results and statistics
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting complete pipeline processing: {input_path} -> {output_path}")
            
            # Phase 1: Load and validate raw data
            logger.info("Phase 1: Loading raw data")
            raw_df = self.data_loader.load_data(input_path)
            
            # Validate data structure
            validation_result = self.data_loader.validate_structure(raw_df)
            if not validation_result['is_valid']:
                logger.warning(f"Data structure validation warnings: {validation_result}")
            
            # Generate quality report
            quality_report = self.data_loader.generate_quality_report(raw_df)
            logger.info(f"Data quality: {quality_report['total_rows']} rows, {quality_report['duplicates']['duplicate_percentage']:.1f}% duplicates")
            
            # Phase 2: Convert to AI-friendly format
            logger.info("Phase 2: Converting to AI-friendly format")
            ai_df = self.ai_converter.convert_to_ai_format(raw_df)
            
            # Estimate processing cost
            cost_estimate = self.ai_converter.estimate_batch_cost(ai_df, self.model)
            logger.info(f"Estimated processing cost: ${cost_estimate['total_cost']:.2f} ({cost_estimate['cost_per_item']:.4f} per item)")
            
            # Phase 3: AI extraction (if enabled)
            if self.enable_ai and self.ai_extractor:
                logger.info("Phase 3: AI product data extraction")
                extracted_df = self._process_ai_extraction(ai_df)
            else:
                logger.info("Phase 3: Skipping AI extraction (disabled)")
                extracted_df = self._process_template_extraction(ai_df)
            
            # Phase 4: Golden master validation
            logger.info("Phase 4: Golden master validation")
            validated_df = self._process_golden_master_validation(extracted_df)
            
            # Phase 5: Model refinement (if AI enabled)
            if self.enable_ai and self.model_refiner:
                logger.info("Phase 5: AI model refinement")
                refined_df = self.model_refiner.refine_models_with_ai(validated_df)
            else:
                logger.info("Phase 5: Skipping model refinement (AI disabled)")
                refined_df = validated_df
            
            # Phase 6: Generate Shopify format
            logger.info("Phase 6: Generating Shopify format")
            shopify_df = self.shopify_generator.generate_shopify_format(refined_df)
            
            # Validate Shopify format
            validation_result = self.shopify_generator.validate_column_compliance(shopify_df)
            if not validation_result['is_compliant']:
                logger.warning(f"Shopify format validation issues: {validation_result['validation_errors']}")
            
            # Save output
            logger.info(f"Saving output to: {output_path}")
            shopify_df.to_csv(output_path, index=False)
            
            # Calculate final statistics
            processing_time = time.time() - start_time
            self.processing_stats = ProcessingStats(
                total_items=len(raw_df),
                processed_items=len(shopify_df),
                failed_items=len(raw_df) - len(shopify_df),
                processing_time=processing_time,
                total_cost=cost_estimate['total_cost'],
                average_processing_time=processing_time / len(raw_df) if len(raw_df) > 0 else 0,
                success_rate=len(shopify_df) / len(raw_df) if len(raw_df) > 0 else 0
            )
            
            logger.info(f"Pipeline completed successfully in {processing_time:.2f}s")
            logger.info(f"Processed {len(shopify_df)}/{len(raw_df)} items ({self.processing_stats.success_rate:.1%} success rate)")
            
            # Return results
            return {
                'total_items': self.processing_stats.total_items,
                'processed_items': self.processing_stats.processed_items,
                'failed_items': self.processing_stats.failed_items,
                'success_rate': self.processing_stats.success_rate,
                'processing_time': self.processing_stats.processing_time,
                'total_cost': self.processing_stats.total_cost,
                'cost_per_item': cost_estimate['cost_per_item'],
                'output_file': output_path,
                'validation_compliant': validation_result['is_compliant']
            }
            
        except Exception as e:
            logger.error(f"Pipeline processing failed: {str(e)}")
            raise DataProcessingError(f"Pipeline processing failed: {str(e)}")
    
    def _process_ai_extraction(self, ai_df: pd.DataFrame) -> pd.DataFrame:
        """Process AI extraction phase"""
        extracted_products = []
        
        for index, row in ai_df.iterrows():
            try:
                # Get golden master for validation context
                golden_df = self.golden_validator.golden_df
                
                # Extract product data using AI
                product_data = self.ai_extractor.extract_product_data(
                    row['product_info'], golden_df
                )
                
                # Convert to DataFrame row
                extracted_row = product_data.dict()
                extracted_row['original_index'] = index
                extracted_products.append(extracted_row)
                
            except Exception as e:
                logger.error(f"AI extraction failed for item {index}: {str(e)}")
                # Add fallback data
                fallback_row = {
                    'title': 'Processing Error',
                    'year_min': 1900,
                    'year_max': 2024,
                    'make': 'Unknown',
                    'model': 'Unknown',
                    'mpn': row.get('stock_code', 'ERROR'),
                    'cost': row.get('cost', 0),
                    'price': row.get('price', 0),
                    'body_html': '<p>Error processing this product</p>',
                    'collection': 'Error Items',
                    'product_type': 'Unknown',
                    'meta_title': 'Processing Error',
                    'meta_description': 'This product requires manual review',
                    'original_index': index
                }
                extracted_products.append(fallback_row)
        
        return pd.DataFrame(extracted_products)
    
    def _process_template_extraction(self, ai_df: pd.DataFrame) -> pd.DataFrame:
        """Process template-based extraction (no AI)"""
        template_products = []
        
        for index, row in ai_df.iterrows():
            # Use template-based extraction
            template_row = {
                'title': 'Automotive Part',
                'year_min': 1900,
                'year_max': 2024,
                'make': 'Universal',
                'model': 'All',
                'mpn': row.get('stock_code', 'UNKNOWN'),
                'cost': row.get('cost', 0),
                'price': row.get('price', 0),
                'body_html': '<p>Universal automotive part</p>',
                'collection': 'Universal Parts',
                'product_type': 'Automotive Part',
                'meta_title': 'Universal Automotive Part',
                'meta_description': 'Universal automotive part for all vehicles',
                'original_index': index
            }
            template_products.append(template_row)
        
        return pd.DataFrame(template_products)
    
    def _process_golden_master_validation(self, extracted_df: pd.DataFrame) -> pd.DataFrame:
        """Process golden master validation phase"""
        if self.golden_validator.golden_df is None:
            self.golden_validator.load_golden_master()
        
        # Generate validation report
        validation_report = self.golden_validator.generate_validation_report(extracted_df)
        
        logger.info(f"Golden master validation: {validation_report['valid_items']}/{validation_report['total_items']} valid ({validation_report['validation_rate']:.1%})")
        
        if validation_report['invalid_items'] > 0:
            logger.warning(f"Found {validation_report['invalid_items']} items with invalid vehicle combinations")
        
        return extracted_df
    
    def estimate_processing_cost(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Estimate processing cost for input data
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dict: Cost estimation
        """
        if not self.enable_ai:
            return {
                'total_input_tokens': 0,
                'estimated_output_tokens': 0,
                'total_cost': 0.0,
                'cost_per_item': 0.0,
                'model': self.model
            }
        
        # Convert to AI-friendly format
        ai_df = self.ai_converter.convert_to_ai_format(df)
        
        # Estimate cost
        return self.ai_converter.estimate_batch_cost(ai_df, self.model)
    
    def generate_processing_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive processing report
        
        Returns:
            Dict: Processing report
        """
        return {
            'summary': {
                'total_items': self.processing_stats.total_items,
                'processed_items': self.processing_stats.processed_items,
                'failed_items': self.processing_stats.failed_items,
                'success_rate': self.processing_stats.success_rate
            },
            'performance_metrics': {
                'total_processing_time': self.processing_stats.processing_time,
                'average_processing_time': self.processing_stats.average_processing_time,
                'items_per_minute': (self.processing_stats.processed_items / (self.processing_stats.processing_time / 60)) if self.processing_stats.processing_time > 0 else 0
            },
            'cost_analysis': {
                'total_cost': self.processing_stats.total_cost,
                'cost_per_item': self.processing_stats.total_cost / self.processing_stats.total_items if self.processing_stats.total_items > 0 else 0,
                'model_used': self.model,
                'ai_enabled': self.enable_ai
            },
            'quality_metrics': {
                'success_rate': self.processing_stats.success_rate,
                'failure_rate': 1 - self.processing_stats.success_rate,
                'batch_size': self.batch_size
            },
            'configuration': {
                'model': self.model,
                'batch_size': self.batch_size,
                'ai_enabled': self.enable_ai,
                'max_retries': self.max_retries
            }
        }


class PipelineValidator:
    """Validates pipeline configuration and requirements"""
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate pipeline configuration
        
        Args:
            config: Pipeline configuration
            
        Returns:
            Dict: Validation results
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Required configuration fields
        required_fields = [
            'golden_master_path',
            'column_requirements_path'
        ]
        
        # Check required fields
        for field in required_fields:
            if field not in config:
                validation_result['errors'].append(f"Missing required field: {field}")
                validation_result['is_valid'] = False
        
        # Check AI configuration
        if config.get('enable_ai', True):
            if 'openai_api_key' not in config:
                validation_result['errors'].append("OpenAI API key required when AI is enabled")
                validation_result['is_valid'] = False
        
        # Check file paths
        for path_field in ['golden_master_path', 'column_requirements_path']:
            if path_field in config:
                path = Path(config[path_field])
                if not path.exists():
                    validation_result['errors'].append(f"File not found: {path}")
                    validation_result['is_valid'] = False
        
        # Check numeric configuration
        numeric_fields = {
            'batch_size': (1, 1000),
            'max_retries': (1, 10)
        }
        
        for field, (min_val, max_val) in numeric_fields.items():
            if field in config:
                value = config[field]
                if not isinstance(value, int) or value < min_val or value > max_val:
                    validation_result['warnings'].append(f"{field} should be between {min_val} and {max_val}")
        
        # Check model name
        valid_models = ['gpt-4.1-mini', 'gpt-4o', 'gpt-3.5-turbo']
        if 'model' in config and config['model'] not in valid_models:
            validation_result['warnings'].append(f"Unknown model: {config['model']}. Valid models: {valid_models}")
        
        return validation_result
    
    @staticmethod
    def validate_input_data(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate input data format
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dict: Validation results
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'data_quality': {}
        }
        
        if df.empty:
            validation_result['errors'].append("Input data is empty")
            validation_result['is_valid'] = False
            return validation_result
        
        # Check required columns
        required_columns = ['StockCode', 'Product Name', 'MAP', 'Dealer Price']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            validation_result['errors'].append(f"Missing required columns: {missing_columns}")
            validation_result['is_valid'] = False
        
        # Data quality checks
        if not df.empty:
            validation_result['data_quality'] = {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'missing_data_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
                'duplicate_rows': df.duplicated().sum()
            }
            
            # Check for high missing data percentage
            if validation_result['data_quality']['missing_data_percentage'] > 20:
                validation_result['warnings'].append(f"High missing data percentage: {validation_result['data_quality']['missing_data_percentage']:.1f}%")
            
            # Check for many duplicates
            duplicate_percentage = (validation_result['data_quality']['duplicate_rows'] / len(df)) * 100
            if duplicate_percentage > 10:
                validation_result['warnings'].append(f"High duplicate percentage: {duplicate_percentage:.1f}%")
        
        return validation_result