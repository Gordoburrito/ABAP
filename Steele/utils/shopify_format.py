import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import logging
from utils.exceptions import ShopifyFormatError

logger = logging.getLogger(__name__)


class SEOContentGenerator:
    """Handles generation of SEO-optimized content for Shopify products"""
    
    def __init__(self):
        self.max_meta_title_length = 60
        self.max_meta_description_length = 160
        
    def generate_meta_title(self, product_data: Dict[str, Any]) -> str:
        """
        Generate SEO-optimized meta title
        
        Args:
            product_data: Product data dictionary
            
        Returns:
            str: Optimized meta title
        """
        title = product_data.get('title', 'Automotive Part')
        make = product_data.get('make', '')
        model = product_data.get('model', '')
        year_min = product_data.get('year_min', 1900)
        year_max = product_data.get('year_max', 2024)
        
        # Generate year range string
        year_str = self.generate_year_range_string(year_min, year_max)
        
        # Build meta title
        if make and model and make.lower() not in ['universal', 'unknown'] and model.lower() not in ['all', 'unknown']:
            meta_title = f"{make} {model} {title} {year_str}"
        elif make and make.lower() not in ['universal', 'unknown']:
            meta_title = f"{make} {title} {year_str}"
        else:
            meta_title = f"{title} {year_str}"
        
        # Optimize length
        return self.optimize_for_seo_length(meta_title, self.max_meta_title_length)
    
    def generate_meta_description(self, product_data: Dict[str, Any]) -> str:
        """
        Generate SEO-optimized meta description
        
        Args:
            product_data: Product data dictionary
            
        Returns:
            str: Optimized meta description
        """
        title = product_data.get('title', 'automotive part')
        make = product_data.get('make', '')
        model = product_data.get('model', '')
        year_min = product_data.get('year_min', 1900)
        year_max = product_data.get('year_max', 2024)
        mpn = product_data.get('mpn', '')
        
        # Generate year range string
        year_str = self.generate_year_range_string(year_min, year_max)
        
        # Build meta description
        if make and model and make.lower() not in ['universal', 'unknown'] and model.lower() not in ['all', 'unknown']:
            meta_description = f"High-quality {title.lower()} for {year_str} {make} {model}."
        elif make and make.lower() not in ['universal', 'unknown']:
            meta_description = f"High-quality {title.lower()} for {year_str} {make} vehicles."
        else:
            meta_description = f"High-quality {title.lower()} for {year_str} vehicles."
        
        # Add part number if available
        if mpn:
            meta_description += f" Part #{mpn}."
        
        # Add standard closing
        meta_description += " Fast shipping and great prices."
        
        # Optimize length
        return self.optimize_for_seo_length(meta_description, self.max_meta_description_length)
    
    def generate_body_html(self, product_data: Dict[str, Any]) -> str:
        """
        Generate HTML body content for product
        
        Args:
            product_data: Product data dictionary
            
        Returns:
            str: HTML body content
        """
        title = product_data.get('title', 'Automotive Part')
        make = product_data.get('make', '')
        model = product_data.get('model', '')
        year_min = product_data.get('year_min', 1900)
        year_max = product_data.get('year_max', 2024)
        mpn = product_data.get('mpn', '')
        
        # Generate year range string
        year_str = self.generate_year_range_string(year_min, year_max)
        
        # Build HTML content
        html_parts = []
        
        # Product title and description
        html_parts.append(f"<p><strong>{title}</strong></p>")
        
        # Compatibility information
        if make and model and make.lower() not in ['universal', 'unknown'] and model.lower() not in ['all', 'unknown']:
            html_parts.append(f"<p>Compatible with: {year_str} {make} {model}</p>")
        elif make and make.lower() not in ['universal', 'unknown']:
            html_parts.append(f"<p>Compatible with: {year_str} {make} vehicles</p>")
        else:
            html_parts.append(f"<p>Universal compatibility for {year_str} vehicles</p>")
        
        # Part number
        if mpn:
            html_parts.append(f"<p>Part Number: {mpn}</p>")
        
        # Product description
        html_parts.append("<p>High-quality automotive part designed for reliable performance and durability.</p>")
        
        # Product features
        features = self.generate_product_features_list(product_data)
        if features:
            html_parts.append("<ul>")
            for feature in features:
                html_parts.append(f"<li>{feature}</li>")
            html_parts.append("</ul>")
        
        return "\n".join(html_parts)
    
    def generate_year_range_string(self, year_min: int, year_max: int) -> str:
        """
        Generate year range string for display
        
        Args:
            year_min: Minimum year
            year_max: Maximum year
            
        Returns:
            str: Year range string
        """
        if year_min == year_max:
            return str(year_min)
        elif year_min <= 1900 and year_max >= 2024:
            return "Universal"
        else:
            return f"{year_min}-{year_max}"
    
    def generate_product_features_list(self, product_data: Dict[str, Any]) -> List[str]:
        """
        Generate list of product features
        
        Args:
            product_data: Product data dictionary
            
        Returns:
            List[str]: Product features
        """
        features = [
            "Direct fit replacement",
            "Quality tested for performance",
            "Backed by manufacturer warranty",
            "Professional installation recommended"
        ]
        
        # Add product-specific features based on type
        product_type = product_data.get('product_type', '').lower()
        
        if 'brake' in product_type:
            features.extend([
                "Superior stopping power",
                "Noise-free operation",
                "Extended service life"
            ])
        elif 'engine' in product_type:
            features.extend([
                "Optimized for performance",
                "Enhanced durability",
                "Precise engineering specifications"
            ])
        elif 'electrical' in product_type:
            features.extend([
                "Reliable electrical connections",
                "Weather-resistant design",
                "OEM-quality components"
            ])
        
        return features[:5]  # Limit to 5 features
    
    def optimize_for_seo_length(self, text: str, max_length: int) -> str:
        """
        Optimize text for SEO length requirements
        
        Args:
            text: Text to optimize
            max_length: Maximum allowed length
            
        Returns:
            str: Optimized text
        """
        if len(text) <= max_length:
            return text
        
        # Truncate and add ellipsis
        return text[:max_length - 3] + "..."


class ShopifyFormatGenerator:
    """Handles generation of Shopify-compatible product import format"""
    
    def __init__(self, column_requirements_path: str):
        self.column_requirements_path = Path(column_requirements_path)
        self.seo_generator = SEOContentGenerator()
        self.column_requirements = None
        self.required_columns = []
        
    def load_column_requirements(self) -> Dict[str, Any]:
        """
        Load column requirements from Python file
        
        Returns:
            Dict: Column requirements
            
        Raises:
            FileNotFoundError: If requirements file doesn't exist
            ShopifyFormatError: If requirements can't be loaded
        """
        try:
            if not self.column_requirements_path.exists():
                raise FileNotFoundError(f"Column requirements file not found: {self.column_requirements_path}")
            
            # Execute the Python file to get variables
            namespace = {}
            with open(self.column_requirements_path, 'r') as f:
                exec(f.read(), namespace)
            
            # Get the columns list
            if 'cols_list' in namespace:
                required_columns = namespace['cols_list']
            elif 'cols' in namespace:
                required_columns = namespace['cols'].split(',')
            else:
                raise ShopifyFormatError("No column list found in requirements file")
            
            # Clean column names
            required_columns = [col.strip() for col in required_columns]
            
            self.required_columns = required_columns
            self.column_requirements = {
                'required_columns': required_columns,
                'column_order': required_columns,
                'total_columns': len(required_columns)
            }
            
            logger.info(f"Loaded {len(required_columns)} column requirements")
            return self.column_requirements
            
        except Exception as e:
            logger.error(f"Failed to load column requirements: {str(e)}")
            raise ShopifyFormatError(f"Failed to load column requirements: {str(e)}")
    
    def generate_shopify_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate Shopify-compatible format from product data
        
        Args:
            df: DataFrame with product data
            
        Returns:
            pd.DataFrame: Shopify-formatted DataFrame
        """
        if self.column_requirements is None:
            self.load_column_requirements()
        
        if df.empty:
            # Return empty DataFrame with correct columns
            empty_df = pd.DataFrame(columns=self.required_columns)
            return empty_df
        
        shopify_df = pd.DataFrame()
        
        # Process each product
        for index, row in df.iterrows():
            try:
                # Map product data to Shopify columns
                mapped_data = self.map_product_data_to_shopify_columns(row)
                
                # Add default values
                defaults = self.set_default_shopify_values()
                
                # Combine mapped data with defaults
                product_row = {**defaults, **mapped_data}
                
                # Ensure all required columns are present
                for col in self.required_columns:
                    if col not in product_row:
                        product_row[col] = self._get_default_value_for_column(col)
                
                # Add to DataFrame
                shopify_df = pd.concat([shopify_df, pd.DataFrame([product_row])], ignore_index=True)
                
            except Exception as e:
                logger.error(f"Failed to process product {index}: {str(e)}")
                # Add row with default values to maintain row count
                default_row = self.set_default_shopify_values()
                for col in self.required_columns:
                    if col not in default_row:
                        default_row[col] = self._get_default_value_for_column(col)
                shopify_df = pd.concat([shopify_df, pd.DataFrame([default_row])], ignore_index=True)
        
        # Ensure column order matches requirements
        shopify_df = shopify_df.reindex(columns=self.required_columns, fill_value='')
        
        logger.info(f"Generated Shopify format for {len(shopify_df)} products")
        return shopify_df
    
    def map_product_data_to_shopify_columns(self, row: pd.Series) -> Dict[str, Any]:
        """
        Map product data to Shopify columns
        
        Args:
            row: Product data row
            
        Returns:
            Dict: Mapped Shopify data
        """
        mapped_data = {}
        
        # Basic product information
        mapped_data['Title'] = row.get('title', 'Automotive Part')
        mapped_data['Body HTML'] = row.get('body_html', self.seo_generator.generate_body_html(row.to_dict()))
        mapped_data['Vendor'] = row.get('make', 'Unknown')
        
        # Tags
        mapped_data['Tags'] = self.generate_tags_from_product_data(row)
        
        # Variant information
        mapped_data['Variant SKU'] = row.get('mpn', 'UNKNOWN')
        mapped_data['Variant Price'] = float(row.get('price', 0))
        mapped_data['Variant Cost'] = float(row.get('cost', 0))
        
        # SEO fields
        mapped_data['Metafield: title_tag [string]'] = row.get('meta_title', self.seo_generator.generate_meta_title(row.to_dict()))
        mapped_data['Metafield: description_tag [string]'] = row.get('meta_description', self.seo_generator.generate_meta_description(row.to_dict()))
        
        # Google Shopping fields
        mapped_data['Variant Metafield: mm-google-shopping.mpn [single_line_text_field]'] = row.get('mpn', '')
        mapped_data['Variant Metafield: mm-google-shopping.condition [single_line_text_field]'] = 'new'
        mapped_data['Metafield: mm-google-shopping.mpn [single_line_text_field]'] = row.get('mpn', '')
        
        # Product categorization
        mapped_data['Category'] = row.get('collection', 'Automotive Parts')
        mapped_data['Custom Collections'] = row.get('collection', 'Automotive Parts')
        
        return mapped_data
    
    def generate_handle_from_title(self, title: str) -> str:
        """
        Generate Shopify handle from product title
        
        Args:
            title: Product title
            
        Returns:
            str: Shopify handle
        """
        if not title:
            return 'automotive-part'
        
        # Convert to lowercase and replace spaces with hyphens
        handle = title.lower()
        
        # Remove special characters except hyphens and alphanumeric
        handle = re.sub(r'[^a-z0-9\s\-]', '', handle)
        
        # Replace spaces with hyphens
        handle = re.sub(r'\s+', '-', handle)
        
        # Remove multiple consecutive hyphens
        handle = re.sub(r'-+', '-', handle)
        
        # Remove leading/trailing hyphens
        handle = handle.strip('-')
        
        return handle or 'automotive-part'
    
    def generate_tags_from_product_data(self, row: pd.Series) -> str:
        """
        Generate ONLY vehicle-specific YEAR_MAKE_MODEL format tags from golden master
        
        This extracts vehicle compatibility tags from golden master car_id field.
        No descriptive tags - only vehicle-specific fitment tags.
        
        Args:
            row: Product data row with vehicle information
            
        Returns:
            str: Comma-separated vehicle-specific tags in YEAR_MAKE_MODEL format
        """
        # Import here to avoid circular imports
        from utils.golden_master_tag_generator import GoldenMasterTagGenerator
        
        # Initialize tag generator if needed
        if not hasattr(self, 'tag_generator') or self.tag_generator is None:
            try:
                # Load golden master data
                import pandas as pd
                import os
                
                golden_master_path = '../shared/data/master_ultimate_golden.csv'
                if not os.path.exists(golden_master_path):
                    golden_master_path = 'shared/data/master_ultimate_golden.csv'
                
                if os.path.exists(golden_master_path):
                    golden_df = pd.read_csv(golden_master_path)
                    self.tag_generator = GoldenMasterTagGenerator(golden_df)
                else:
                    logger.warning("Golden master file not found, tags will be empty")
                    self.tag_generator = None
            except Exception as e:
                logger.error(f"Failed to initialize tag generator: {str(e)}")
                self.tag_generator = None
        
        # Extract vehicle information from row
        make = row.get('make', '')
        model = row.get('model', '')
        year_min = row.get('year_min', 1900)
        year_max = row.get('year_max', 2024)
        
        # Handle model parsing (could be string or list)
        models = []
        if isinstance(model, str):
            if model.lower() in ['unknown', 'universal', 'all', 'none']:
                models = []
            else:
                # Split on common delimiters
                models = [m.strip() for m in model.replace(',', '|').split('|') if m.strip()]
        elif isinstance(model, list):
            models = [m for m in model if m and str(m).lower() not in ['unknown', 'universal', 'all', 'none']]
        
        # Generate vehicle-specific tags using golden master
        if self.tag_generator and make and models:
            try:
                # Debug: Show year range and vehicle info
                logger.info(f"🔍 Tag Generation Debug: year_min={year_min}, year_max={year_max}, make='{make}', models={models}")
                
                vehicle_tags = self.tag_generator.generate_vehicle_tags_from_car_ids(
                    int(year_min), int(year_max), make, models
                )
                
                if vehicle_tags:
                    logger.info(f"✅ Generated {len(vehicle_tags)} vehicle tags for {make} {models}: {vehicle_tags}")
                    return ', '.join(vehicle_tags)
                else:
                    logger.info(f"⚠️ No vehicle tags found for year_min={year_min}, year_max={year_max}, make='{make}', models={models}")
                    return ''
                    
            except Exception as e:
                logger.error(f"❌ Failed to generate vehicle tags for year_min={year_min}, year_max={year_max}, make='{make}', models={models}: {str(e)}")
                return ''
        else:
            # No specific vehicle compatibility - return empty tags
            logger.info(f"ℹ️ No vehicle tags generated: year_min={year_min}, year_max={year_max}, make='{make}', models={models}, tag_generator={self.tag_generator is not None}")
            return ''
    
    
    def set_default_shopify_values(self) -> Dict[str, Any]:
        """
        Set default values for Shopify fields
        
        Returns:
            Dict: Default values
        """
        defaults = {
            'Command': 'MERGE',
            'Published': True,
            'Variant Command': 'MERGE',
            'Variant Inventory Tracker': 'shopify',
            'Variant Fulfillment Service': 'manual',
            'Variant Requires Shipping': True,
            'Variant Taxable': True,
            'Gift Card': False,
            'Variant Inventory Policy': 'deny',
            'Variant Weight Unit': 'lb',
            'Variant Country of Origin': 'US',
            'Image Position': 1,
            'Variant Position': 1,
            'Tags Command': 'REPLACE'
        }
        
        return defaults
    
    def _get_default_value_for_column(self, column: str) -> Any:
        """
        Get default value for a specific column
        
        Args:
            column: Column name
            
        Returns:
            Any: Default value
        """
        # Boolean fields
        if any(keyword in column.lower() for keyword in ['published', 'taxable', 'shipping', 'gift']):
            if 'gift' in column.lower():
                return False
            return True
        
        # Numeric fields
        if any(keyword in column.lower() for keyword in ['price', 'cost', 'weight', 'qty', 'position']):
            return 0
        
        # Command fields
        if 'command' in column.lower():
            return 'MERGE'
        
        # Default to empty string
        return ''
    
    def validate_column_compliance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate column compliance with Shopify requirements
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dict: Validation results
        """
        if self.column_requirements is None:
            self.load_column_requirements()
        
        required_columns = set(self.required_columns)
        actual_columns = set(df.columns)
        
        missing_columns = required_columns - actual_columns
        extra_columns = actual_columns - required_columns
        
        is_compliant = (
            len(missing_columns) == 0 and
            len(df.columns) == len(self.required_columns)
        )
        
        validation_result = {
            'is_compliant': is_compliant,
            'column_count': len(df.columns),
            'expected_count': len(self.required_columns),
            'missing_columns': list(missing_columns),
            'extra_columns': list(extra_columns),
            'validation_errors': []
        }
        
        # Add specific validation errors
        if missing_columns:
            validation_result['validation_errors'].append(f"Missing {len(missing_columns)} required columns")
        
        if extra_columns:
            validation_result['validation_errors'].append(f"Found {len(extra_columns)} unexpected columns")
        
        # Validate required data
        if not df.empty:
            # Check for missing titles
            if 'Title' in df.columns and df['Title'].isna().any():
                validation_result['validation_errors'].append("Missing product titles detected")
            
            # Check for missing SKUs
            if 'Variant SKU' in df.columns and df['Variant SKU'].isna().any():
                validation_result['validation_errors'].append("Missing variant SKUs detected")
            
            # Check for missing prices
            if 'Variant Price' in df.columns and (df['Variant Price'].isna().any() or (df['Variant Price'] == 0).any()):
                validation_result['validation_errors'].append("Missing or zero prices detected")
        
        return validation_result
    
    def generate_validation_report(self, df: pd.DataFrame) -> str:
        """
        Generate human-readable validation report
        
        Args:
            df: DataFrame to validate
            
        Returns:
            str: Validation report
        """
        validation_result = self.validate_column_compliance(df)
        
        report_lines = [
            "Shopify Format Validation Report",
            "=" * 40,
            f"Status: {'PASS' if validation_result['is_compliant'] else 'FAIL'}",
            f"Total Products: {len(df)}",
            f"Column Count: {validation_result['column_count']} (Expected: {validation_result['expected_count']})",
            ""
        ]
        
        if validation_result['missing_columns']:
            report_lines.extend([
                "Missing Columns:",
                *[f"  - {col}" for col in validation_result['missing_columns']],
                ""
            ])
        
        if validation_result['extra_columns']:
            report_lines.extend([
                "Extra Columns:",
                *[f"  - {col}" for col in validation_result['extra_columns']],
                ""
            ])
        
        if validation_result['validation_errors']:
            report_lines.extend([
                "Validation Errors:",
                *[f"  - {error}" for error in validation_result['validation_errors']],
                ""
            ])
        
        if validation_result['is_compliant']:
            report_lines.append("✅ All validation checks passed!")
        else:
            report_lines.append("❌ Validation failed. Please review and fix issues above.")
        
        return "\n".join(report_lines)