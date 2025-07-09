import pandas as pd
import os
import sys
from pathlib import Path
from typing import Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
steele_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

class SteeleRawDataProcessor:
    """
    Processes raw Steele data from Excel format with two sheets:
    1. Product data sheet (StockCode, Product Name, Description, etc.)
    2. Fitment data sheet (PartNumber, Year, Make, Model, etc.)
    
    Combines them into a processed CSV with one row per product-fitment combination.
    """
    
    def __init__(self):
        self.steele_root = steele_root
        self.raw_data_path = steele_root / "data" / "raw"
        self.processed_data_path = steele_root / "data" / "processed"
        
        # Ensure processed directory exists
        self.processed_data_path.mkdir(exist_ok=True)
    
    def load_raw_excel_data(self, excel_file_path: str = "steele.xlsx") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load both sheets from the raw Steele Excel file.
        
        Args:
            excel_file_path: Name of Excel file in raw data directory
            
        Returns:
            Tuple of (product_data_df, fitment_data_df)
        """
        full_path = self.raw_data_path / excel_file_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"Raw Excel file not found: {full_path}")
        
        print(f"📖 Loading raw Excel data from: {excel_file_path}")
        
        try:
            # Load both sheets
            excel_data = pd.ExcelFile(full_path)
            sheet_names = excel_data.sheet_names
            print(f"   Found sheets: {sheet_names}")
            
            if len(sheet_names) < 2:
                raise ValueError(f"Expected at least 2 sheets, found {len(sheet_names)}: {sheet_names}")
            
            # Load first sheet (product data)
            product_df = pd.read_excel(full_path, sheet_name=sheet_names[0])
            print(f"   Sheet 1 ({sheet_names[0]}): {len(product_df)} products")
            print(f"   Product columns: {list(product_df.columns)}")
            
            # Load second sheet (fitment data)
            fitment_df = pd.read_excel(full_path, sheet_name=sheet_names[1])
            print(f"   Sheet 2 ({sheet_names[1]}): {len(fitment_df)} fitment records")
            print(f"   Fitment columns: {list(fitment_df.columns)}")
            
            return product_df, fitment_df
            
        except Exception as e:
            raise ValueError(f"Error loading Excel file: {str(e)}")
    
    def validate_data_structure(self, product_df: pd.DataFrame, fitment_df: pd.DataFrame) -> None:
        """
        Validate that both DataFrames have expected columns.
        
        Args:
            product_df: Product data DataFrame
            fitment_df: Fitment data DataFrame
        """
        # Expected columns for product data
        required_product_cols = [
            'StockCode', 'Product Name', 'Description', 'StockUom', 
            'UPC Code', 'MAP', 'Dealer Price'
        ]
        
        # Expected columns for fitment data
        required_fitment_cols = [
            'PartNumber', 'Year', 'Make', 'Model'
        ]
        
        # Optional fitment columns that we'll include if available
        optional_fitment_cols = [
            'Submodel', 'Type', 'Doors', 'BodyType'
        ]
        
        # Check product columns
        missing_product_cols = set(required_product_cols) - set(product_df.columns)
        if missing_product_cols:
            raise ValueError(f"Missing required product columns: {missing_product_cols}")
        
        # Check fitment columns
        missing_fitment_cols = set(required_fitment_cols) - set(fitment_df.columns)
        if missing_fitment_cols:
            raise ValueError(f"Missing required fitment columns: {missing_fitment_cols}")
        
        print("✅ Data structure validation passed")
        
        # Report on optional columns
        available_optional = set(optional_fitment_cols) & set(fitment_df.columns)
        missing_optional = set(optional_fitment_cols) - set(fitment_df.columns)
        
        if available_optional:
            print(f"   Optional fitment columns available: {list(available_optional)}")
        if missing_optional:
            print(f"   Optional fitment columns missing: {list(missing_optional)}")
    
    def clean_and_standardize_data(self, product_df: pd.DataFrame, fitment_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Clean and standardize the data before joining.
        
        Args:
            product_df: Raw product data
            fitment_df: Raw fitment data
            
        Returns:
            Tuple of cleaned DataFrames
        """
        print("🧹 Cleaning and standardizing data...")
        
        # Clean product data
        product_clean = product_df.copy()
        
        # Ensure StockCode is string and remove any whitespace
        product_clean['StockCode'] = product_clean['StockCode'].astype(str).str.strip()
        
        # Clean text fields
        text_columns = ['Product Name', 'Description']
        for col in text_columns:
            if col in product_clean.columns:
                product_clean[col] = product_clean[col].astype(str).str.strip()
        
        # Clean numeric fields
        numeric_columns = ['MAP', 'Dealer Price']
        for col in numeric_columns:
            if col in product_clean.columns:
                product_clean[col] = pd.to_numeric(product_clean[col], errors='coerce')
        
        # Clean fitment data
        fitment_clean = fitment_df.copy()
        
        # Ensure PartNumber is string and matches StockCode format
        fitment_clean['PartNumber'] = fitment_clean['PartNumber'].astype(str).str.strip()
        
        # Clean vehicle data
        vehicle_columns = ['Make', 'Model', 'Submodel', 'BodyType']
        for col in vehicle_columns:
            if col in fitment_clean.columns:
                fitment_clean[col] = fitment_clean[col].astype(str).str.strip()
                # Replace empty strings with NaN for proper handling
                fitment_clean[col] = fitment_clean[col].replace('', pd.NA)
        
        # Clean Year column
        if 'Year' in fitment_clean.columns:
            fitment_clean['Year'] = pd.to_numeric(fitment_clean['Year'], errors='coerce')
        
        # Clean numeric columns in fitment data
        if 'Doors' in fitment_clean.columns:
            fitment_clean['Doors'] = pd.to_numeric(fitment_clean['Doors'], errors='coerce')
        
        print(f"✅ Cleaned {len(product_clean)} products and {len(fitment_clean)} fitment records")
        
        return product_clean, fitment_clean
    
    def join_product_and_fitment_data(self, product_df: pd.DataFrame, fitment_df: pd.DataFrame) -> pd.DataFrame:
        """
        Join product and fitment data on StockCode = PartNumber.
        
        Args:
            product_df: Cleaned product data
            fitment_df: Cleaned fitment data
            
        Returns:
            Combined DataFrame with one row per product-fitment combination
        """
        print("🔗 Joining product and fitment data...")
        
        # Perform left join to keep all products, even if no fitment data
        combined_df = product_df.merge(
            fitment_df,
            left_on='StockCode',
            right_on='PartNumber',
            how='left'
        )
        
        print(f"✅ Created {len(combined_df)} product-fitment combinations")
        
        # Report on join results
        products_with_fitment = combined_df['PartNumber'].notna().sum()
        products_without_fitment = combined_df['PartNumber'].isna().sum()
        
        print(f"   Products with fitment data: {products_with_fitment}")
        print(f"   Products without fitment data: {products_without_fitment}")
        
        if products_without_fitment > 0:
            print("   ⚠️  Some products have no fitment data - they will be included with empty vehicle fields")
        
        return combined_df
    
    def format_for_pipeline(self, combined_df: pd.DataFrame) -> pd.DataFrame:
        """
        Format the combined data to match the expected pipeline input format.
        
        Args:
            combined_df: Combined product and fitment data
            
        Returns:
            DataFrame formatted for the Steele pipeline
        """
        print("📋 Formatting for pipeline compatibility...")
        
        # Define expected output columns in correct order
        output_columns = [
            'StockCode', 'Product Name', 'Description', 'StockUom', 'UPC Code',
            'MAP', 'Dealer Price', 'PartNumber', 'Year', 'Make', 'Model',
            'Submodel', 'Type', 'Doors', 'BodyType'
        ]
        
        # Create formatted DataFrame
        formatted_df = pd.DataFrame()
        
        for col in output_columns:
            if col in combined_df.columns:
                formatted_df[col] = combined_df[col]
            else:
                # Add missing columns with default values
                if col == 'PartNumber':
                    # If PartNumber is missing, use StockCode
                    formatted_df[col] = combined_df['StockCode']
                elif col in ['Submodel', 'Type', 'BodyType']:
                    formatted_df[col] = 'U/K'  # Unknown
                elif col == 'Doors':
                    formatted_df[col] = 0.0
                else:
                    formatted_df[col] = ''
        
        # Fill NaN values with appropriate defaults
        formatted_df['PartNumber'] = formatted_df['PartNumber'].fillna(formatted_df['StockCode'])
        formatted_df['Submodel'] = formatted_df['Submodel'].fillna('Base')
        formatted_df['Type'] = formatted_df['Type'].fillna('Car        ')  # Note: original has spaces
        formatted_df['BodyType'] = formatted_df['BodyType'].fillna('U/K')
        formatted_df['Doors'] = formatted_df['Doors'].fillna(0.0)
        
        # Handle missing vehicle data
        formatted_df['Year'] = formatted_df['Year'].fillna(0).astype(int)
        formatted_df['Make'] = formatted_df['Make'].fillna('Unknown')
        formatted_df['Model'] = formatted_df['Model'].fillna('Unknown')
        
        print(f"✅ Formatted {len(formatted_df)} records for pipeline")
        print(f"   Columns: {list(formatted_df.columns)}")
        
        return formatted_df
    
    def save_processed_data(self, formatted_df: pd.DataFrame, 
                          output_filename: str = "steele_processed_complete.csv") -> str:
        """
        Save the processed data to CSV file.
        
        Args:
            formatted_df: Formatted DataFrame
            output_filename: Name of output file
            
        Returns:
            Path to saved file
        """
        output_path = self.processed_data_path / output_filename
        
        # Save to CSV
        formatted_df.to_csv(output_path, index=False)
        
        print(f"💾 Saved processed data to: {output_path}")
        print(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")
        
        return str(output_path)
    
    def generate_summary_report(self, formatted_df: pd.DataFrame) -> None:
        """
        Generate a summary report of the processed data.
        
        Args:
            formatted_df: Final formatted DataFrame
        """
        print("\n📊 PROCESSING SUMMARY REPORT")
        print("=" * 50)
        
        print(f"Total records: {len(formatted_df)}")
        print(f"Unique products: {formatted_df['StockCode'].nunique()}")
        
        # Vehicle data summary
        records_with_vehicles = formatted_df[
            (formatted_df['Year'] > 0) & 
            (formatted_df['Make'] != 'Unknown') &
            (formatted_df['Model'] != 'Unknown')
        ]
        
        print(f"Records with vehicle data: {len(records_with_vehicles)}")
        print(f"Records without vehicle data: {len(formatted_df) - len(records_with_vehicles)}")
        
        if len(records_with_vehicles) > 0:
            print(f"Year range: {records_with_vehicles['Year'].min()} - {records_with_vehicles['Year'].max()}")
            print(f"Unique makes: {records_with_vehicles['Make'].nunique()}")
            print(f"Unique models: {records_with_vehicles['Model'].nunique()}")
            
            # Top makes
            top_makes = records_with_vehicles['Make'].value_counts().head(5)
            print(f"Top 5 makes:")
            for make, count in top_makes.items():
                print(f"  {make}: {count} records")
        
        # Price data summary
        valid_prices = formatted_df[formatted_df['MAP'] > 0]
        if len(valid_prices) > 0:
            print(f"Products with valid MAP prices: {len(valid_prices)}")
            print(f"Price range: ${valid_prices['MAP'].min():.2f} - ${valid_prices['MAP'].max():.2f}")
            print(f"Average price: ${valid_prices['MAP'].mean():.2f}")
        
        print("=" * 50)
    
    def process_complete_pipeline(self, excel_file_path: str = "steele.xlsx",
                                output_filename: str = "steele_processed_complete.csv") -> str:
        """
        Execute the complete raw data processing pipeline.
        
        Args:
            excel_file_path: Name of Excel file in raw data directory
            output_filename: Name of output CSV file
            
        Returns:
            Path to processed CSV file
        """
        print("🚀 STEELE RAW DATA PROCESSING PIPELINE")
        print("=" * 50)
        
        try:
            # Step 1: Load raw Excel data
            product_df, fitment_df = self.load_raw_excel_data(excel_file_path)
            
            # Step 2: Validate data structure
            self.validate_data_structure(product_df, fitment_df)
            
            # Step 3: Clean and standardize
            product_clean, fitment_clean = self.clean_and_standardize_data(product_df, fitment_df)
            
            # Step 4: Join data
            combined_df = self.join_product_and_fitment_data(product_clean, fitment_clean)
            
            # Step 5: Format for pipeline
            formatted_df = self.format_for_pipeline(combined_df)
            
            # Step 6: Save processed data
            output_path = self.save_processed_data(formatted_df, output_filename)
            
            # Step 7: Generate summary report
            self.generate_summary_report(formatted_df)
            
            print(f"\n✅ Processing complete! Output file: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"\n❌ Processing failed: {str(e)}")
            raise


def main():
    """Main function to run the raw data processing."""
    
    processor = SteeleRawDataProcessor()
    
    try:
        # Process the raw data
        output_file = processor.process_complete_pipeline()
        
        print(f"\n🎉 Success! Processed data saved to: {output_file}")
        print("\nNext steps:")
        print("1. Review the processed data file")
        print("2. Run the main transformation pipeline with this data")
        print("3. Validate the final output")
        
    except Exception as e:
        print(f"\n💥 Error: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Check that steele.xlsx exists in data/raw/")
        print("2. Verify Excel file has two sheets with expected columns")
        print("3. Check file permissions")


if __name__ == "__main__":
    main() 