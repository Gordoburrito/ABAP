import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transform import transform_data_with_ai
from src.extract_golden_df import load_master_ultimate_golden_df

def transform_REM(input_file: str) -> pd.DataFrame:
    """
    Transform REM vendor data to the standardized format.
    
    Args:
        input_file: Path to REM CSV file
    """
    # Read the input CSV
    vendor_df = pd.read_csv(input_file)
    
    # Print columns to debug
    print("Original columns:", vendor_df.columns.tolist())
    
    # Clean and rename columns
    vendor_df = vendor_df.rename(columns={
        'Inventory ID': 'SKU',
        ' Level-3 ': 'Price',  # Note the space at the end
        'Description': 'Description'
    })
    
    # Print columns after rename to debug
    print("Columns after rename:", vendor_df.columns.tolist())
    
    # Clean price column - remove '$' and convert to float
    vendor_df['Price'] = vendor_df['Price'].str.strip().str.replace('$', '').str.strip().astype(float)
    
    # Read golden copy
    golden_df = load_master_ultimate_golden_df()
    
    # Transform the data using the common transformer
    return transform_data_with_ai(golden_df, vendor_df)

if __name__ == "__main__":
    result_df = transform_REM('./data/REM.csv')
    print(result_df)
    
    # Save the result to CSV
    output_path = './data/transformed/REM_transformed.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Create directory if it doesn't exist
    result_df.to_csv(output_path, index=False)
    print(f"Transformed data saved to: {output_path}")
