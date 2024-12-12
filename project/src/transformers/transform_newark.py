import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transform import transform_data_with_ai

def transform_newark(input_file: str, golden_file: str) -> pd.DataFrame:
    """
    Transform Newark vendor data to the standardized format.
    
    Args:
        input_file: Path to Newark CSV file
        golden_file: Path to golden copy CSV file
    """
    # Read the input CSV
    vendor_df = pd.read_csv(input_file)
    
    # Print columns to debug
    print("Original columns:", vendor_df.columns.tolist())
    
    # Clean and rename columns
    vendor_df = vendor_df.rename(columns={
        'Part ID': 'mpn',
        'Product Name': 'title',
        'Year Start': 'year_min',
        'Year End': 'year_max',
        'Price Level 1': 'cost',  # Vendor cost
        'Price Level 2': 'price', # Retail price
        'Description': 'body_html'
    })
    
    # Print columns after rename to debug
    print("Columns after rename:", vendor_df.columns.tolist())
    
    # Clean price columns - remove '$' and convert to float
    vendor_df['cost'] = vendor_df['cost'].str.strip().str.replace('$', '').str.strip().astype(float)
    vendor_df['price'] = vendor_df['price'].str.strip().str.replace('$', '').str.strip().astype(float)
    
    # Read golden copy
    golden_df = pd.read_csv(golden_file)
    
    # Transform the data using the common transformer
    return transform_data_with_ai(golden_df, vendor_df)

if __name__ == "__main__":
    result_df = transform_newark('./data/samples/newark_sample.csv', './data/golden.csv')
    print(result_df)
    
    # Save the result to CSV
    output_path = './data/transformed/newark_transformed.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Create directory if it doesn't exist
    result_df.to_csv(output_path, index=False)
    print(f"Transformed data saved to: {output_path}")
