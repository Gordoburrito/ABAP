import pandas as pd
from typing import Optional
from openai import OpenAI
import os
from dotenv import load_dotenv
from models.product_data import ProductData

def extract_product_data_with_ai(row: pd.Series, golden_df: pd.DataFrame, client: OpenAI) -> ProductData:
    """
    Convert a single vendor row to ProductData format, using OpenAI for missing information
    """
    # Get valid options from golden copy, filtering out NaN values
    valid_makes = sorted([x for x in golden_df['Make|Include'].unique() if pd.notna(x)])
    valid_models = sorted([x for x in golden_df['Model|Include'].unique() if pd.notna(x)])
    
    # Create a range of valid years from Year Min to Year Max
    years = []
    for _, golden_row in golden_df.iterrows():
        min_year = golden_row['Year Min']
        max_year = golden_row['Year Max']
        if pd.notna(min_year) and pd.notna(max_year):
            years.extend(range(int(min_year), int(max_year) + 1))
    valid_years = sorted(list(set(years)))
    
    # Create a clean product info string from available row data
    product_info = "\n".join([f"{key}: {value}" for key, value in row.items() if pd.notna(value)])

    prompt = f"""Based on this product information, determine the car details and create a product description.
    If the product is compatible with ALL makes, models, or years, indicate this by using "ALL" for that field.
    If specific options are required, choose ONLY from the provided valid options.
    
    Product Information:
    {product_info}
    
    Valid Options:
    Makes: {valid_makes} (or "ALL" if compatible with all makes)
    Models: {valid_models} (or "ALL" if compatible with all models)
    Years: {valid_years} (or "ALL" if compatible with all years)
    
    Note: Use "ALL" when the product is universally compatible for that category.
    """
    
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Extract the product information based on the valid options provided. Use 'ALL' for universal compatibility."},
            {"role": "user", "content": prompt}
        ],
        response_format=ProductData,
        temperature=0.7
    )
    
    product_data = response.choices[0].message.parsed
    
    return product_data

def transform_data(golden_df: pd.DataFrame, vendor_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform vendor data to required format, validating against golden copy.
    """
    # Load environment variables
    load_dotenv()
    
    # Initialize OpenAI client with API key from environment
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # Transform each row
    transformed_records = []
    for index, vendor_row in vendor_df.iterrows():
        print(f"Processing row {index}:")
        transformed_record = extract_product_data_with_ai(vendor_row, golden_df, client)
        transformed_records.append(transformed_record.model_dump())
    
    # Create DataFrame with specific column order
    result_df = pd.DataFrame(transformed_records)
    result_df = result_df[[
        'title',
        'year_min',
        'year_max',
        'make',
        'model',
        'mpn',
        'cost',
        'price',
        'body_html'
    ]]
    
    return result_df 