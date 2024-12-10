import pandas as pd
import pytest
from src.transform import transform_data

def test_transform_data():
    # Load test data
    golden_df = pd.read_csv('data/sample_golden_copy.csv')
    vendor_df = pd.read_csv('data/sample_vendor_x.csv')
    
    # Transform data
    result_df = transform_data(golden_df, vendor_df)
    
    # Assert expected columns are present
    expected_cols = [
        'Material Number', 
        'Material Description', 
        'Material Type',
        'Base Price',
        'Currency',
        'Base Car ID'
    ]
    assert all(col in result_df.columns for col in expected_cols)
    
    # Test specific mapping
    test_row = result_df[result_df['Base Car ID'] == 'DODGE_001'].iloc[0]
    assert test_row['Material Number'] == 'MAT123'
    assert test_row['Material Type'] == 'ZAUT'
    assert test_row['Base Price'] == 299.99
    assert test_row['Currency'] == 'USD'
    
    # Test that inactive cars are filtered out
    assert 'CHEV_001' not in result_df['Base Car ID'].values

def test_transform_handles_empty_data():
    golden_df = pd.DataFrame(columns=['Base Car ID', 'Make', 'Model', 'Year', 'Status'])
    vendor_df = pd.DataFrame(columns=['SKU', 'Title', 'Price', 'Base Car ID'])
    
    result_df = transform_data(golden_df, vendor_df)
    assert len(result_df) == 0