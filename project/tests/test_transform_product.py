import pytest
import pandas as pd
from pathlib import Path

def test_transform_product_data():
    # Read input files
    project_root = Path(__file__).parent.parent
    product_data = pd.read_csv(project_root / "data/samples/product_data_samp.csv")
    golden_data = pd.read_csv(project_root / "data/golden.csv")
    expected_abap = pd.read_csv(project_root / "data/samples/ABAP_sample.csv")

    # Transform the data
    result = transform_product_data_to_ABAP(product_data, golden_data)

    # Verify key columns exist and have expected format
    assert "(Internal) Car ID - Expanded" in result.columns
    
    # Test specific row transformations
    plymouth_row = result[result["Title"].str.contains("64/65 Plymouth")].iloc[0]
    
    # Check year range expansion
    car_ids = plymouth_row["(Internal) Car ID - Expanded"].split(", ")
    assert "1964_Plymouth_ALL" in car_ids
    assert "1965_Plymouth_ALL" in car_ids

    # Test ALL model expansion
    satellite_row = result[result["Title"].str.contains("66/70 Satellite")].iloc[0]
    car_ids = satellite_row["(Internal) Car ID - Expanded"].split(", ")
    assert "1966_Plymouth_Satellite" in car_ids
    assert "1967_Plymouth_Satellite" in car_ids
    assert "1968_Plymouth_Satellite" in car_ids
    assert "1969_Plymouth_Satellite" in car_ids
    assert "1970_Plymouth_Satellite" in car_ids 