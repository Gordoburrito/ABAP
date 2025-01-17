import pytest
import pandas as pd
from pathlib import Path
from src.format_product_data_to_ABAP import format_product_data_to_ABAP, get_year_range, expand_all_models
from src.extract_golden_df import load_golden_df

def test_get_year_range():
    year_min = "1960"
    year_max = "1965"
    assert get_year_range(year_min, year_max) == ["1960", "1961", "1962", "1963", "1964", "1965"]

def test_expand_all_models():
    golden_df = load_golden_df()
    year = "1964"
    make = "Plymouth"
    models = expand_all_models(year, make, golden_df)
    assert sorted(list(models)) == sorted(["Valiant", "Savoy", "Fury", "Belvedere", "Barracuda"])


def test_transform_product_data():
    # Read input files
    project_root = Path(__file__).parent.parent
    product_data = pd.read_csv(project_root / "data/samples/product_data_samp.csv")

    # Format the data
    golden_df = load_golden_df()
    result = format_product_data_to_ABAP(product_data, golden_df)

    # Verify key columns exist and have expected format
    assert "Tag" in result.columns

    # Test specific row transformations
    plymouth_row = result[result["Title"].str.contains("64/65 Plymouth")].iloc[0]

    # Test ALL model expansion and Year Expansion
    car_ids = plymouth_row["Tag"].split(", ")
    expected_ids = ['1964_Plymouth_Valiant', '1964_Plymouth_Savoy', '1964_Plymouth_Fury', 
                   '1964_Plymouth_Belvedere', '1964_Plymouth_Barracuda', '1965_Plymouth_Valiant', 
                   '1965_Plymouth_Satellite', '1965_Plymouth_Fury III', '1965_Plymouth_Fury II', 
                   '1965_Plymouth_Fury', '1965_Plymouth_Belvedere II', '1965_Plymouth_Belvedere', 
                   '1965_Plymouth_Barracuda']
    assert sorted(car_ids) == sorted(expected_ids)

    # Check year range expansion
    satellite_row = result[result["Title"].str.contains("66/70 Satellite")].iloc[0]
    car_ids = satellite_row["Tag"].split(", ")
    print(car_ids)
    expected_ids = [
        "1966_Plymouth_Satellite",
        "1966_Plymouth_Belvedere",
        "1967_Plymouth_Satellite",
        "1967_Plymouth_Belvedere",
        "1968_Plymouth_Satellite",
        "1968_Plymouth_Belvedere",
        "1969_Plymouth_Satellite",
        "1969_Plymouth_Belvedere",
        "1970_Plymouth_Satellite",
        "1970_Plymouth_Belvedere",
    ]
    assert car_ids == expected_ids
