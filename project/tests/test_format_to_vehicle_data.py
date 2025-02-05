import pandas as pd
import pytest
from src.format_vehicle_data_to_JSON import format_vehicle_data_to_JSON
import json

def test_format_to_vehicle_data():
    transformed_df = pd.read_csv("data/master_ultimate_golden_transformed.csv")
    json_str = format_vehicle_data_to_JSON(transformed_df)
    json_data = json.loads(json_str)
    print("First 5 items:")
    for i, item in enumerate(json_data[:5]):
        print(item)
    assert len(json_data) == 1206