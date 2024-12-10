import pandas as pd
import pytest
from src.extract_golden_list import load_golden_list

def test_load_golden_list():
    car_ids, years, makes, models = load_golden_list('data/golden.csv')
    assert len(car_ids) == 1206
    assert len(years) == 56
    assert len(makes) == 5
    assert len(models) == 325