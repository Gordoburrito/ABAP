import pandas as pd
import pytest
from src.extract_golden_df import load_golden_df, load_master_ultimate_golden_df


# def test_load_golden_df():
#     df = load_golden_df()
#     # Check DataFrame structure
#     expected_columns = ["car_id", "year", "make", "model", "engine"]
#     assert isinstance(df, pd.DataFrame)
#     assert list(df.columns) == expected_columns
#     assert len(df) > 0  # Ensure the DataFrame is not empty

#     # Check unique value counts
#     assert len(df['car_id'].unique()) == 1206
#     assert len(df['year'].unique()) == 56
#     assert len(df['make'].unique()) == 5
#     assert len(df['model'].unique()) == 325
#     assert len(df['engine'].unique()) == 98

def test_load_master_ultimate_golden_df():
    df = load_master_ultimate_golden_df()
    # Check DataFrame structure
    print(df.head())
    expected_columns = ["car_id", "year", "make", "model", "engine", "engine_ids", "engine_readable"]
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == expected_columns
    assert len(df) > 0  # Ensure the DataFrame is not empty

    # Check unique value counts
    assert len(df['car_id'].unique()) == 47755
    assert len(df['year'].unique()) == 131
    assert len(df['make'].unique()) == 426
    assert len(df['model'].unique()) == 6849
    assert len(df['engine'].unique()) == 3031
    # These should two below be the same right?
    assert len(df['engine_ids'].unique()) == 1096
    assert len(df['engine_readable'].unique()) == 1399