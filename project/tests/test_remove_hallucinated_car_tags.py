import pandas as pd
import pytest
from src.extract_golden_df import load_master_ultimate_golden_df
from src.formatters.remove_hallucinated_car_tags import remove_hallucinated_car_tags, remove_hallucinated_car_tags_from_df
import json

def test_remove_hallucinated_car_tags():
    # get the golden df
    golden_df = load_master_ultimate_golden_df()
    golden_tags = golden_df["car_id"].unique()
    test_tags_str = "1938_Chrysler_Imperial,1939_Chrysler_Imperial,1939_Chrysler_Imperial,foo_tag_1,foo_tag_2,foo_tag_3"
    real_tags = remove_hallucinated_car_tags(test_tags_str, golden_tags)
    
    # sort the tags
    real_tags = real_tags.split(",")
    real_tags = sorted(real_tags)
    real_tags = ",".join(real_tags)

    assert real_tags == "1938_Chrysler_Imperial,1939_Chrysler_Imperial"

# fails
def test_remove_hallucinated_car_tags_from_df():
    # Create test DataFrame
    test_df = pd.DataFrame({
        'Tag': [
            '1938_Chrysler_Imperial,fake_car_1,1939_Chrysler_Imperial',
            'fake_car_2,1938_Chrysler_Imperial',
            'all,fake,tags,here',
            '',  # Empty string case
            '1939_Chrysler_Imperial'
        ]
    })

    # Get golden tags
    golden_df = load_master_ultimate_golden_df()
    golden_tags = golden_df["car_id"].unique()

    # Process DataFrame
    result_df = remove_hallucinated_car_tags_from_df(test_df, golden_tags)

    # Sort tags in each row before comparison
    result_df['Tag'] = result_df['Tag'].apply(lambda x: ','.join(sorted(x.split(','))) if x else '')

    # Assert results
    expected_tags = [
        '1938_Chrysler_Imperial,1939_Chrysler_Imperial',
        '1938_Chrysler_Imperial',
        '',
        '',
        '1939_Chrysler_Imperial'
    ]
    
    pd.testing.assert_series_equal(
        result_df['Tag'],
        pd.Series(expected_tags, name='Tag'),
        check_names=False
    )