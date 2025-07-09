import pandas as pd
def remove_hallucinated_car_tags(test_tags_str, golden_tags):
    """
    Remove hallucinated car tags from the dataframe.
    """
    # print(f"test_tags_str: {test_tags_str}")

    test_tags = test_tags_str.split(",")

    # Find tags that are in test_tags but not in golden_tags
    hallucinated_tags = set(test_tags) - set(golden_tags)
    real_tags = set(test_tags) & set(golden_tags)

    # if hallucinated_tags:
    if hallucinated_tags:
        print(f"Found {len(hallucinated_tags)} hallucinated tags: {hallucinated_tags}")
    else:
        print("No hallucinated tags found")
    return ",".join(real_tags)


def remove_hallucinated_car_tags_from_df(df: pd.DataFrame, golden_tags: list[str]) -> pd.DataFrame:
    """
    Remove hallucinated car tags from the dataframe.
    """
    df["Tag"] = df["Tag"].apply(lambda x: remove_hallucinated_car_tags(x, golden_tags))
    return df
