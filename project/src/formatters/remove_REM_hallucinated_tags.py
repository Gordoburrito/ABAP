import pandas as pd
from src.extract_golden_df import load_master_ultimate_golden_df
from src.formatters.remove_hallucinated_car_tags import remove_hallucinated_car_tags_from_df

def remove_REM_hallucinated_tags() -> pd.DataFrame:
    """
    Remove hallucinated tags from the dataframe.
    """
    # load the REM df
    rem_df = pd.read_csv("data/formatted/REM_formatted_v3.1.csv")
    
    # Handle NaN values in the dataframe
    rem_df = rem_df.fillna("")  # Replace NaN with empty string

    # load the golden df
    golden_df = load_master_ultimate_golden_df()
    golden_tags = golden_df["car_id"].unique()

    # remove the hallucinated tags
    df = remove_hallucinated_car_tags_from_df(rem_df, golden_tags)

    # save the df
    return df

if __name__ == "__main__":
    remove_REM_hallucinated_tags()