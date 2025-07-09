import pandas as pd

def format_vehicle_data_to_JSON(df: pd.DataFrame) -> pd.DataFrame:
    return df.to_json(orient='records')
