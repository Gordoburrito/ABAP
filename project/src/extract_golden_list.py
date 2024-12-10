import pandas as pd

def load_golden_list(filepath: str) -> tuple:
    """
    Load the golden copy dataset from a CSV file.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        tuple: Tuple containing lists of unique car IDs, years, makes, and models
    """
    df = pd.read_csv(filepath)
    car_ids = df['Base Car ID'].unique().tolist()
    years = df['Year Min'].unique().tolist()
    makes = df['Make|Include'].unique().tolist()
    models = df['Model|Include'].unique().tolist()
    return car_ids, years, makes, models
