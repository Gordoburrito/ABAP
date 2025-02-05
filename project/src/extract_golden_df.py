import pandas as pd

def load_golden_df() -> tuple:
    """Load the golden copy dataset from CSV."""
    column_mapping = {
        'Base Car ID': 'car_id',
        'Year Min': 'year', 
        'Make|Include': 'make',
        'Model|Include': 'model',
        'Engine Tag': 'engine'
    }
    
    return pd.read_csv(
        "data/golden.csv",
        usecols=column_mapping.keys(),
        dtype={'Year Min': str},
    ).rename(columns=column_mapping)[list(column_mapping.values())]

def load_master_ultimate_golden_df() -> tuple:
    column_mapping = {
        'Car ID': 'car_id',
        'Year': 'year',
        'Make': 'make',
        'Model': 'model',
        'Engine': 'engine',
        'Engine Tag': 'engine_ids',
        'Human Readable': 'engine_readable'
    }

    return pd.read_csv(
        "data/master_ultimate_golden.csv",
        usecols=column_mapping.keys(),
        dtype={'Year': str},
        low_memory=False,
    ).rename(columns=column_mapping)[list(column_mapping.values())]
