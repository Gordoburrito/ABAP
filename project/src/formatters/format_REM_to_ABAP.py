import pandas as pd
from pathlib import Path
from src.format_product_data_to_ABAP import format_product_data_to_ABAP
from src.extract_golden_df import load_master_ultimate_golden_df
# Get the project root directory (2 levels up from this file)
project_root = Path(__file__).parent.parent.parent

# Read the CSV files using the correct paths
product_data = pd.read_csv(project_root / "data/transformed/REM_transformed.csv")

print(product_data.head())

def format_REM_to_ABAP(product_data):
    golden_df = load_master_ultimate_golden_df()
    return format_product_data_to_ABAP(product_data, golden_df)

formatted_df = format_REM_to_ABAP(product_data)
print(formatted_df)

# Save the formatted DataFrame to a CSV file
output_path = project_root / "data/formatted/REM_formatted.csv"
# Create the formatted directory if it doesn't exist
output_path.parent.mkdir(parents=True, exist_ok=True)
formatted_df.to_csv(output_path, index=False)
print(f"Formatted data saved to: {output_path}")
