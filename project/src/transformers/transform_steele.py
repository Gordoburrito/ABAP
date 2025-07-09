import pandas as pd

def read_excel_file(file_path):
    """
    Read an Excel file and return a dictionary of DataFrames, one for each sheet
    
    Args:
        file_path (str): Path to the Excel file
        
    Returns:
        dict: Dictionary of DataFrames, keys are sheet names
    """
    try:
        # Read all sheets into a dictionary of DataFrames
        dfs = pd.read_excel(file_path, sheet_name=None)
        return dfs
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error reading Excel file: {str(e)}")
        return None

file_path = "./data/steele.xlsx"
data = read_excel_file(file_path)

if data is not None:
    print("Excel file successfully loaded!")
    
    # Get the sheet names and DataFrames
    sheet_names = list(data.keys())
    if len(sheet_names) >= 2:
        merged_df = pd.merge(
            data[sheet_names[0]],
            data[sheet_names[1]],
            how='inner',
            left_on=['StockCode'],
            right_on=['PartNumber']
        )
        
        print("\nMerged DataFrame:")
        print(merged_df.head())
        
        # Save first 20 rows to CSV in current directory
        sample_path = "merged_sample.csv"
        merged_df.head(20).to_csv(sample_path, index=False)
        print(f"\nSaved sample of 20 rows to {sample_path}")
    else:
        print("Error: At least two sheets are required for merging")
