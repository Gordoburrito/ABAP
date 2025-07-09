import pandas as pd
import os


def load_data(
    base_path="src/ABAP/meta_info/data/processed/meta_title_and_meta_description/",
):
    """
    Load all CSV files for sampling
    Returns a dictionary of dataframes
    """
    data = {}
    # Load main dataframes
    data['df1'] = pd.read_csv(os.path.join(base_path, "1-4o.csv"))
    data['df2'] = pd.read_csv(os.path.join(base_path, "2-4o.csv"))
    data['df3'] = pd.read_csv(os.path.join(base_path, "3-4o.csv"))

    # Load mini dataframes
    data['df1_mini'] = pd.read_csv(os.path.join(base_path, "1-4o-mini.csv"))
    data['df2_mini'] = pd.read_csv(os.path.join(base_path, "2-4o-mini.csv"))
    data['df3_mini'] = pd.read_csv(os.path.join(base_path, "3-4o-mini.csv"))

    return data


def find_unique_collections(df):
    """
    Find all unique collections in a dataframe,
    excluding collections that contain commas
    """
    # Filter out collections that contain commas
    filtered_collections = df[~df['Collection'].str.contains(',', na=False)]
    return filtered_collections['Collection'].unique()


def get_sample_from_each_collection(df):
    """
    Get one sample row from each unique collection in the dataframe
    Keep only title, Tag, meta_title, meta_description, product_description and validation error
    """
    collections = find_unique_collections(df)
    samples = []
    
    for collection in collections:
        # Get filtered rows for this collection
        filtered_df = df[df['Collection'] == collection]
        # Only try to get a sample if there are rows
        if not filtered_df.empty:
            sample_row = filtered_df.iloc[0]
            samples.append(sample_row)
    
    # Create DataFrame and keep only the specified columns
    samples_df = pd.DataFrame(samples)
    columns_to_keep = ['Title', 'Tag', 'meta_title', 'meta_description', 'product_description', 'validation_error']
    # Only keep columns that exist in the DataFrame
    available_columns = [col for col in columns_to_keep if col in samples_df.columns]
    return samples_df[available_columns]


def create_sample_spreadsheets(output_dir="src/ABAP/meta_info/data/samples/"):
    """
    Create separate spreadsheets with samples from each dataframe
    """
    # Load all data
    data = load_data()
    
    print("Loaded all dataframes successfully")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each dataframe
    for df_name, df in data.items():
        print(f"\nProcessing {df_name}...")
        
        # Get sample from each collection
        samples = get_sample_from_each_collection(df)
        
        # Save to CSV
        output_path = os.path.join(output_dir, f"{df_name}_samples.csv")
        samples.to_csv(output_path, index=False)
        print(f"Created sample spreadsheet at {output_path} with {len(samples)} rows")


# Main execution
if __name__ == "__main__":
    create_sample_spreadsheets()
