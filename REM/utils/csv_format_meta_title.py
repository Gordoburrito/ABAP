import pandas as pd
import os
from post_format_meta_title import format_meta_title, calculate_pixel_width

def format_csv_meta_titles(input_path, output_path):
    """
    Read a CSV file, format each meta_title field, and save the results to a new CSV file.
    """
    # Read the CSV file
    df = pd.read_csv(input_path)
    
    # Create columns for the formatted titles and transformations
    df['formatted_meta_title'] = ""
    df['pixel_width_original'] = 0
    df['pixel_width_formatted'] = 0
    df['transformations'] = ""
    
    # Process each row
    for index, row in df.iterrows():
        if pd.notna(row['meta_title']):
            # Apply the formatting function
            formatted_title, transformations = format_meta_title(row['meta_title'])
            
            # Calculate pixel widths
            pixel_width_original = calculate_pixel_width(row['meta_title'])
            pixel_width_formatted = calculate_pixel_width(formatted_title)
            
            # Update the DataFrame
            df.at[index, 'formatted_meta_title'] = formatted_title
            df.at[index, 'pixel_width_original'] = pixel_width_original
            df.at[index, 'pixel_width_formatted'] = pixel_width_formatted
            df.at[index, 'transformations'] = "; ".join(transformations)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the result to a new CSV file
    df.to_csv(output_path, index=False)

def main():
    # Determine the paths
    input_path = os.path.join('meta_info', 'data', 'processed', 'meta_title_and_meta_description', '1-4o.csv')
    output_path = os.path.join('meta_info', 'data', 'results', 'formatted_meta_titles.csv')
    
    # Format the meta titles and save the results
    format_csv_meta_titles(input_path, output_path)

if __name__ == "__main__":
    main()
