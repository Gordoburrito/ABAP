import pandas as pd

def format_product_data_to_ABAP(product_df):
    """
    Convert product data to ABAP format using DataFrame operations.
    
    Parameters:
        product_df (DataFrame): DataFrame containing product data.
    
    Returns:
        DataFrame: Transformed product data formatted for ABAP.
    """
    # Rename the columns to match the desired output keys
    result = product_df.rename(columns={
        "title": "Title",
        "models_expanded": "Tag",
        "mpn": "MPN",
        "cost": "Cost",
        "price": "Price",
        "body_html": "Body HTML",
        "collection": "Collection",
        "product_type": "Product Type",
        "meta_title": "Meta Title",
        "meta_description": "Meta Description",
    })

    # Assign constant values
    result["Dropship"] = "always drop ship"
    result["Notes"] = ""

    # Ensure the DataFrame has the expected column order
    columns_order = [
        "Title", "Tag", "MPN", "Cost", "Price", "Dropship",
        "Body HTML", "Collection", "Product Type", "Meta Title",
        "Meta Description", "Notes"
    ]
    result = result[columns_order]

    return result 