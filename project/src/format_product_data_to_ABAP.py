import pandas as pd

def get_year_range(year_min, year_max):
    """Get all valid years between min and max from golden data"""
    year_min = int(year_min)
    year_max = int(year_max)
    return [str(year) for year in range(year_min, year_max + 1)]

def expand_all_models(year, make, golden_df):
    """Get all models for a given year and make"""
    models = golden_df[
        (golden_df["year"] == year) & 
        (golden_df["make"] == make)
    ]["model"].unique()
    return models

def format_product_data_to_ABAP(product_df, golden_df):
    """Transform product data to ABAP format"""
    results = []
    
    for _, row in product_df.iterrows():
        year_min = row["year_min"]
        year_max = row["year_max"]
        make = row["make"]
        # Handle NaN or float values in models_expanded
        if pd.isna(row["models_expanded"]):
            models = []
        else:
            models = str(row["models_expanded"]).strip('[]').replace("'", "").split(",")
        # Clean up any whitespace from each model
        models = [model.strip() for model in models]
        
        # Get all valid years in range
        years = get_year_range(year_min, year_max)
        car_ids = []
        for year in years:
            for model in models:  # Now iterating through the list of models
                if model == "ALL":
                    # Expand ALL to include all models for that year/make
                    expanded_models = expand_all_models(year, make, golden_df)
                    print("models")
                    print(expanded_models)
                    for m in expanded_models:
                        car_ids.append(f"{year}_{make}_{m}")
                else:
                    car_ids.append(f"{year}_{make}_{model}")
        
        # Create ABAP format row
        abap_row = {
            "Title": row["title"],
            "Tag": ", ".join(car_ids),
            "MPN": row["mpn"],
            "Cost": row["cost"],
            "Price": row["price"],
            "Dropship": "always drop ship",
            "Body HTML": row["body_html"],
            "Collection": row["collection"],
            "Product Type": row["product_type"],
            "Meta Title": row["meta_title"],
            "Meta Description": row["meta_description"],
            "Notes": ""
        }
        
        results.append(abap_row)
    
    return pd.DataFrame(results) 