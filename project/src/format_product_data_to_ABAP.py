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
        model = row["model"]
        
        # Get all valid years in range
        years = get_year_range(year_min, year_max)
        car_ids = []
        for year in years:
            if model == "ALL":
                # Expand ALL to include all models for that year/make
                models = expand_all_models(year, make, golden_df)
                print("models")
                print(models)
                for m in models:
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
            "Body HTML": row["body_html"]
            # Add other required ABAP columns...
        }
        
        results.append(abap_row)
    
    return pd.DataFrame(results) 