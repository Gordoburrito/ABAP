import pandas as pd

def get_year_range(year_min, year_max, golden_df):
    """Get all valid years between min and max from golden data"""
    valid_years = golden_df[
        (golden_df["Year Min"].between(year_min, year_max)) |
        (golden_df["Year Max"].between(year_min, year_max))
    ]["Year"].unique()
    return sorted([y for y in valid_years if year_min <= y <= year_max])

def expand_all_models(year, make, golden_df):
    """Get all models for a given year and make"""
    models = golden_df[
        (golden_df["Year"] == year) &
        (golden_df["Make|Include"] == make)
    ]["Model|Include"].unique()
    return models

def transform_product_data_to_ABAP(product_df, golden_df):
    """Transform product data to ABAP format"""
    results = []
    
    for _, row in product_df.iterrows():
        year_min = row["year_min"]
        year_max = row["year_max"]
        make = row["make"]
        model = row["model"]
        
        # Get all valid years in range
        years = get_year_range(year_min, year_max, golden_df)
        
        car_ids = []
        for year in years:
            if model == "ALL":
                # Expand ALL to include all models for that year/make
                models = expand_all_models(year, make, golden_df)
                for m in models:
                    car_ids.append(f"{year}_{make}_{m}")
            else:
                car_ids.append(f"{year}_{make}_{model}")
        
        # Create ABAP format row
        abap_row = {
            "Title": row["title"],
            "(Internal) Car ID - Expanded": ", ".join(car_ids),
            "MPN": row["mpn"],
            "Cost": row["cost"],
            "Price": row["price"],
            "Body HTML": row["body_html"]
            # Add other required ABAP columns...
        }
        
        results.append(abap_row)
    
    return pd.DataFrame(results) 