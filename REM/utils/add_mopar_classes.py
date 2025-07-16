from extract_golden_df import load_master_ultimate_golden_df
from openai import OpenAI
import pandas as pd
import os
from dotenv import load_dotenv
from typing import Literal
from pydantic import BaseModel

class VehicleClassification(BaseModel):
    vehicle_type: Literal["car", "truck"]

def classify_vehicle_with_ai(year: str, make: str, model: str, client: OpenAI) -> VehicleClassification:
    prompt = f"""Determine if this vehicle is a car or truck.

Vehicle Details:
Year: {year}
Make: {make}
Model: {model}

Rules:
1. Only classify as either "car" or "truck"
2. Trucks include: pickup trucks, commercial trucks, and cargo vans
3. Cars include: sedans, coupes, wagons, passenger vans, and SUVs

Output must be in JSON format with a single field "vehicle_type" that is either "car" or "truck".
"""

    response = client.beta.chat.completions.parse(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "Classify vehicles as either car or truck based on the provided details."},
            {"role": "user", "content": prompt}
        ],
        response_format=VehicleClassification,
        temperature=0.1
    )

    return VehicleClassification.model_validate_json(response.choices[0].message.content)

def add_mopar_classes(golden_df=None):
    # Load environment variables
    load_dotenv()
    
    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # Load the golden dataframe if not provided
    if golden_df is None:
        golden_df = load_master_ultimate_golden_df()
    
    # Create a copy to store results
    result_df = golden_df.copy()
    result_df['vehicle_type'] = 'unknown'
    
    # Process each unique year/make/model combination
    unique_vehicles = golden_df.drop_duplicates(subset=['year', 'make', 'model'])
    total_vehicles = len(unique_vehicles)
    
    for idx, row in unique_vehicles.iterrows():
        print(f"Processing {idx + 1}/{total_vehicles}: {row['year']} {row['make']} {row['model']}")
        
        try:
            classification = classify_vehicle_with_ai(row['year'], row['make'], row['model'], client)
            
            # Update all matching rows in the result dataframe
            mask = (
                (result_df['year'] == row['year']) & 
                (result_df['make'] == row['make']) & 
                (result_df['model'] == row['model'])
            )
            result_df.loc[mask, 'vehicle_type'] = classification.vehicle_type
            
            # Save progress every 10 vehicles
            if (idx + 1) % 10 == 0:
                result_df.to_csv('data/master_ultimate_golden_transformed.csv', index=False)
                print(f"Progress saved at {idx + 1} vehicles")
        
        except Exception as e:
            print(f"Error processing {row['year']} {row['make']} {row['model']}: {str(e)}")
    
    # Save final results
    result_df.to_csv('data/master_ultimate_golden_transformed.csv', index=False)
    print("Classification complete. Results saved to master_ultimate_golden_transformed.csv")
    
    return result_df

if __name__ == "__main__":
    add_mopar_classes()