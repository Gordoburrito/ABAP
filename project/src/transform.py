import pandas as pd
from typing import Optional
from openai import OpenAI
import os
from dotenv import load_dotenv
from models.product_data import ProductData

def extract_product_data_with_ai(row: pd.Series, golden_df: pd.DataFrame, client: OpenAI) -> ProductData:
    """
    Convert a single vendor row to ProductData format, using OpenAI for missing information
    """
    # Get valid options from golden copy, filtering out NaN values
    valid_makes = sorted([x for x in golden_df['make'].unique() if pd.notna(x)])
    valid_models = sorted([x for x in golden_df['model'].unique() if pd.notna(x)])
    valid_years = [str(year) for year in range(1925, 2025)]
    valid_collections = ['Accessories', 'Body & Glass', 'Clutch', 'Cooling', 'Electrical', 'Engine', 'Fuel', 'Weatherstripping', 'Exterior Rubber', 'Interior Rubber and Carpets', 'Front Axle', 'Wheels', 'Rear Axle', 'Service Brakes', 'Universal Joint', 'Springs', 'Transmission', 'Literature']
    valid_product_types = ['Windshield','Weatherstrip Adhesive Black','Vent Window Rubber','Cowl Vent Gasket','Cowl Vent Gasket Rubber Kit','Windshield Rubber','Windshield And Center Bar Rubber','Rear Window Rubber','Quarter Window Rubber','Quarter Window Rubber For Coupes','Quarter Window Rubber With Non- Opening Quarter Window','Door Weatherstrip','Front Door Hinge Post Weatherstrip Set','Door Top Rubber','Door Bottom Rubber','Door Bottom Rubber Weatherstrip','Rumble Seat Lid Rubber','Trunk Rubber','Anti-Squeak Material','Fender Welt','Stainless Chrome Beaded Fenderwelt','Soft Top Insert Rubber','Top Side Rubber','Window Belt Line Sweeper Material','Generator And Starter Motor Rebuild Service','Window Belt Line Sweeper Clips','Window Run Channel','Window Run Channel Clips','Window Belt Line Sweeper','Window Guide','Cowl Lacing','Hood','Fender Filler','Vent Window Vertical Channel','Vent Window Division Bar L-Shaped Fuzzy','Trunk Weatherstrip','Roofrail Weatherstrip','Top Header Rubber','Top Header Rubber Set','Quarter Window Divider Rubber','Cowl To Hood Seal','Radiator To Hood Seal','Rear Window Gasket','Vent Window Gasket Front','Front Vent Window Gaskets','Front Vent Window Rubber','Windshield Gasket!','Rear Window Rubber Gasket','Front Vent Window Gasket','Vent Window Gasket','Windshield Gasket','Moulded Windshield Rubber Gasket','Beltstrip "Sweeper" Set','Door Seal/Windlace','Windshield Seal!','Beltline Molding Kit','Windshield Rubber Gasket','Door Weatherstripping','Windshield Lower Chrome Lockstrip','Windshield Chrome Lockstrip Set','Rear Window Chrome Trim','Pop Out Quarter Window Rubber','Vent Window Rubber Set','Door Seal Master Kit','Door Weatherstrip Kit','Beltline "Sweeper" Moulding Kit','Beltline "Sweeper" Moulding And Window Channel Kit','Rubber Window Lockstrip','Chrome Lockstrip','Door Seal Kit','Side Door Seal Kit','Rear Doors Seal Kit','Cab To Fender Rubber Seal','Door Window Glass Run Channel','Rear Of Vent Window Glass Run Channel','Beltline "Sweeper" And Glass Run Channel Set','Quarter Window Rubber Seals','Scamp Rear Window Rubber','Beltline "Sweeper" Weatherstrip Set','"B Body" Beltline "Sweeper" Weatherstrip Set','"B Body" Beltline "Sweeper" Set','Beltstrip "Sweeper" Weatherstrip Set!','Station Wagon Beltstrip "Sweeper" Weatherstrip Set','Beltstrip "Sweeper" Weatherstrip Set','"B-Body" Beltstrip "Sweeper" Weatherstrip Set','Beltline "Sweeper" Set','Roofrail Weatherstrip Set','Moulded Door Weatherstrip Set','Door Weatherstrip End Caps With Metal Core','U Jamb Door Lock Pillar Filler Seal','Trunk Weatherstrip Kit','Rear Side Window/ Quarter Window Weatherstrip Set','Windshield And Rear Window Trim Clip Set Set','Quarter Window Seal','Front Door Vent Window Gasket','Top Weatherstrip Set!','Header Seal','Roof Rail Set','Pillar Post/ U Jamb Seals','Vent Window Gaskets','Trunk Weatherstrip Seal','Top Seal Set','Roof Rail Weatherstrip','Front Door Weatherstrip','Rear Door Weatherstrip','Roofrail Set','Door Vent Window Gaskets','Weatherstripping Seal Set','U Jamb Lock Pillar Filler','Door Weatherstrip Set','Door Weatherstrip End Cap Fillers','Door End Cap Filler Seals/ U Jams','Roof Rail Weathertrip Set','Door Seal Set','Front Doors Seal Set','Rear Doors Seal Set','Quarter Window Pillar-Post Seals','Convertible Roof Rail Weathertrip Set','Trunk Seal','Roofrail Seal Set','Door End Filler Cap/ Seals','Basic Weatherstripping Kit','Roof Rail Weathrstripping Set','Moulded Door Seals','Door Seal End Caps','Rear Roof Rail Seals','Top Seal Set!','Door Vent Window Gasket','Van Fixed And Rear Door Window Rubber','Van Large Fixed Side Window Rubber','Moulded Windshield Rubber','Moulded Rear Window Rubber','Door Weatherstrip Seals','Rear Bumper Arm Grommets','Moulded Windshield Gasket','Radiator To Hood Rubber','Rear Door Vent Window Gasket Set','Rear Vent Window Gasket','Rear Quarter Window Gasket','Stationary Quarter Window Rubber','Moulded 8 Piece Roofrail Set','Swing Out Windshield To Body And Windshield Pillars Stationary Seal Set!','Swing Out Windshield Molded Seal Set!','Tail Light Pads','Headlight Pads','Gas Filler Neck Grommet','Bumper Arm Grommets','Door Handle Pad','Door Handle Pads','Door Handle Pad Set','Door Handle / Trunk Handle Pad','Hood Corners','License Lamp Pad','Headlight','Headlight Inner Pads','Trunk Hinge Pads','Tail Lamp Pads','License Light Housing Pad','Door Bumper','License Plate Mounting Bracket Pad','Body Mounts','Headlight Thick Base Pads','Horn Vent Pad','License Lamp Housing Base Pad','Trunk Handle Base Pad','Wiper Base Pivot Pads','Brake Light Base Pad','License Lens Gasket','Parking Light Base Pad','Park Light Lens Gasket Set','Tail Light Lens Gasket Set','Headlight To Fender Pads','Brake Light Lens Gasket','Headlight Lens Gasket','Hood Side Bumper','Parking Light Lens Gasket','Tail Light Lens Gasket','Back-Up Light Pads','Cowl Light Mounting Pads','Stop & License Light Housing Base Pad','Bumper Arm Grommets Rear','Bumper Arm Grommet','Rumble Seat Step Pad','Tail Light Pad','Running Board Rubber','Tail Light Lens Gaskets','Trunk Bumper','Front Bumper Grommets','Rear Fender Gravel Shields, Pair','Running Board Rubber Matting','Headlight Bucket To Fender Gasket','Dimmer Switch Grommet','Door Check Link Bumper','Door Sill Step Mats','Starter Pedal Top Pad','Starter Pedal Top','Starter Pedal Shaft Draft Seal','Steering Column To-Dash Pad','Clutch And Brake Pedal Draft Seal','Door Sill Mat Step Mat Grommets','Firewall Grommet Set','Headliner Bow Grommet','Hood Bumpers','Clutch And Brake Pedal','Gas Pedal Stem Grommet','Cowl Vent Gasket 1-Piece','Gearshift Dust Cover','Starter Pedal Shaft Seal','Clutch And Brake Pedal Accessory Slip-On Pad','Steering Column Post Floor Pad','Clutch Pedal','Brake Pedal','Steering Column To Floor Pad','Trunk Matt','Trunk Mat','Steering Column And Gearshift Boot','Floor Mat','Gearshift Boot Black','Gearshift Boot','Gearshift Boot Brown','Gas Pedal','Rear Package Tray Replacement','Cowl Vent Grommet','Brake Pedal Pad','Clutch And Brake Pedal Pad','Parking Brake Pedal Pad','Parking Brake Pedal Pad!','Accelerator Pedal!','Fuel Filler Neck Grommet','Clutch Or Brake Pedal Pad','Black Steering Column Grommet','Handbrake Floor Plate','Brown Steering Column Grommet','Handbrake Floor Plate!','Reproduction Door Check Straps','Starter Pedal Rod Shaft Boot','Reproduction Carpet Heel Plate','Reproduction Carpet Heel Plate!','Carpet Set','"B-Body" Carpet Set','Catalog','Refundable Core Charge','Control Arms Master Rebuild Kit','Drag Link Service Package','Drag Link Dust Cover Package','Drag Link End Assembly','King Pins Package','Tie Rod End','Tie Rod End Outer Left','Tie Tod End','Outer Right Tie Rod End','Outer Left Tie Rod End','Drag Link Assembly','Idler Arm','Tie Rod End Boot','Control Arm, Inner Shaft, And Bushings Package','Upper Control Arm Inner Shaft And Bushing Package','Control Arm, Inner Shaft, And Pins Package','Upper Control Arm Bumpers','Upper Control Arm Bumper','Control Arm Bumper','Control Arm Bumper, Upper','Rebound Bumper','Control Arm, Inner Shaft, Pins, And Bushings Package','Ball Joint Package Upper','Ball Joint Package Lower','Control Arm Bushing','Strut Arm Bushing','Sway Bar Bushing','Sway Bar Hanger Bushing','Center Link','Tie Rod Adjusting Sleeve Or Tube','Idler Arm Bushing','Front End Rebuild Kit']

    # Create a clean product info string from available row data
    product_info = "\n".join([f"{key}: {value}" for key, value in row.items() if pd.notna(value)])

    prompt = f"""Based on this product information, determine the car details and create a product description.
    
    Product Information:
    {product_info}
    
    Valid Options:
    Makes: {valid_makes}
    Models: {valid_models}
    Years: {valid_years}
    Collections: {valid_collections}
    Product Type: {valid_product_types}

    Compatibility Rules:
    1. Universal Compatibility:
       - Use "ALL" if a component is compatible with all options in any category
       - Example: A generic rubber grommet might be "ALL" for makes, models, and years
       - Body type specifications override "ALL" for models

    2. Body Type Handling:
       - If "2 Door" or "2-Door" is specified: Select only 2-door models or use "ALL (2-Door)"
       - If "4 Door" or "4-Door" is specified: Select only 4-door models or use "ALL (4-Door)"
       - If "Truck" is specified: Select only truck models or use "ALL (Truck)"
       - If "Non-Truck" or similar is specified: Select only car/sedan models or use "ALL (Non-Truck)"
       - If "A-Body" is specified: Select only A-Body models or use "ALL (A-Body)"
       - If "B-Body" is specified: Select only B-Body models or use "ALL (B-Body)"
       - If "C-Body" is specified: Select only C-Body models or use "ALL (C-Body)"
       - If no body type specified: Use "ALL" if universally compatible

    3. Year Processing:
       - For ranges (e.g., "34/64"): Convert to full years (1934-1964)
       - For single years (e.g., "62"): Convert to full year (1962)
       - Use exact years when specified, don't invent ranges
       
    4. Invalid Options:
       - Use "NONE" if no valid option matches the category
       - Must choose only from provided valid options lists

    Output must be in JSON format matching the example structure.
    """

    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Extract the product information based on the valid options provided. Use 'ALL' for universal compatibility."},
            {"role": "user", "content": prompt}
        ],
        response_format=ProductData,
        temperature=0.5
    )

    product_data = response.choices[0].message.parsed

    return product_data


def get_refined_models_from_ai(
    title: str, year_min: int, year_max: int, make: str, models_str: str, valid_models: list, client: OpenAI
) -> str:
    """
    Use AI to refine the models list based on body type rules.
    """
    prompt = f"""Given a part that fits:
    Title: {title}
    Year Range: {year_min}-{year_max}
    Make: {make}
    Model Input: {models_str}

    Available Models: {valid_models}

    Task: Determine which models this part fits based on these rules:
    1. For "ALL (4-Door)" specification:
       - Include models that were available as 4-door variants
       - Example: 1939 Plymouth P7 Road King was available as 4-door sedan
       - Don't rely only on model names containing "4-door"
       - Consider historical model configurations
    
    2. For "ALL (2-Door)" specification:
       - Include models that were available as 2-door variants
       - Don't rely only on model names containing "2-door"
       - Consider historical model configurations

    3. Similar rules apply for other body types:
       - "ALL" → Part fits all valid models for the make/year
       - "ALL (2-Door)" → Part fits models available in 2-door configuration
       - "ALL (Truck)" → Part fits truck models
       - "ALL (Non-Truck)" → Part fits passenger car models
       - "ALL (A-Body)" → Part fits A-Body chassis models
       - "ALL (B-Body)" → Part fits B-Body chassis models
       - "ALL (C-Body)" → Part fits C-Body chassis models

    3. For specific model lists:
       - Include only models that this part is confirmed to fit
       - Each model must exist in the valid_models list
       - Consider all body style variants when specified

    Return a comma-separated list of models that this part fits.
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a vehicle model validator. Return only the valid models as a comma-separated list.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )

    # Extract the comma-separated list from the response
    return response.choices[0].message.content.strip()


def refine_models_with_ai(df: pd.DataFrame, golden_df: pd.DataFrame, client: OpenAI) -> pd.DataFrame:
    """
    Second pass over the Models column to ensure correct model selection based on body type rules.
    """
    result_df = df.copy()
    
    # Add new column for expanded models
    result_df['models_expanded'] = result_df['model']
    
    for idx in range(len(result_df)):
        row = result_df.iloc[idx]
        models_str = str(row['model'])  # Convert to string to handle the comparison
        
        # Skip if no models or NONE
        if models_str == 'nan' or models_str == 'NONE':
            continue
            
        # Get valid models for this row's year range and make
        valid_models = golden_df[
            (golden_df["year"].fillna(0).astype(int) >= int(row["year_min"])) & 
            (golden_df["year"].fillna(0).astype(int) <= int(row["year_max"])) & 
            (golden_df["make"] == row["make"])
        ]["model"].unique()
        
        # If it's ALL, just use all valid models
        if models_str == "ALL":
            result_df.at[idx, 'models_expanded'] = ",".join(valid_models)
        else:
            # Get refined models from AI for specific cases
            refined_models = get_refined_models_from_ai(row["title"], int(row["year_min"]), int(row["year_max"]), row["make"], models_str, valid_models, client)
            result_df.at[idx, 'models_expanded'] = refined_models

    return result_df

def transform_data_with_ai(golden_df: pd.DataFrame, vendor_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform vendor data to required format, validating against golden copy.
    """
    # Load environment variables
    load_dotenv()
    
    # Initialize OpenAI client with API key from environment
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # Transform each row
    transformed_records = []
    for index, vendor_row in vendor_df.iterrows():
        print(f"Processing row {index}:")
        transformed_record = extract_product_data_with_ai(vendor_row, golden_df, client)
        transformed_records.append(transformed_record.model_dump())
    
    # Create DataFrame with specific column order
    result_df = pd.DataFrame(transformed_records)
    
    # Add models_expanded column before reordering
    result_df['models_expanded'] = result_df['model']
    
    result_df = result_df[[
        'title',
        'year_min',
        'year_max',
        'make',
        'model',
        'models_expanded',
        'mpn',
        'cost',
        'price',
        'body_html',
        'collection',
        'product_type',
        'meta_title',
        'meta_description',
    ]]
    
    # Second pass to refine models
    result_df = refine_models_with_ai(result_df, golden_df, client)
    
    return result_df 
