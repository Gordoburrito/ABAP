from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd
import json
from openai import OpenAI
from pathlib import Path
from datetime import datetime

class ProductSEOOutput(BaseModel):
    meta_title: str = Field(..., max_length=60, description="MAX 57 characters")
    meta_description: str = Field(..., max_length=160, description="MAX 157 characters")
    product_description: str = Field(
        ..., description="HTML formatted product description (NO anchor tags)"
    )

# Define the system prompt
product_seo_prompt = """
You are a professional e-commerce technical content writer and SEO specialist. Your task is to generate optimized content for product pages to inform the customer what the product does. Respond as a trusted mechanic.

For each product, create:
I'm building descriptions for my ecommerce website for vintage auto parts for dodge, Chrysler, Desoto and Plymouth cars. I'll provide you a title of the part, the vehicle it fits, the category of part it is and any engine data (if applicable). I'd like you write me a full product description, a meta title and a meta description (limited to 160 characters) for each product . I'd like you to use the voice of a trusted neighborly mechanic that isn't trying to sell the product, but rather just explain it's usefulness in an informative way. Where possible highlight how to install, why it's useful and the quality of the product. Try to be brief

For the description make it as short as possible without losing any context related to what's included in the parts and instructions to the customer if applicable. If it includes multiple parts, list those in bullet points after the shortened description

Output must be valid JSON in this format:
{
    "meta_title": "string",
    "meta_description": "string",
    "product_description": "string with HTML markup"
}

# Mopar Product Naming & Description Guidelines

## **General Rules**
- **MOPAR** should be written as **Mopar**.
- **Mopar Cars** include all makes: **Chrysler, DeSoto, Dodge, and Plymouth**.
- **Year range** should reflect the earliest and oldest year for that product.
- **Vehicle make** should list **all applicable makes** for the product.

## **Vintage vs. Classic**
- **Vintage:** 1949 or earlier
- **Classic:** 1950 and after
- **Usage Examples:**
  - **Vintage Mopar Car** – before 1949
  - **Classic Mopar Car** – after 1950
  - **Vintage & Classic Mopar Cars** – when covering both time periods

## **Meta Titles**
- **Google typically displays 50-60 characters, but can index up to 70 characters.**
- **Do not exceed 70 characters.**
- **Ensure uniformity in titles.**
- **Titles should start with the product name.**

### **Title Formatting Examples**
✅ **Correct:**
- `Cowl Vent Gasket for 1939 Chrysler, DeSoto, and Plymouth`
❌ **Incorrect:**
- `1939 Cowl Vent Gasket for Chrysler, DeSoto, and Plymouth`

✅ **Correct:**
- `Cowl Vent Gasket Rubber for 1928 - 1961 Vintage Dodge & Mopar`
❌ **Incorrect:**
- `Cowl Vent Gasket Rubber for Vintage Dodge & MOPAR`

✅ **Shortened Title:**
- `Cowl Vent Gasket for 1928 - 1961 Vintage Dodge & Mopar`

✅ **Include Specific Makes:


Make the content engaging, professional, and optimized for both users and search engines.
"""


def create_batch_tasks(df):
    tasks = []

    for index, row in df.iterrows():
        # Clean and format features from available data
        specifications = [
            f"Product Type: {row['Product Type']}" if pd.notna(row['Product Type']) else None,
        ]
        specifications = [spec for spec in specifications if spec is not None]

        # Extract vehicle compatibility from tags
        vehicle_compatibility = row['Tag'].split(', ') if pd.notna(row['Tag']) else []
        engine_compatibility = row['Metafield: custom.engine_types [single_line_text_field]'] if pd.notna(row['Metafield: custom.engine_types [single_line_text_field]']) else []
        include_description = row['AI Description Editor'] == 'x'

        # TODO: add metafield custom_engine_types

        product_context = f"""
        Title: {row['Title']}
        Category: {row['Collection']}
        
        Features:
        - {row['Body HTML'] if pd.notna(row['Body HTML'] and include_description) else 'no description provided'}
        {", ".join(f"{vehicle}" for vehicle in vehicle_compatibility)}
        {f"Engine Fitment: {', '.join(f'{engine}' for engine in engine_compatibility)}" if engine_compatibility else ''}
        """
        print("--------------------------------")
        print("product_seo_prompt")
        print(product_seo_prompt)
        print("--------------------------------")
        print("product_context")
        print(product_context)
        print("--------------------------------")

        task = {
            "custom_id": f"task-{index}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o",
                "temperature": 0.7,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": product_seo_prompt},
                    {"role": "user", "content": product_context},
                ],
            },
        }
        tasks.append(task)

    return tasks


def process_batch():
    # Load product data
    df = pd.read_csv("src/ABAP/meta_info/data/raw/ABAP - MASTER IMPORT FILE v3.csv")

    # Create batch tasks
    tasks = create_batch_tasks(df)

    # Ensure the data directory exists
    Path("data").mkdir(exist_ok=True)

    # Save tasks to JSONL file
    file_name = "data/batch_tasks_products.jsonl"
    with open(file_name, "w") as file:
        for task in tasks:
            file.write(json.dumps(task) + "\n")

    # Initialize OpenAI client
    client = OpenAI()

    # Upload file for batch processing
    batch_file = client.files.create(file=open(file_name, "rb"), purpose="batch")

    # Create batch job
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    return batch_job.id


def process_results(batch_job_id):
    client = OpenAI()
    
    # Load original dataframe
    df = pd.read_csv("src/ABAP/meta_info/data/raw/ABAP - MASTER IMPORT FILE v3.csv")
    df = df.reset_index().rename(columns={'index': 'original_index'})  # Keep track of original indices
    
    # Retrieve batch job
    batch_job = client.batches.retrieve(batch_job_id)

    # Get results file
    result_file_id = batch_job.output_file_id
    result = client.files.content(result_file_id).content

    # Save results
    result_file_name = "data/batch_job_results_products.jsonl"
    with open(result_file_name, "wb") as file:
        file.write(result)

    # Process results and create a results dataframe
    processed_results = []
    with open(result_file_name, "r") as file:
        for line_number, line in enumerate(file, 1):
            try:
                result = json.loads(line.strip())
                task_id = result["custom_id"]
                # Extract the index from task-{index} format
                original_index = int(task_id.split('-')[1])
                
                try:
                    content = json.loads(
                        result["response"]["body"]["choices"][0]["message"]["content"]
                    )
                except json.JSONDecodeError as json_error:
                    print(f"Error parsing content JSON for task {task_id} (line {line_number}): {json_error}")
                    # Add the raw content as fallback
                    raw_content = result["response"]["body"]["choices"][0]["message"]["content"]
                    print(f"Raw content: {raw_content[:100]}...")  # Print first 100 chars for debugging
                    processed_results.append({
                        "original_index": original_index,
                        "validation_error": f"JSON parse error: {json_error}",
                        "raw_content": raw_content
                    })
                    continue

                try:
                    validated_content = ProductSEOOutput(**content)
                    processed_results.append({
                        "original_index": original_index,
                        **validated_content.model_dump()
                    })
                except Exception as validation_error:
                    print(f"Validation error for task {task_id}: {validation_error}")
                    processed_results.append({
                        "original_index": original_index,
                        **content,
                        "validation_error": str(validation_error)
                    })
            except json.JSONDecodeError as outer_json_error:
                print(f"Error parsing response JSON at line {line_number}: {outer_json_error}")
                print(f"Problematic line: {line[:100]}...")  # Print first 100 chars for debugging
                # Add a placeholder in results to maintain indexing
                processed_results.append({
                    "original_index": f"error_line_{line_number}",
                    "validation_error": f"Outer JSON parse error: {outer_json_error}",
                    "raw_line": line
                })
                continue

    # Create DataFrame from results and merge with original
    results_df = pd.DataFrame(processed_results)
    
    # Filter out error entries before merging
    valid_results_df = results_df[results_df["original_index"].apply(lambda x: isinstance(x, int))]
    
    merged_df = pd.merge(
        df,
        valid_results_df,
        on='original_index',
        how='left',
        suffixes=('', '_generated')
    )
    
    return merged_df


def save_results_to_csv(results_df):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_df.to_csv(
        f"src/ABAP/meta_info/data/processed/processed_products_{timestamp}.csv", 
        index=False
    )
