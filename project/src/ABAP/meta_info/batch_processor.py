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
You are a professional e-commerce content writer and SEO specialist. Your task is to generate optimized content for product pages.

For each product, create:
I'm building descriptions for my ecommerce website that sells vintage auto parts for dodge, Chrysler, Desoto and Plymouth cars. I'll provide you a title of the part, the vehicle it fits, the category of part it is and any engine data (if applicable). I'd like you write me a full product description, a meta title (max 60 characters) and a meta description (limited to 160 characters) for each product

Output must be valid JSON in this format:
{
    "meta_title": "string",
    "meta_description": "string",
    "product_description": "string with HTML markup"
}

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

        product_context = f"""
        Title: {row['Title']}
        Category: {row['Collection']}
        
        Features:
        - {row['Body HTML'] if pd.notna(row['Body HTML']) else 'no description provided'}
        
        Fitment:
        {", ".join(f"{vehicle}" for vehicle in vehicle_compatibility)}
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
                "model": "gpt-4-turbo-preview",
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
    df = pd.read_csv("src/ABAP/meta_info/data/SAMPLE ABAP - MASTER IMPORT FILE.csv")

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
    df = pd.read_csv("src/ABAP/meta_info/data/SAMPLE ABAP - MASTER IMPORT FILE.csv")
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
        for line in file:
            result = json.loads(line.strip())
            task_id = result["custom_id"]
            # Extract the index from task-{index} format
            original_index = int(task_id.split('-')[1])
            content = json.loads(
                result["response"]["body"]["choices"][0]["message"]["content"]
            )

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

    # Create DataFrame from results and merge with original
    results_df = pd.DataFrame(processed_results)
    merged_df = pd.merge(
        df,
        results_df,
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
