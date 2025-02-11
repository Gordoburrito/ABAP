from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd
import json
from openai import OpenAI
from pathlib import Path

class ProductSEOOutput(BaseModel):
    meta_title: str = Field(..., max_length=60)
    meta_description: str = Field(..., max_length=160)
    product_description: str = Field(
        ..., description="HTML formatted product description"
    )

# Define the system prompt
product_seo_prompt = """
You are a professional e-commerce content writer and SEO specialist. Your task is to generate optimized content for product pages.

For each product, create:
1. A meta title (max 60 characters) that is compelling and includes key product features
2. A meta description (exactly 160 characters) that drives clicks and summarizes key benefits
3. A beautifully formatted product description using HTML that includes:
   - Opening paragraph highlighting key benefits
   - Bullet points of features
   - Technical specifications where relevant
   - Call to action

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
        # Combine relevant product data into a context string
        product_context = f"""
        Product Name: {row['product_name']}
        Category: {row['category']}
        Price: {row['price']}
        Features: {row['features']}
        Specifications: {row['specifications']}
        """

        task = {
            "custom_id": f"task-{index}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "temperature": 0.7,
                "response_format": ProductSEOOutput,
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
    df = pd.read_csv("data/product_data.csv")

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

    # Retrieve batch job
    batch_job = client.batches.retrieve(batch_job_id)

    # Get results file
    result_file_id = batch_job.output_file_id
    result = client.files.content(result_file_id).content

    # Save results
    result_file_name = "data/batch_job_results_products.jsonl"
    with open(result_file_name, "wb") as file:
        file.write(result)

    # Process results and validate with Pydantic
    processed_results = []
    with open(result_file_name, "r") as file:
        for line in file:
            result = json.loads(line.strip())
            task_id = result["custom_id"]
            content = json.loads(
                result["response"]["body"]["choices"][0]["message"]["content"]
            )

            # Validate with Pydantic model
            validated_content = ProductSEOOutput(**content)

            processed_results.append(
                {"task_id": task_id, **validated_content.model_dump()}
            )

    return processed_results


def save_results_to_csv(results):
    df = pd.DataFrame(results)
    df.to_csv("data/processed_products.csv", index=False)
