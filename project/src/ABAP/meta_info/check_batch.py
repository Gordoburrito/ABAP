import sys
import os
from batch_processor import process_results, save_results_to_csv
from openai import Client

def check_batch_status(batch_job_id):
    client = Client()
    
    try:
        batch_job = client.batches.retrieve(batch_job_id)
        status = batch_job.status
        
        print(f"Status for batch {batch_job_id}: {status}")
        
        if status == 'completed':
            print("Processing results...")
            results = process_results(batch_job_id)
            
            print("Saving results to CSV...")
            save_results_to_csv(results)
            
            print("Process completed successfully!")
        elif status in ['failed', 'cancelled']:
            print(f"Batch job {status}")
        else:
            print("Batch job still in progress")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    # If no batch_job_id provided, try to read from file

    batch_job_id = sys.argv[1]
        
    check_batch_status(batch_job_id) 