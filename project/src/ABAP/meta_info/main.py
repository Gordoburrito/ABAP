from batch_processor import process_batch, process_results, save_results_to_csv
import time
import os
from openai import OpenAI

client = OpenAI()

def main():
    try:
        print("Starting batch processing...")
        batch_job_id = process_batch()
        print(f"Batch job created with ID: {batch_job_id}")
        
        # Save batch ID with timestamp in filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"src/ABAP/meta_info/batch_ids/batch_id_{timestamp}.txt"
        os.makedirs(os.path.dirname(filename), exist_ok=True)  # Create directory if it doesn't exist
        with open(filename, 'w') as f:
            f.write(batch_job_id)
        print(f"Batch ID saved to {filename}")

        print("Checking batch processing status...")
        while True:
            batch_job = client.batches.retrieve(batch_job_id)
            status = batch_job.status
            
            print(f"Current status: {status}")
            
            if status == 'completed':
                break
            elif status in ['failed', 'cancelled']:
                raise Exception(f"Batch job {status}")
            
            # Check every 5 minutes
            time.sleep(300)

        print("Processing results...")
        results = process_results(batch_job_id)
        
        print("Saving results to CSV...")
        save_results_to_csv(results)

        print("Process completed successfully!")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
