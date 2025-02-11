from batch_processor import process_batch, process_results, save_results_to_csv
import time


def main():
    try:
        print("Starting batch processing...")
        batch_job_id = process_batch()
        print(f"Batch job created with ID: {batch_job_id}")

        print("Waiting for batch processing to complete...")
        # You might want to implement a proper polling mechanism here
        time.sleep(300)  # Wait 5 minutes for demo purposes

        print("Processing results...")
        results = process_results(batch_job_id)

        print("Saving results to CSV...")
        save_results_to_csv(results)

        print("Process completed successfully!")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
