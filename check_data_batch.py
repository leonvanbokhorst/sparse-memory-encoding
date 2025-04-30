import openai
import os
import json

# --- Configuration ---
# Ensure your OPENAI_API_KEY is set in your environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Error: OPENAI_API_KEY environment variable not set.")
    exit()

client = openai.OpenAI(api_key=api_key)
BATCH_ID = "batch_6811d87aaba8819096c10fc203b2c61c"  # The ID from your run

batches = client.batches.list()

for batch in batches:
    print(batch.model_dump_json(indent=2))


# --- Check Batch Status ---
print(f"--- Checking status for Batch ID: {BATCH_ID} ---")
try:
    batch_status = client.batches.retrieve(BATCH_ID)

    status = batch_status.status
    completed = batch_status.request_counts.completed
    total = batch_status.request_counts.total
    failed = batch_status.request_counts.failed
    output_file_id = batch_status.output_file_id
    error_file_id = batch_status.error_file_id

    print(f"  Status: {status}")
    print(f"  Requests: {completed}/{total} completed, {failed} failed.")

    if output_file_id:
        print(f"  Output File ID: {output_file_id}")
    if error_file_id:
        print(f"  Error File ID: {error_file_id}")

    if status == "completed":
        print(
            "\nBatch job is complete! You can now download the results using the Output File ID."
        )
    elif status in ["failed", "cancelled", "expired"]:
        print("\nBatch job finished with a non-completed status.")
    else:
        print("\nBatch job is still in progress.")

except Exception as e:
    print(f"An error occurred while checking batch status: {e}")
