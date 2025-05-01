import openai
import os
import json
import time
from datetime import datetime

# --- Configuration ---
# Ensure your OPENAI_API_KEY is set in your environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Error: OPENAI_API_KEY environment variable not set.")
    exit()

client = openai.OpenAI(api_key=api_key)
# BATCH_ID = "batch_68132475ee8481908e4db8b9c4605c0c" # No longer needed for listing all

# --- Continuously Monitor All Active Batches ---
print(
    f"--- Continuously checking status for ALL active Batch Jobs every 60 seconds (Press Ctrl+C to stop) ---"
)

finished_statuses = ["completed", "failed", "cancelled", "expired"]

try:
    while True:
        now_str = datetime.now().isoformat()
        active_batches_found_this_cycle = 0
        print(f"\n--- [{now_str}] Polling active batches ---")
        try:
            batches = client.batches.list(limit=100)  # List recent batches

            for batch in batches:
                status = batch.status
                batch_id = batch.id

                if status not in finished_statuses:
                    active_batches_found_this_cycle += 1
                    completed = batch.request_counts.completed
                    total = batch.request_counts.total
                    failed = batch.request_counts.failed

                    # Single line output format
                    print(
                        f"  [{now_str}] ID: {batch_id} | Status: {status:<12} | Requests: {completed}/{total} ({failed} failed)"
                    )

            if active_batches_found_this_cycle == 0:
                print(f"  [{now_str}] No active batch jobs found this cycle.")

        except Exception as e:
            print(
                f"  [{now_str}] An error occurred while listing/checking batches: {e}"
            )

        # Wait for 60 seconds before the next check
        time.sleep(60)

except KeyboardInterrupt:
    print("\n--- Monitoring stopped by user. ---")

print("--- Script Finished ---")

# --- Old monitoring loop code removed ---
# print(f"--- Continuously checking status for Batch ID: {BATCH_ID} every 60 seconds (Press Ctrl+C to stop) ---")
# try:
#     while True:
#         try:
#             now_str = datetime.now().isoformat()
#             print(f"\n--- [{now_str}] Checking status --- ")
#             # ... rest of loop ...
#         except Exception as e:
#             # ... error handling ...
#         time.sleep(60)
# except KeyboardInterrupt:
#     print("\n--- Monitoring stopped by user. ---")
