import openai
import os
import json
from datetime import datetime

# --- Configuration ---
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Error: OPENAI_API_KEY environment variable not set.")
    exit()

client = openai.OpenAI(api_key=api_key)

OUTPUT_DIRECTORY = "processed_batches"
BATCH_LIST_LIMIT = 100  # How many recent batches to check

# Ensure output directory exists
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

# --- Get Completed Batches ---
print(f"--- Checking for completed batch jobs (limit {BATCH_LIST_LIMIT}) ---")
completed_batches = []
try:
    batches = client.batches.list(limit=BATCH_LIST_LIMIT)
    for batch in batches:
        if batch.status == "completed":
            completed_batches.append(batch)
    print(f"Found {len(completed_batches)} completed batch jobs.")
except Exception as e:
    print(f"Error listing batches: {e}")
    exit()

if not completed_batches:
    print("No completed batches found to process.")
    exit()

# --- Process Each Completed Batch ---
total_processed = 0
total_skipped = 0
for batch in completed_batches:
    batch_id = batch.id
    output_file_id = batch.output_file_id
    error_file_id = batch.error_file_id  # Good to know if errors occurred

    print(f"\n--- Processing Batch ID: {batch_id} ---")

    if not output_file_id:
        print(f"Skipping Batch {batch_id}: No output file ID found.")
        total_skipped += 1
        continue

    # Define target filename
    target_filename = os.path.join(OUTPUT_DIRECTORY, f"results_{batch_id}.json")

    # Check if file already exists
    if os.path.exists(target_filename):
        print(
            f"Skipping Batch {batch_id}: Output file '{target_filename}' already exists."
        )
        total_skipped += 1
        continue

    print(f"Target output file: {target_filename}")
    if error_file_id:
        print(
            f"Note: This batch had an error file associated ({error_file_id}). Results might be incomplete."
        )

    # --- Retrieve & Parse Results for this batch ---
    results_data = []
    try:
        print(f"Downloading results from Output File ID: {output_file_id}...")
        output_content_response = client.files.content(output_file_id)
        output_content = output_content_response.read().decode("utf-8")
        print("Successfully downloaded.")

        # Parse the JSONL content
        lines = output_content.strip().splitlines()
        print(f"Parsing {len(lines)} lines from output file...")
        successful_parses = 0
        for line_num, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                result_item = json.loads(line)
                custom_id = result_item.get("custom_id")
                response_body = result_item.get("response", {}).get("body", {})
                choices = response_body.get("choices", [])

                if choices:
                    message = choices[0].get("message", {})
                    if message:
                        narrative = message.get("content", "").strip().strip('"')
                        parts = custom_id.split("_") if custom_id else []
                        category = "Unknown"
                        if len(parts) >= 3:
                            category_slug = "_".join(parts[1:-1])
                            # Map slug to category name (Adjust mapping if your slugs/categories change!)
                            if category_slug == "strong_emotional":
                                category = "emotional"
                            elif (
                                category_slug == "complex_chaotic"
                            ):  # Match potential slug format
                                category = "complex"
                            elif (
                                category_slug == "complex/chaotic"
                            ):  # Match potential slug format
                                category = "complex"
                            elif category_slug == "routine":
                                category = "routine"
                            else:
                                print(
                                    f"  Warning (L{line_num+1}): Unrecognized category slug '{category_slug}' from custom_id: {custom_id}"
                                )
                        else:
                            print(
                                f"  Warning (L{line_num+1}): Could not parse category structure from custom_id: {custom_id}"
                            )

                        results_data.append(
                            {
                                "text": narrative,
                                "category": category,
                                "batch_id": batch_id,
                                "custom_id": custom_id,
                            }
                        )
                        successful_parses += 1
                    else:
                        print(
                            f"  Warning (L{line_num+1}): 'message' missing in choices for custom_id: {custom_id}"
                        )
                else:
                    print(
                        f"  Warning (L{line_num+1}): No choices found in response for custom_id: {custom_id}"
                    )

            except json.JSONDecodeError as json_e:
                print(
                    f"  Error decoding JSON on line {line_num + 1}: {json_e}\n  Content Start: '{line[:80]}...'"
                )
            except Exception as parse_e:
                print(f"  Error processing item on line {line_num + 1}: {parse_e}")

        print(
            f"Successfully parsed {successful_parses} results out of {len(lines)} lines."
        )

        # --- Save final data for this batch ---
        if results_data:
            print(f"Saving structured data to {target_filename}...")
            try:
                with open(target_filename, "w", encoding="utf-8") as f:
                    json.dump(results_data, f, indent=2, ensure_ascii=False)
                print("Data successfully saved.")
                total_processed += 1
            except IOError as e:
                print(f"Error saving data to file: {e}")
        else:
            print("No results were successfully parsed to save for this batch.")

    except Exception as e:
        print(
            f"An error occurred while retrieving or processing results for Batch {batch_id}: {e}"
        )
        # Decide if you want to skip or stop on error

print(f"\n--- Batch Processing Summary ---")
print(f"Batches Processed and Saved: {total_processed}")
print(f"Batches Skipped (No Output ID or Already Existed): {total_skipped}")
print("--- Script Finished ---")
