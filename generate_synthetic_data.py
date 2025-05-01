import openai
import json
import os
import time
import tempfile  # To create temporary file for upload
import random
from tqdm import tqdm  # Keep for polling maybe

# -------------------------------------
# Configuration
# -------------------------------------
# --- API Key ---
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")
client = openai.OpenAI(api_key=api_key)  # Use client instance

# --- Generation Parameters ---
MODEL_NAME = "gpt-4.1-nano"
CATEGORIES = ["Strong Emotional", "Complex/Chaotic", "Routine"]
EXAMPLES_PER_CATEGORY = 500
OUTPUT_FILENAME = "synthetic_narrative_data.json"
MAX_TOKENS = 60
TEMPERATURE = 0.7

# --- Scenario Diversity ---  <-- ADD THIS SECTION
SCENARIO_TYPES = [
    "something someone says in a specific situation",
    "a description of what someone sees",
    "how someone physically or emotionally reacts to an event",
    "a brief memory someone recalls",
    "how someone wishes they could respond or act",
    "a moment of self-reflection or internal thought",
    "an expression of a need or want",
    "a description of a physical or emotional feeling",
    "a short plan or intention someone has",
    "an observation about another person's behavior",
]

# --- Batch API Config ---
BATCH_INPUT_FILENAME = "batch_input_requests.jsonl"
BATCH_POLL_INTERVAL = 30  # seconds


# -------------------------------------
# 1. Prepare Batch Input File
# -------------------------------------
def create_batch_input_file(filename):
    print(f"--- Preparing Batch Input File: {filename} ---")
    requests = []
    request_id_counter = 0
    for category in CATEGORIES:
        # --- Create a balanced and shuffled list of scenario types for this category ---
        num_scenario_types = len(SCENARIO_TYPES)
        examples_per_type = EXAMPLES_PER_CATEGORY // num_scenario_types
        # Ensure it divides evenly (add check later if needed, it does for 500/10)
        category_scenario_assignments = []
        for type_name in SCENARIO_TYPES:
            category_scenario_assignments.extend([type_name] * examples_per_type)

        # Add remaining examples if not perfectly divisible (not needed for 500/10)
        remainder = EXAMPLES_PER_CATEGORY % num_scenario_types
        if remainder > 0:
            category_scenario_assignments.extend(SCENARIO_TYPES[:remainder])

        random.shuffle(category_scenario_assignments)  # Shuffle the assignments
        # --- End scenario type preparation ---

        for i in range(EXAMPLES_PER_CATEGORY):
            request_id_counter += 1
            custom_id = f"request_{category.lower().replace(' ', '_')}_{i+1}"

            # Get the pre-assigned scenario type for this index
            scenario_type = category_scenario_assignments[i]

            # New prompt emphasizing uniqueness, category, and scenario type
            prompt = (
                f"You are a narrative generator. Please provide a **unique** short textual scenario "
                f"(1-2 concise sentences) that clearly illustrates the category: '{category}'. "
                f"Specifically, the scenario should describe: '{scenario_type}'. "
                f"Give a **distinct example**, different from common clichés. "
                f"Scenario {i+1} for category '{category}', focusing on '{scenario_type}'. "
                f"Do not add labels or extra formatting."
            )

            request_body = {
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE,
                "n": 1,
                "stop": None,
            }

            requests.append(
                {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": request_body,
                }
            )

    # Write to JSONL file
    try:
        with open(filename, "w", encoding="utf-8") as f:
            for request in requests:
                f.write(json.dumps(request) + "\n")
        print(f"Successfully wrote {len(requests)} requests to {filename}")
        return filename
    except IOError as e:
        print(f"Error writing batch input file: {e}")
        return None


# -------------------------------------
# 2. Upload Input File
# -------------------------------------
def upload_file(filename):
    print(f"--- Uploading Input File: {filename} ---")
    try:
        with open(filename, "rb") as f:
            batch_input_file = client.files.create(file=f, purpose="batch")
        print(f"File uploaded successfully. File ID: {batch_input_file.id}")
        # Clean up local file after upload
        try:
            # os.remove(filename) # <-- Commented out to keep the input file
            # print(f"Removed local input file: {filename}")
            print(
                f"Keeping local input file for inspection: {filename}"
            )  # Added message
        except OSError as e:
            print(f"Warning: Could not remove local input file {filename}: {e}")
        return batch_input_file.id
    except Exception as e:
        print(f"Error uploading file: {e}")
        return None


# -------------------------------------
# 3. Create Batch Job
# -------------------------------------
def create_batch(file_id):
    print(f"--- Creating Batch Job with File ID: {file_id} ---")
    try:
        batch_job = client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",  # Or adjust as needed
            metadata={"description": "Synthetic narrative generation"},
        )
        print(f"Batch job created successfully. Batch ID: {batch_job.id}")
        return batch_job.id
    except Exception as e:
        print(f"Error creating batch job: {e}")
        return None


# -------------------------------------
# 4. Monitor Batch Job
# -------------------------------------
def monitor_batch(batch_id):
    print(f"--- Monitoring Batch Job: {batch_id} ---")
    while True:
        try:
            batch_status = client.batches.retrieve(batch_id)
            status = batch_status.status
            print(
                f"  Current status: {status} (Requests: {batch_status.request_counts.completed}/{batch_status.request_counts.total}, Failed: {batch_status.request_counts.failed})"
            )

            if status in ["completed", "failed", "cancelled", "expired"]:
                print(f"Batch job finished with status: {status}")
                return batch_status

            # Wait before polling again
            time.sleep(BATCH_POLL_INTERVAL)

        except Exception as e:
            print(
                f"Error retrieving batch status: {e}. Retrying in {BATCH_POLL_INTERVAL}s..."
            )
            time.sleep(BATCH_POLL_INTERVAL)


# -------------------------------------
# 5. Retrieve & Parse Results
# -------------------------------------
def get_and_parse_results(batch_status):
    if batch_status.status != "completed":
        print("Batch job did not complete successfully. Cannot retrieve results.")
        return None

    output_file_id = batch_status.output_file_id
    error_file_id = batch_status.error_file_id

    if error_file_id:
        print(f"Warning: Batch job produced errors. Error File ID: {error_file_id}")
        # Optionally download and inspect the error file here

    if not output_file_id:
        print("Error: Batch job completed but no output file ID found.")
        return None

    print(f"--- Downloading Results from Output File ID: {output_file_id} ---")
    try:
        # Download the file content
        output_content_response = client.files.content(output_file_id)
        output_content = output_content_response.read().decode("utf-8")
        print("Successfully downloaded output file content.")

        # Clean up the downloaded file on OpenAI's side (optional)
        try:
            # client.files.delete(output_file_id) # Uncomment to delete after download
            pass
        except Exception as del_e:
            print(f"Warning: Could not delete output file {output_file_id}: {del_e}")

        # Parse the JSONL content
        results_data = []
        lines = output_content.strip().split("\n")
        print(f"Parsing {len(lines)} lines from output file...")
        for line in lines:
            try:
                result_item = json.loads(line)
                custom_id = result_item.get("custom_id")
                response_body = result_item.get("response", {}).get("body", {})
                choices = response_body.get("choices", [])

                if choices:
                    narrative = (
                        choices[0]
                        .get("message", {})
                        .get("content", "")
                        .strip()
                        .strip('"')
                    )
                    # Infer category from custom_id (e.g., "request_strong_emotional_1")
                    parts = custom_id.split("_")
                    if len(parts) > 1:
                        category_key = (
                            parts[1].replace("-", " ").title()
                        )  # Simple heuristic
                        if category_key == "Strong Emotional":
                            category = "Strong Emotional"
                        elif category_key == "Complex Chaotic":
                            category = "Complex/Chaotic"
                        elif category_key == "Routine":
                            category = "Routine"
                        else:
                            category = "Unknown"

                        results_data.append({"text": narrative, "category": category})
                    else:
                        print(
                            f"Warning: Could not parse category from custom_id: {custom_id}"
                        )
                else:
                    print(
                        f"Warning: No choices found in response for custom_id: {custom_id}"
                    )

            except json.JSONDecodeError as json_e:
                print(f"Error decoding JSON line: {json_e}\nLine: '{line[:100]}...'")
            except Exception as parse_e:
                print(f"Error parsing result item: {parse_e}")

        print(f"Successfully parsed {len(results_data)} results.")

        # --- Deduplication Step ---
        print("--- Deduplicating Results ---")
        unique_narratives = set()
        deduplicated_data = []
        duplicates_removed = 0
        for item in results_data:
            # Normalize text (lowercase, strip whitespace) for comparison
            narrative_text = item.get("text", "").strip().lower()
            if narrative_text and narrative_text not in unique_narratives:
                unique_narratives.add(narrative_text)
                deduplicated_data.append(item)
            elif narrative_text:
                duplicates_removed += 1

        print(f"Removed {duplicates_removed} duplicate narratives.")
        print(f"Final unique narratives: {len(deduplicated_data)}")
        # --- End Deduplication ---

        return deduplicated_data  # Return the deduplicated list

    except Exception as e:
        print(f"Error retrieving or parsing results: {e}")
        return None


# -------------------------------------
# Main Execution Flow
# -------------------------------------
if __name__ == "__main__":
    print("--- Starting Synthetic Narrative Data Generation using Batch API ---")

    # 1. Create input file
    input_file = create_batch_input_file(BATCH_INPUT_FILENAME)
    if not input_file:
        exit()

    # 2. Upload file
    uploaded_file_id = upload_file(input_file)
    if not uploaded_file_id:
        exit()

    # 3. Create batch job
    batch_job_id = create_batch(uploaded_file_id)
    if not batch_job_id:
        exit()

    # 4. Monitor batch job
    final_batch_status = monitor_batch(batch_job_id)
    if not final_batch_status:
        exit()

    # 5. Get and parse results
    synthetic_data = get_and_parse_results(final_batch_status)
    if not synthetic_data:
        print("Failed to retrieve valid results from batch job.")
        exit()

    print(
        f"\n--- Batch Processing Complete. Retrieved {len(synthetic_data)} examples total. ---"
    )

    # 6. Save final data
    print(f"Saving final structured data to {OUTPUT_FILENAME}...")
    try:
        with open(OUTPUT_FILENAME, "w", encoding="utf-8") as f:
            json.dump(synthetic_data, f, indent=2, ensure_ascii=False)
        print("Data successfully saved.")
    except IOError as e:
        print(f"Error saving data to file: {e}")

    print("--- Script Finished ---")
