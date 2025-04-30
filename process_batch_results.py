import openai
import os
import json

# --- Configuration ---
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Error: OPENAI_API_KEY environment variable not set.")
    exit()

client = openai.OpenAI(api_key=api_key)

OUTPUT_FILE_ID = "file-XXSUtCWHePEbk7PJajNXCm"  # The ID from the completed batch job
FINAL_JSON_OUTPUT_FILENAME = "synthetic_narrative_data.json"

# --- Retrieve & Parse Results ---
print(f"--- Downloading Results from Output File ID: {OUTPUT_FILE_ID} ---")
results_data = []
try:
    # Download the file content
    output_content_response = client.files.content(OUTPUT_FILE_ID)
    output_content = output_content_response.read().decode("utf-8")
    print("Successfully downloaded output file content.")

    # Parse the JSONL content more robustly
    lines = output_content.strip().splitlines()
    print(f"Parsing {len(lines)} lines from output file...")
    successful_parses = 0
    for line_num, line in enumerate(lines):
        if not line.strip():  # Skip empty lines if any
            # print(f"Skipping empty line {line_num + 1}") # Optional: for debugging
            continue
        try:
            result_item = json.loads(line)
            custom_id = result_item.get("custom_id")
            response_body = result_item.get("response", {}).get("body", {})
            choices = response_body.get("choices", [])

            if choices:
                message = choices[0].get("message", {})
                if message:  # Check if message exists
                    narrative = message.get("content", "").strip().strip('"')
                    # Infer category from custom_id (e.g., "request_strong_emotional_1")
                    parts = custom_id.split("_") if custom_id else []
                    category = "Unknown"  # Default
                    # Need at least 3 parts: request, category_slug_part(s), index
                    if len(parts) >= 3:
                        # Reconstruct the category slug (e.g., "strong_emotional")
                        # It's all parts between the first ("request") and the last (index)
                        category_slug = "_".join(parts[1:-1])

                        # Map the reconstructed slug back to the original category name
                        if category_slug == "strong_emotional":
                            category = "emotional"
                        elif category_slug == "complex/chaotic":
                            category = "complex"
                        elif category_slug == "routine":
                            category = "routine"
                        else:
                            print(
                                f"Warning: Unrecognized category slug '{category_slug}' from custom_id: {custom_id}"
                            )
                    else:
                        print(
                            f"Warning: Could not parse category structure from custom_id: {custom_id}"
                        )

                    results_data.append({"text": narrative, "category": category})
                    successful_parses += 1
                else:
                    print(
                        f"Warning: 'message' object missing in choices for custom_id: {custom_id} on line {line_num + 1}"
                    )
            else:
                print(
                    f"Warning: No choices found in response for custom_id: {custom_id} on line {line_num + 1}"
                )

        except json.JSONDecodeError as json_e:
            print(
                f"Error decoding JSON on line {line_num + 1}: {json_e}\\nContent Start: '{line[:100]}...'"
            )
            # Show line number and start of content
        except Exception as parse_e:
            print(f"Error processing result item on line {line_num + 1}: {parse_e}")

    print(f"Successfully parsed {successful_parses} results out of {len(lines)} lines.")

    # --- Save final data ---
    if results_data:
        print(f"Saving final structured data to {FINAL_JSON_OUTPUT_FILENAME}...")
        try:
            with open(FINAL_JSON_OUTPUT_FILENAME, "w", encoding="utf-8") as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)
            print("Data successfully saved.")
        except IOError as e:
            print(f"Error saving data to file: {e}")
    else:
        print("No results were successfully parsed to save.")

except Exception as e:
    print(f"An error occurred while retrieving or processing results: {e}")

print("--- Result processing finished ---")
