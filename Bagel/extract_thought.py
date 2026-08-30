import json
import os
import sys
import re

def process_and_save_jsonl(input_file):
    # 1. Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found {input_file}")
        return

    # 2. Determine output file path
    input_dir = os.path.dirname(os.path.abspath(input_file))
    output_filename = "think-generated_prompts.jsonl"
    output_file = os.path.join(input_dir, output_filename)

    print(f"Processing data...")
    
    # --- Core regular expression ---
    pattern = re.compile(
        r'(?:\*\*|)\b(?:refined\s+prompt|the\s+refined\s+prompt|final\s+scene\s+prompt|final\s+prompt|prompt)\b(?:\s+is|):*', 
        re.IGNORECASE
    )

    success_count = 0
    fallback_count = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:
            
            for line_idx, line in enumerate(f_in):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    row_data = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Warning: Line {line_idx + 1} JSON parsing failed, skipping.")
                    continue

                record_id = row_data.get("id", line_idx)
                content = row_data.get("thought", "")

                if not isinstance(content, str):
                    continue

                # --- Use regex to find the last match ---
                matches = list(pattern.finditer(content))
                
                if matches:
                    last_match = matches[-1]
                    extracted_text = content[last_match.end():].strip()
                    extracted_text = extracted_text.lstrip('*').strip()
                    success_count += 1
                else:
                    extracted_text = content.strip()
                    fallback_count += 1

                # =======================================================
                # [Modification]: Clean up </think> and <image_start> here
                # =======================================================
                extracted_text = extracted_text.replace("</think>", "")
                extracted_text = extracted_text.replace("<image_start>", "")
                # Strip again after cleaning to remove leading/trailing spaces
                extracted_text = extracted_text.strip()
                # =======================================================

                # Write to JSONL
                output_record = {
                    "id": record_id,
                    "prompt": extracted_text
                }
                f_out.write(json.dumps(output_record, ensure_ascii=False) + "\n")

        print(f"\n--- Processing Complete ---")
        print(f"Successfully extracted: {success_count} entries")
        print(f"Full text retained: {fallback_count} entries")
        print(f"Output file: {output_file}")

    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    input_path = "input.jsonl" 
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    process_and_save_jsonl(input_path)