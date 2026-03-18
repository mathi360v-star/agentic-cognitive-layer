import json
import re
import os

def clean_code_block(code: str) -> str:
    code = re.sub(r"^```[a-zA-Z]*\n", "", code, flags=re.MULTILINE)
    code = re.sub(r"^```\n?", "", code, flags=re.MULTILINE)
    return code.strip()

def sanitize_dataset():
    print("\n--- Starting Data Deep-Sanitization ---")
    
    # 1. Force the creation of the dataset directory to prevent path crashes
    os.makedirs("dataset", exist_ok=True)
    
    RAW_DATA_PATH = "dataset/training_traces.jsonl"
    CLEAN_DATA_PATH = "dataset/clean_training_data.jsonl"

    if not os.path.exists(RAW_DATA_PATH):
        print(f"[-] FATAL: No raw traces found. The AI did not harvest any valid solutions this round.")
        # Create an empty file so the pipeline doesn't crash, but it will be ignored by the uploader
        open(CLEAN_DATA_PATH, 'a').close() 
        return

    valid_traces = 0

    with open(RAW_DATA_PATH, 'r', encoding='utf-8') as infile, \
         open(CLEAN_DATA_PATH, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            if not line.strip(): continue
            try:
                trace = json.loads(line)
                problem = trace.get("problem", "")
                final_code = trace.get("final_correct_code", "")
                rca_history = trace.get("rca_history", [])
                
                if len(final_code) < 30: continue 
                
                clean_code = clean_code_block(final_code)
                
                thought_process = "Analyzing constraints to prevent edge case failure.\n"
                if rca_history:
                    thought_process += "Internal debugging history:\n"
                    for rca in rca_history:
                        thought_process += f"- Flaw: {rca.get('mechanical_failure', '')}\n"
                        thought_process += f"- Correction: {rca.get('generalized_rule', '')}\n"
                
                formatted_trace = {
                    "instruction": f"Domain: Engineering\nProblem:\n{problem}",
                    "output": f"<think>\n{thought_process}\n</think>\n\n{clean_code}"
                }
                
                outfile.write(json.dumps(formatted_trace) + "\n")
                valid_traces += 1
                
            except Exception as e:
                print(f"[!] Sanitizer caught a malformed line: {e}")
                pass 

    print(f"[+] Sanitization Complete! {valid_traces} Golden Traces extracted to {CLEAN_DATA_PATH}.")

if __name__ == "__main__":
    sanitize_dataset()