import json
import re
import os

def clean_code_block(code: str) -> str:
    if not code: return ""
    code = re.sub(r"^```[a-zA-Z]*\n", "", code, flags=re.MULTILINE)
    code = re.sub(r"^```\n?", "", code, flags=re.MULTILINE)
    return code.strip()

def sanitize_dataset():
    print("\n--- [GATE 1] Deep-Sanitization & Data Integrity Check ---")
    
    os.makedirs("dataset", exist_ok=True)
    RAW_DATA_PATH = "dataset/training_traces.jsonl"
    CLEAN_DATA_PATH = "dataset/clean_training_data.jsonl"

    if not os.path.exists(RAW_DATA_PATH):
        print("[-] No raw data found. Halting sanitizer.")
        open(CLEAN_DATA_PATH, 'w').close() # Ensure an empty file exists so bash doesn't crash
        return

    valid_traces = 0

    with open(RAW_DATA_PATH, 'r', encoding='utf-8') as infile, \
         open(CLEAN_DATA_PATH, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            if not line.strip(): continue # Skip blank lines
            
            try:
                trace = json.loads(line)
                problem = trace.get("problem", "").strip()
                final_code = trace.get("final_correct_code", "").strip()
                rca_history = trace.get("rca_history", [])
                
                # ---------------------------------------------------------
                # STRICT INTEGRITY CHECKS (Drops corrupted/null data)
                # ---------------------------------------------------------
                # 1. Reject if problem statement is missing or too short
                if not problem or len(problem) < 30:
                    print("[-] Dropping trace: Problem statement missing or too short.")
                    continue
                    
                # 2. Reject if code is missing or impossibly short
                if not final_code or len(final_code) < 30:
                    print("[-] Dropping trace: Code missing or too short.")
                    continue
                
                clean_code = clean_code_block(final_code)
                if not clean_code: 
                    continue # Reject if markdown stripping leaves nothing
                
                # Construct DeepSeek-style thought process
                thought_process = "Analyzing constraints to prevent edge case failure.\n"
                if rca_history:
                    thought_process += "Internal debugging history:\n"
                    for rca in rca_history:
                        flaw = rca.get('mechanical_failure', '').strip()
                        rule = rca.get('generalized_rule', '').strip()
                        # Only append if the AI actually provided text, not nulls
                        if flaw and rule:
                            thought_process += f"- Flaw: {flaw}\n- Correction: {rule}\n"
                
                formatted_trace = {
                    "instruction": f"Domain: Engineering\nProblem:\n{problem}",
                    "output": f"<think>\n{thought_process}\n</think>\n\n{clean_code}"
                }
                
                outfile.write(json.dumps(formatted_trace) + "\n")
                valid_traces += 1
                
            except json.JSONDecodeError:
                print("[!] Dropping trace: Critical JSON corruption detected.")
                continue
            except Exception as e:
                print(f"[!] Dropping trace: Unknown error -> {e}")
                continue

    print(f"[+] Integrity Check Passed. {valid_traces} Flawless Traces securely extracted to {CLEAN_DATA_PATH}.")

if __name__ == "__main__":
    sanitize_dataset()