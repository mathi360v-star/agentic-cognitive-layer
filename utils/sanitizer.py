import json
import re
import os

# --- [UPGRADE: CLEANING LOGIC] ---
def clean_solution_block(text: str, target_lang: str = "C/Python") -> str:
    """Strips markdown artifacts while preserving math/code integrity."""
    if not text: return ""
    
    # If it's math/physics, we keep the formatting but trim whitespace
    if target_lang == "Agnostic/Math":
        return text.strip()
        
    # If it's code, we strip the ```python or ```c tags
    text = re.sub(r"^```[a-zA-Z]*\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"^```\n?", "", text, flags=re.MULTILINE)
    return text.strip()

# --- [UPGRADE: DUAL-STREAM EXTRACTION] ---
def sanitize_dataset():
    print("\n--- [GATE 1] Hybrid-Density SFT & DPO Extraction ---")
    
    # Use absolute paths to prevent "File Not Found" errors in GitHub Actions
    base_dir = os.getcwd()
    raw_path = os.path.join(base_dir, "dataset/training_traces.jsonl")
    sft_out = os.path.join(base_dir, "dataset/clean_sft.jsonl")
    dpo_out = os.path.join(base_dir, "dataset/clean_dpo.jsonl")

    if not os.path.exists(raw_path):
        print(f"[-] No raw data found at {raw_path}. Skipping sanitization.")
        return
    
    sft_count = 0
    dpo_count = 0

    with open(raw_path, 'r', encoding='utf-8') as f, \
         open(sft_out, 'w', encoding='utf-8') as sft, \
         open(dpo_out, 'w', encoding='utf-8') as dpo:
        
        for line in f:
            if not line.strip(): continue
            
            try:
                data = json.loads(line)
                
                # Extract and clean data
                problem = data.get('problem', '').strip()
                chosen = clean_solution_block(data.get('chosen', ''), data.get('target_language'))
                rejected = clean_solution_block(data.get('rejected', ''), data.get('target_language'))
                
                # Integrity Check: Do not save empty or corrupted traces
                if not problem or not chosen:
                    continue

                # 1. SFT Stream (Supervised Fine-Tuning - "The Gold Standard")
                sft_entry = {
                    "instruction": problem,
                    "output": chosen
                }
                sft.write(json.dumps(sft_entry) + "\n")
                sft_count += 1

                # 2. DPO Stream (Preference Optimization - "Right vs Wrong")
                if rejected and len(rejected) > 10:
                    dpo_entry = {
                        "prompt": problem,
                        "chosen": chosen,
                        "rejected": rejected
                    }
                    dpo.write(json.dumps(dpo_entry) + "\n")
                    dpo_count += 1
                    
            except Exception as e:
                print(f"[!] Sanitizer Trace Error: {e}")
                continue

    print(f"[+] Extraction Complete: {sft_count} SFT samples | {dpo_count} DPO pairs.")

if __name__ == "__main__":
    sanitize_dataset()