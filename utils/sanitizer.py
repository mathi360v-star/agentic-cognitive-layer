import json
import re
import os

def clean_solution_block(solution: str, target_language: str) -> str:
    if not solution: return ""
    # Preserve LaTeX for Math/Physics
    if target_language == "Agnostic/Math":
        return solution.strip()
    # Strip markdown for Code
    solution = re.sub(r"^```[a-zA-Z]*\n", "", solution, flags=re.MULTILINE)
    solution = re.sub(r"^```\n?", "", solution, flags=re.MULTILINE)
    return solution.strip()

def sanitize_dataset():
    print("\n--- [GATE 1] Dual-Stream SFT & DPO Extraction ---")
    
    os.makedirs("dataset", exist_ok=True)
    RAW_DATA_PATH = "dataset/training_traces.jsonl"
    SFT_OUT = "dataset/clean_sft_data.jsonl" # For Knowledge
    DPO_OUT = "dataset/clean_dpo_data.jsonl" # For Reasoning
    
    if not os.path.exists(RAW_DATA_PATH):
        print("[-] No raw data found. Halting.")
        return

    sft_count = 0
    dpo_count = 0

    with open(RAW_DATA_PATH, 'r', encoding='utf-8') as infile, \
         open(SFT_OUT, 'w', encoding='utf-8') as sft_file, \
         open(DPO_OUT, 'w', encoding='utf-8') as dpo_file:
        
        for line in infile:
            try:
                trace = json.loads(line)
                
                # Metadata Extraction
                problem = trace.get("problem_statement", trace.get("problem", "")).strip()
                final_code = trace.get("final_correct_code", "").strip()
                laws = trace.get("fundamental_laws", "Standard STEM Principles").strip()
                tier = trace.get("difficulty_tier", "Tier 1 (Foundational)").strip()
                rca_history = trace.get("rca_history", [])
                target_lang = trace.get("target_language", "C/Python")
                domain = trace.get("domain", "Engineering")

                # Basic Integrity Check
                if not problem or not final_code or len(final_code) < 30:
                    continue

                clean_final = clean_solution_block(final_code, target_lang)

                # --- STREAM 1: THE SFT DATA (The "Gold" Answer) ---
                # We save EVERY success here.
                sft_thought = f"Difficulty: {tier}\nLaws Applied: {laws}\n"
                if rca_history:
                    sft_thought += f"Self-Correction: {rca_history[-1].get('generalized_rule', 'Refined logic')}"
                
                sft_json = {
                    "instruction": f"Domain: {domain}\nProblem: {problem}",
                    "output": f"<think>\n{sft_thought}\n</think>\n\n{clean_final}"
                }
                sft_file.write(json.dumps(sft_json) + "\n")
                sft_count += 1

                # --- STREAM 2: THE DPO DATA (The "Preference" Pair) ---
                # We only save if there was a struggle (RCA history exists)
                if rca_history:
                    raw_fail = rca_history[0].get("failed_code_snapshot", "")
                    clean_fail = clean_solution_block(raw_fail, target_lang)
                    
                    if clean_fail and len(clean_fail) > 10:
                        dpo_json = {
                            "prompt": f"Domain: {domain}\nDifficulty: {tier}\nProblem:\n{problem}",
                            "chosen": f"<think>\nVerification against laws: {laws}\nCorrection: {rca_history[0].get('generalized_rule', '')}\n</think>\n\n{clean_final}",
                            "rejected": f"<think>\nInitial approach assuming standard patterns...\n</think>\n\n{clean_fail}"
                        }
                        dpo_file.write(json.dumps(dpo_json) + "\n")
                        dpo_count += 1
                
            except Exception:
                continue

    print(f"[+] SFT Extracted: {sft_count} samples.")
    print(f"[+] DPO Extracted: {dpo_count} preference pairs.")

if __name__ == "__main__":
    sanitize_dataset()