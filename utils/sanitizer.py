import json
import re
import os

def clean_code_block(code: str) -> str:
    """
    Bulletproof markdown stripper. Handles trailing spaces, missing newlines, 
    and weird LLM formatting anomalies.
    """
    # 1. First, try to extract strictly what is inside the code blocks if they exist
    match = re.search(r'```[a-zA-Z]*\s*([\s\S]*?)```', code)
    if match:
        code = match.group(1)
    
    # 2. Fallback: Aggressively rip out any remaining backticks or language tags
    code = re.sub(r'```[a-zA-Z]*', '', code)
    code = re.sub(r'```', '', code)
    
    return code.strip()

def sanitize_dataset():
    print("\n--- Starting Data Deep-Sanitization ---")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    RAW_DATA_PATH = os.path.join(current_dir, "../dataset/training_traces.jsonl")
    CLEAN_DATA_PATH = os.path.join(current_dir, "../dataset/clean_training_data.jsonl")

    # Ensure the target directory exists so the file writer doesn't crash
    os.makedirs(os.path.dirname(CLEAN_DATA_PATH), exist_ok=True)

    if not os.path.exists(RAW_DATA_PATH):
        print(f"[-] No raw traces found to sanitize. Skipping.")
        return

    valid_traces = 0
    rejected_traces = 0

    with open(RAW_DATA_PATH, 'r', encoding='utf-8') as infile, \
         open(CLEAN_DATA_PATH, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            try:
                trace = json.loads(line)
                problem = trace.get("problem", "")
                final_code = trace.get("final_correct_code", "")
                rca_history = trace.get("rca_history", [])
                
                # Make the domain dynamic if it exists in your harvester, otherwise fallback
                domain = trace.get("domain", "Advanced Engineering & Computer Science")
                
                # Filter 1: Reject hallucinations (code too short)
                if len(final_code) < 30:
                    rejected_traces += 1
                    continue
                
                clean_code = clean_code_block(final_code)
                
                # Filter 2: Build the DeepSeek-R1 style internal monologue
                thought_process = "Analyzing the constraints. I must avoid edge case failures.\n"
                
                if rca_history:
                    thought_process += "\nMy internal debugging history for this concept:\n"
                    for rca in rca_history:
                        thought_process += f"- Flaw identified: {rca.get('mechanical_failure', 'Unknown error')}\n"
                        thought_process += f"- Corrective Logic: {rca.get('generalized_rule', 'Verify constraints')}\n"
                
                thought_process += "\nFinalizing optimal logic architecture."
                
                # Format into the strict Instruction/Response structure for ORPO Fine-Tuning
                formatted_trace = {
                    "instruction": f"Domain: {domain}\nProblem:\n{problem}",
                    "output": f"<think>\n{thought_process}\n</think>\n\n{clean_code}"
                }
                
                outfile.write(json.dumps(formatted_trace) + "\n")
                valid_traces += 1
                
            except Exception as e:
                rejected_traces += 1
                pass # Silently skip corrupted lines to keep the pipeline moving

    print(f"[+] Sanitization Complete!")
    print(f"    -> Golden Traces Kept: {valid_traces}")
    print(f"    -> Poisoned Traces Rejected: {rejected_traces}")

if __name__ == "__main__":
    sanitize_dataset()