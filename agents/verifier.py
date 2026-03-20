import json
import re
from langchain_core.messages import SystemMessage, HumanMessage
from schemas.models import AgenticState
from utils.llm_router import heavy_async_invoke 

async def node_verifier(state: AgenticState) -> AgenticState:
    print(f"\n--- [Layer 1.5] The Verifier is auditing the {state.get('domain', 'STEM')} problem ---")
    
    problem = state.get("problem_statement", "").strip()
    domain = state.get("domain", "General Engineering")
    
    if not problem or len(problem) < 30:
        print("[-] Verifier REJECTED: Problem statement too short.")
        return {"problem_is_valid": False, "audit_feedback": "Empty problem."}

    try:
        # THE UNIVERSAL STEM CONSTITUTION
        system_prompt = f"""
        Role: Principal STEM Auditor & Formal Logic Verifier.
        Expertise: {domain}
        
        Task: Perform an Epistemic Audit. You must catch "AI Hallucinations" before they enter the dataset.
        
        AUDIT CRITERIA:
        1. DIMENSIONAL/LOGICAL REALITY: In Math/Physics, are the units consistent? In Code, is the logic physically possible?
        2. CONSTRAINTS: Are the constraints (e.g., O(n) time, 8-byte alignment) mutually exclusive? 
        3. SOLVABILITY: Can a 70B model solve this in 5 steps? If it requires a supercomputer, ABORT.
        
        OUTPUT FORMAT (Strict JSON Only):
        {{
            "is_valid": true,
            "flaw_reasoning": "Valid"
        }}
        """

        human_prompt = f"Problem: {problem}\nLanguage/Context: {state.get('target_language')}"

        # HEAVY LANE: This requires Llama 3.3 70B or Gemini 2.0 level intelligence
        raw_response = await heavy_async_invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ], temperature=0.0)

        # ENTERPRISE JSON CLEANING: Bypasses markdown and escape char errors
        # Removes ```json ... ``` blocks if present
        clean_json = re.sub(r'```json\s*|\s*```', '', raw_response).strip()
        # Find the first { and last }
        start_idx = clean_json.find('{')
        end_idx = clean_json.rfind('}')
        
        if start_idx == -1 or end_idx == -1:
            raise ValueError("No JSON structure found in response.")
            
        final_json = clean_json[start_idx:end_idx+1]
        parsed_data = json.loads(final_json)
        
        is_valid = parsed_data.get("is_valid", False)
        feedback = parsed_data.get("flaw_reasoning", "Unknown logical error.")
        
        if is_valid:
            print(f"[+] Verifier APPROVED {domain} problem.")
        else:
            print(f"[-] Verifier REJECTED. Reason: {feedback}")
            
        return {
            "problem_is_valid": is_valid,
            "audit_feedback": feedback
        }
            
    except Exception as e:
        print(f"[!] Verifier Failure: {str(e)}. Defaulting to rejection to save API credits.")
        return {
            "problem_is_valid": False,
            "audit_feedback": f"System Parsing Error: {str(e)}"
        }