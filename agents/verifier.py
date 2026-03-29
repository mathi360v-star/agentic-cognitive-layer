import json
import re
from schemas.models import AgenticState

async def node_verifier(state: AgenticState, router):
    """
    V3 High-IQ Auditor: Performs an Epistemic Audit on the problem statement.
    Uses the 'Heavy Lane' (70B+ models) for maximum logical rigor.
    """
    print(f"--- [Layer 1.5] Verifying {state.get('domain', 'STEM')} Problem ---")
    
    problem = state.get("problem_statement", "")
    domain = state.get("domain", "General Engineering")
    tier = state.get("difficulty_tier", "Tier 1")
    
    # Safety Check: If the Professor output nothing, abort immediately
    if not problem or len(problem) < 30:
        return {"problem_is_valid": False, "audit_feedback": "Problem is empty or too brief."}
    
    prompt = f"""
    Act as a Principal STEM Auditor. Perform a logic audit on this {tier} problem in {domain}.
    
    PROBLEM:
    {problem}
    
    CRITERIA:
    1. Are there contradictory constraints?
    2. Is the goal clear and solvable?
    3. Is the difficulty actually {tier}?
    
    Output ONLY a JSON object:
    {{"is_valid": true/false, "reason": "Detailed explanation"}}
    """
    
    try:
        # We use 'heavy=True' to ensure Llama-3.3-70B or Gemini 2.0 handles the audit
        raw = await router.invoke([{"role": "user", "content": prompt}], heavy=True)
        
        # Robust JSON Extraction
        match = re.search(r'\{[\s\S]*\}', raw)
        if not match:
            raise ValueError("No JSON found in Verifier response")
            
        data = json.loads(match.group(0))
        
        is_valid = data.get("is_valid", False)
        reason = data.get("reason", "No reason provided")
        
        if is_valid:
            print(f"[+] Verifier APPROVED {domain} problem.")
        else:
            print(f"[-] Verifier REJECTED: {reason}")
            
        return {
            "problem_is_valid": is_valid,
            "audit_feedback": reason
        }
        
    except Exception as e:
        print(f"[!] Verifier System Error: {e}")
        return {
            "problem_is_valid": False, 
            "audit_feedback": f"Verifier Parser Failure: {str(e)}"
        }