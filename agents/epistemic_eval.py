from schemas.models import AgenticState

async def node_epistemic_evaluator(state: AgenticState, router):
    """
    The Epistemic Shield: Stops the graph if the Professor 
    outputs garbage or if the API returns an empty string.
    """
    print("--- [Layer 1.2] The Epistemic Evaluator is testing reality constraints ---")
    
    # 1. Extraction & Basic Validation
    problem = state.get('problem_statement', '')
    
    # CRITICAL: Prevent the 'Empty String' 429 loop
    if not problem or len(problem) < 20:
        print("[-] EPISTEMIC REJECTION: Problem statement is missing or too short.")
        return {
            "problem_is_valid": False, 
            "audit_feedback": "The problem statement was empty or malformed."
        }

    # 2. Logic Check via Sharded Router
    prompt = f"""
    Act as a Principal STEM Auditor. Evaluate the following task for Epistemic Solvability.
    
    THE PROBLEM:
    {problem}
    
    Is this problem logically solvable, or is it a hallucination/empty text?
    Output EXACTLY one of these:
    - STATUS: SOLVABLE
    - STATUS: ABORT (followed by a brief reason)
    """
    
    try:
        # Use the sharded router's invoke method
        res = await router.invoke([{"role": "user", "content": prompt}], temperature=0.0)
        
        is_valid = "STATUS: SOLVABLE" in res.upper()
        
        if is_valid:
            print(f"[+] Epistemic Check Passed: Problem is grounded in reality.")
        else:
            print(f"[-] EPISTEMIC REJECTION: {res.strip()}")
            
        return {
            "problem_is_valid": is_valid, 
            "audit_feedback": res if not is_valid else "Valid"
        }
        
    except Exception as e:
        print(f"[!] Epistemic Evaluator Failed: {e}")
        return {"problem_is_valid": False, "audit_feedback": f"System error: {e}"}