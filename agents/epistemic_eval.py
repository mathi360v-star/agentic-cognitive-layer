from schemas.models import AgenticState
from utils.llm_router import safe_async_invoke

async def node_epistemic_evaluator(state: AgenticState):
    """Refined Gatekeeper: Ensures the problem is solvable within the specific STEM domain."""
    print("--- [Layer 1.2] The Epistemic Evaluator is testing reality constraints ---")
    
    problem = state.get('problem_statement', '')
    domain = state.get('domain', 'General Engineering')
    language = state.get('target_language', 'Agnostic/Logic')
    
    # THE REFINED AUDITOR PROMPT
    prompt = f"""
    Act as a Principal STEM Auditor. Evaluate the following task for Epistemic Solvability.
    
    DOMAIN: {domain}
    CONTEXT/LANGUAGE: {language}
    
    THE PROBLEM:
    {problem}
    
    CRITERIA FOR REJECTION (STATUS: ABORT):
    1. PARADOXES: Does the problem contain mutually exclusive constraints (e.g., O(1) sorting or a 100% efficient heat engine)?
    2. TOOL MISMATCH: Does it ask for a feature that doesn't exist in {language} (e.g., Python manual memory pointers)?
    3. SCOPE CREEP: Is the problem too large to be solved in a single function/derivation (e.g., "Write a full OS")?
    4. DATA ABSURDITY: Are the physical constants or mathematical premises hallucinated?

    OUTPUT:
    If valid, output: "STATUS: SOLVABLE".
    If invalid, output: "STATUS: ABORT" followed by a detailed 1-sentence explanation.
    """
    
    # We use Lane 1 (safe_async_invoke) for the quick logic check
    evaluation = await safe_async_invoke([{"role": "user", "content": prompt}], temperature=0.0)
    
    # Handle the decision
    if "STATUS: ABORT" in evaluation.upper():
        print(f"[-] EPISTEMIC REJECTION: {evaluation.strip()}")
        return {
            "problem_is_valid": False, 
            "audit_feedback": evaluation.strip(),
            "iteration_count": 0 # Reset count for the new problem
        }
    
    print(f"[+] Epistemic Check Passed: {domain} task is grounded.")
    return {"problem_is_valid": True, "audit_feedback": "Solvable"}