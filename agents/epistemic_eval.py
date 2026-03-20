from schemas.models import AgenticState
from utils.llm_router import safe_async_invoke

async def node_epistemic_evaluator(state: AgenticState):
    """Determines if a problem is physically and computationally solvable before wasting API credits."""
    print("--- [Layer 1.2] The Epistemic Evaluator is testing reality constraints ---")
    
    problem = state.get('problem_statement', '')
    domain = state.get('domain', 'General Engineering')
    
    prompt = f"""
    Act as a Principal Staff Engineer specializing in Epistemic Solvability. 
    Evaluate the following problem in the domain of {domain}.
    
    THE PROBLEM:
    {problem}
    
    Your task is to determine if this problem is fundamentally solvable, or if it contains paradoxical, hallucinatory, or impossible constraints. 
    
    Examples of UNSOLVABLE problems:
    - Asking for O(1) sorting.
    - Contradicting the Laws of Thermodynamics.
    - Requesting C-style pointer manipulation in a language that doesn't support it.
    
    OUTPUT FORMAT:
    If the problem is logically sound and solvable, output EXACTLY: "STATUS: SOLVABLE".
    If the problem contains paradoxes or impossible constraints, output EXACTLY: "STATUS: ABORT" followed by a 1-sentence explanation of the violation.
    """
    
    evaluation = await safe_async_invoke([{"role": "user", "content": prompt}])
    
    if "STATUS: ABORT" in evaluation.upper():
        print(f"[-] EPISTEMIC REJECTION: {evaluation.strip()}")
        # We set this to False so the graph routes it back to the Professor
        return {"problem_is_valid": False, "audit_feedback": evaluation.strip()}
    
    print("[+] Epistemic Check Passed: Problem is grounded in reality.")
    return {"problem_is_valid": True, "audit_feedback": "Valid"}