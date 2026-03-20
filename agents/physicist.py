from schemas.models import AgenticState
from utils.llm_router import safe_async_invoke

async def node_physicist(state: AgenticState):
    """Extracts immutable laws/axioms to prevent hallucinated solutions."""
    print("--- [Layer 1.7] The Physicist is locking fundamental laws & axioms ---")
    
    # We safely use .get() to prevent KeyErrors
    problem = state.get('problem_statement', '')
    
    # THE DOMAIN-AWARE PROMPT
    prompt = f"""
    Analyze the following STEM problem:
    {problem}
    
    If this is a Physics or Engineering problem, identify the strict fundamental physical laws (e.g., 'Conservation of Mass', 'Ohm's Law').
    If this is a Mathematics or Computer Science problem, identify the strict Theorems, Axioms, or Big-O constraints (e.g., 'Fundamental Theorem of Calculus', 'O(n log n) limits').
    
    Output ONLY a bulleted list of 2 to 4 laws/theorems. Do not solve the problem.
    """
    
    # Network Call
    laws = await safe_async_invoke([{"role": "user", "content": prompt}])
    
    # Safely return the new variable to the Universal Data Bus
    return {"fundamental_laws": laws.strip()}