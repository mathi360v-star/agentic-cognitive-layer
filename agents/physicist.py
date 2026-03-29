from schemas.models import AgenticState

async def node_physicist(state: AgenticState, config):
    """Extracts immutable laws/axioms to prevent hallucinated solutions."""
    print("--- [Layer 1.7] Locking Fundamental Laws ---")
    
    # 1. Extract the Sharded Router from the configuration
    router = config["configurable"].get("router")
    if not router:
        raise ValueError("Critical Error: Router not found in Node Configuration.")
    
    # 2. Safely extract problem from the state
    problem = state.get('problem_statement', '')
    if not problem:
        return {"fundamental_laws": "Standard STEM Axioms"}
        
    prompt = f"""
    Analyze the following STEM problem:
    {problem}
    
    If this is a Physics or Engineering problem, identify the strict fundamental physical laws (e.g., 'Conservation of Mass', 'Ohm's Law').
    If this is a Mathematics or Computer Science problem, identify the strict Theorems, Axioms, or Big-O constraints (e.g., 'Fundamental Theorem of Calculus', 'O(n log n) limits').
    
    Output ONLY a bulleted list of 2 to 4 laws/theorems. Do not solve the problem.
    """
    
    # 3. Use the Sharded Router
    laws = await router.invoke([{"role": "user", "content": prompt}], temperature=0.0)
    
    return {"fundamental_laws": laws.strip()}