from schemas.models import AgenticState
from utils.llm_router import safe_async_invoke

async def node_physicist(state: AgenticState):
    """Extracts immutable laws to prevent hallucinated solutions."""
    print("--- [Layer 1.7] The Physicist is locking fundamental laws ---")
    
    # We safely use .get() to prevent KeyErrors
    problem = state.get('problem_statement', '')
    
    prompt = f"""
    Analyze the following engineering problem:
    {problem}
    
    Identify the strict fundamental mathematical, physical, or computer science laws required to solve this.
    For example: 'Kirchhoffs Voltage Law', 'Time Complexity O(log n)', 'POSIX Thread Mutex Semantics'.
    Output ONLY a bulleted list of 2 to 4 laws. Do not solve the problem.
    """
    
    laws = await safe_async_invoke([{"role": "user", "content": prompt}])
    
    # Safely return the new variable to the Universal Bus
    return {"fundamental_laws": laws.strip()}