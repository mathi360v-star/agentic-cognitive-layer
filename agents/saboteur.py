import random
from schemas.models import AgenticState
from utils.llm_router import safe_async_invoke

async def node_saboteur(state: AgenticState):
    """Intentionally introduces a subtle, 'plausible' flaw for DPO data."""
    print("--- [Layer 2.5] The Saboteur is engineering a plausible failure ---")
    
    clean_solution = state.get("final_correct_code", state.get("proposed_code", ""))
    domain = state.get("domain", "STEM")
    
    # We want a plausible error, like a sign flip or a off-by-one error
    prompt = f"""
    Act as an Adversarial STEM Engineer. You are given a CORRECT solution in the domain of {domain}.
    
    THE CORRECT SOLUTION:
    {clean_solution}
    
    TASK:
    Create a REJECTED version of this solution. Introduce exactly ONE subtle, high-IQ flaw.
    - If it's Math: Change a sign (e.g., $+$ to $-$) or an exponent.
    - If it's Code: Introduce a subtle memory leak, a race condition, or an off-by-one error in a loop.
    - If it's Physics: Violate a single conservation law (e.g., Energy or Momentum) in the derivation.
    
    Output ONLY the flawed solution text. No explanations.
    """
    
    # We use a lower temperature for the saboteur to keep it focused
    rejected_code = await safe_async_invoke([{"role": "user", "content": prompt}], temperature=0.5)
    
    # Save this to the rca_history so the Sanitizer finds it as a 'Rejected' sample
    return {
        "rca_history": [{
            "failed_code_snapshot": rejected_code.strip(),
            "mechanical_failure": "Adversarial DPO Flaw.",
            "generalized_rule": "Initial logic contained a subtle systemic trap."
        }]
    }