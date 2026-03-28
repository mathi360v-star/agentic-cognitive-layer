import json
import re
from schemas.models import AgenticState
from utils.llm_router import safe_async_invoke 
# Assuming you want to keep your RAG memory for Attempt 2
from memory.vector_store import save_new_heuristic

async def analyze_failure(state: AgenticState) -> dict:
    """Performs STEM Root Cause Analysis on logic, math, or physics failures."""
    print("--- [Layer 4] The Analyst is performing STEM Root Cause Analysis ---")
    
    # 1. Extraction from Universal Bus
    problem = state.get("problem_statement", "")
    failed_logic = state.get("proposed_code", "")
    # Use audit_feedback or red_team_critique depending on which one was populated
    judge_critique = state.get("audit_feedback", state.get("red_team_critique", "Logic error."))
    laws = state.get("fundamental_laws", "Standard STEM Principles")
    
    # 2. The STEM Post-Mortem Prompt
    prompt = f"""
    You are a Formal Logic Analyst. A STEM solution has FAILED the Supreme Judge's audit.
    
    DOMAIN: {state.get('domain', 'Engineering')}
    PROBLEM: {problem}
    LAWS THAT MUST BE OBEYED: {laws}
    
    FAILED SOLUTION: 
    {failed_logic}
    
    JUDGE'S CRITIQUE: 
    {judge_critique}
    
    TASK:
    Identify exactly why the logic failed. Was it a mathematical contradiction? A violation of a physical law? Or a code-level bug?
    
    Output your analysis in this format:
    FAILURE: [Point of logical breakdown]
    RULE: [Specific instruction to the Scientist for the next attempt]
    """
    
    # Lane 1: Safe Router
    rca_raw = await safe_async_invoke([{"role": "user", "content": prompt}])
    
    # 3. Clean and Extract the Heuristic
    failure_desc = rca_raw.split("RULE:")[0].replace("FAILURE:", "").strip()
    rule_desc = rca_raw.split("RULE:")[-1].strip() if "RULE:" in rca_raw else "Refine logic against laws."

    # 4. Memory Injection (ChromaDB Vector Store)
    try:
        save_new_heuristic(
            problem_statement=problem,
            flawed_assumption=failure_desc,
            generalized_rule=rule_desc
        )
    except Exception as e:
        print(f"[!] Vector Vault Sync Failed: {e}")

    # 5. DPO Snapshot Construction
    new_rca = {
        "failed_code_snapshot": failed_logic,
        "mechanical_failure": failure_desc,
        "generalized_rule": rule_desc
    }
    
    # Update history and state
    history = state.get("rca_history", [])
    if history is None: history = []
    history.append(new_rca)
    
    # Return to Bus
    return {
        "rca_history": history, 
        "iteration_count": state.get("iteration_count", 0) + 1,
        "audit_feedback": f"FAILURE ANALYSIS: {failure_desc}\nREQUIRED FIX: {rule_desc}"
    }