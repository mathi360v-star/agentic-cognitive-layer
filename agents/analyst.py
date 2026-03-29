import uuid
import re
from schemas.models import AgenticState
from utils.vector_vault import vault 

async def analyze_failure(state: AgenticState, router) -> dict:
    """
    Performs STEM-focused Root Cause Analysis.
    Saves logical lessons to the Vector Vault to prevent repetitive hallucinations.
    """
    print("\n--- [Layer 4] The Analyst is performing STEM Root Cause Analysis ---")
    
    # 1. Extraction from Universal Data Bus
    problem = state.get("problem_statement", "")
    failed_logic = state.get("proposed_code", "")
    judge_critique = state.get("audit_feedback", "General logic failure.")
    domain = state.get("domain", "STEM")
    laws = state.get("fundamental_laws", "Standard Axioms")
    
    # 2. The Logic-Grounded RCA Prompt
    # We force the Analyst to identify the specific law or theorem violated.
    prompt = f"""
    You are a Formal Logic Analyst. A STEM solution has FAILED the Supreme Judge's audit.
    
    DOMAIN: {domain}
    PROBLEM: {problem}
    GROUNDING LAWS: {laws}
    
    FAILED SOLUTION: 
    {failed_logic}
    
    JUDGE'S CRITIQUE: 
    {judge_critique}
    
    TASK:
    Identify the 'Mechanical Failure' (Where exactly did the math or logic break?).
    Provide a 'Generalized Rule' to ensure the Scientist does not repeat this error.
    
    Format your response EXACTLY like this:
    FAILURE: [Description of the specific logical or physical breakdown]
    RULE: [Direct instruction for the next attempt]
    """
    
    # Lane 1: Use the standard router for the analysis
    rca_raw = await router.invoke([{"role": "user", "content": prompt}], temperature=0.0)
    
    # 3. Memory Injection (The RAG Step)
    # We save this to the Vector Vault so the Scientist can 'Search' for it in the future.
    try:
        lesson_id = f"rca_{uuid.uuid4().hex[:8]}"
        # We store the raw analysis so the Scientist gets the full context
        vault.store_lesson(
            lesson_text=f"Domain: {domain} | {rca_raw}", 
            domain=domain
        )
        print(f"[+] Memory embedded in Vector Vault: {lesson_id}")
    except Exception as e:
        print(f"[!] Vector Vault Sync Failed: {e}")

    # 4. State Update & DPO Snapshot
    # This history allows the sanitizer to build 'Rejected' samples.
    new_rca_entry = {
        "failed_code_snapshot": failed_logic,
        "mechanical_failure": rca_raw.split("RULE:")[0].replace("FAILURE:", "").strip() if "RULE:" in rca_raw else rca_raw,
        "generalized_rule": rca_raw.split("RULE:")[-1].strip() if "RULE:" in rca_raw else "Refine logic."
    }
    
    history = state.get("rca_history", [])
    if history is None: history = []
    history.append(new_rca_entry)
    
    # 5. Return to the Bus
    # We increment the iteration_count here to tell the graph how many attempts have passed.
    return {
        "rca_history": history, 
        "iteration_count": state.get("iteration_count", 0) + 1,
        "audit_feedback": f"RCA LOG: {rca_raw}" 
    }