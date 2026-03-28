import re
import json
from schemas.models import AgenticState
from utils.llm_router import safe_async_invoke
from memory.vector_store import retrieve_past_mistakes # Keep your memory!

async def propose_solution(state: AgenticState) -> dict:
    # 1. Increment Iteration
    iteration = state.get('iteration_count', 0) + 1
    print(f"\n--- [Layer 2] The Scientist is analyzing (Attempt {iteration}) ---")
    
    try:
        # 2. Extract context from the Universal Data Bus
        problem = state.get("problem_statement", "")
        domain = state.get("domain", "General Engineering") # NEW Grounding
        target_lang = state.get("target_language", "C/Python")
        
        # 3. Pull constraints from Physicist and Feedback from Analyst
        laws = state.get("fundamental_laws", "Standard STEM Principles") # NEW Grounding
        rca_feedback = state.get("audit_feedback", "None") # NEW: From the Analyst
        
        # 4. Pull past failures from Vector Memory
        past_warnings = retrieve_past_mistakes(problem)
        
        system_prompt = f"""
        Role: Lead STEM Researcher in {domain}.
        Expertise: Deep logical derivation and error-free implementation.
        Target Output: {target_lang}
        
        IMMUTABLE LAWS (Physicist's Constraints):
        {laws}
        
        PAST FAILURE LOGS (Avoid these patterns):
        {past_warnings}
        
        PREVIOUS ATTEMPT FEEDBACK:
        {rca_feedback}
        
        REASONING PROTOCOL:
        You must start every response with a <think> block.
        Inside:
        1. Assumptions Check: Identify parameters.
        2. Logical Derivation: Break the problem into atomic units.
        3. Law Verification: Cross-reference logic against the IMMUTABLE LAWS.
        
        CONSTRAINTS:
        - NO conversational filler.
        - Start immediately with <think>.
        - After </think>, provide the raw solution/code only.
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Solve this problem using the Reasoning Protocol:\n{problem}"}
        ]

        # 5. Network Call (Lane 1: Safe Lane)
        raw_output = await safe_async_invoke(messages, temperature=0.4)
        
        # 6. Data Integrity Cleaning (Your original Regex logic)
        clean_solution = raw_output.strip()
        if target_lang != "Agnostic/Math" and "```" in clean_solution:
            parts = clean_solution.split("```")
            if len(parts) >= 2:
                think_part = parts[0]
                code_part = re.sub(r"^[a-zA-Z]*\n", "", parts[1])
                clean_solution = f"{think_part}\n{code_part}".strip()

        print("[*] Solution proposed with grounded reasoning. Routing to Constitutional Judge...")
        
        return {
            "proposed_code": clean_solution,
            "iteration_count": iteration
        }
        
    except Exception as e:
        print(f"[!] Scientist Agent Failure: {e}")
        return {
            "proposed_code": f"<think>\nAPI Error encountered.\n</think>\n# SYSTEM API FAILURE: {e}",
            "iteration_count": iteration
        }