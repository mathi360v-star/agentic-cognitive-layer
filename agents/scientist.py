import re
from schemas.models import AgenticState
from utils.vector_vault import vault  # Integrated Long-Term Memory

async def propose_solution(state: AgenticState, router) -> dict:
    """
    The Scientist Node: Derives the solution while grounded in Laws 
    and Vector-Retrieved Memory (RAG).
    """
    
    # 1. Increment Iteration
    iteration = state.get('iteration_count', 0) + 1
    print(f"\n--- [Layer 2] The Scientist is analyzing (Attempt {iteration}) ---")
    
    try:
        # 2. Extract context from the Universal Data Bus
        problem = state.get("problem_statement", "")
        domain = state.get("domain", "STEM") 
        target_lang = state.get("target_language", "C/Python")
        
        # 3. Pull laws from Physicist and Feedback from Analyst/Judge
        laws = state.get("fundamental_laws", "Standard STEM Principles")
        rca_feedback = state.get("audit_feedback", "None") 
        
        # 4. RAG-AUGMENTED REASONING: Pull past failures from the Vector Vault
        # This prevents the model from repeating the same logical errors.
        past_errors = vault.retrieve_lessons(problem, domain)
        memory_context = "\n".join([f"- {err}" for err in past_errors]) if past_errors else "No previous similar failures recorded."
        
        # 5. Build the Grounded System Prompt
        system_prompt = f"""
        Role: Lead STEM Researcher in {domain}.
        Expertise: Deep logical derivation and error-free implementation.
        Target Output Format: {target_lang}
        
        IMMUTABLE LAWS (Physicist's Constraints):
        {laws}
        
        PAST FAILURE LOGS (CRITICAL: Avoid these specific mistakes):
        {memory_context}
        
        PREVIOUS ATTEMPT FEEDBACK (Fix these specific errors):
        {rca_feedback}
        
        REASONING PROTOCOL:
        You must start every response with a <think> block.
        Inside:
        1. Assumptions Check: Identify parameters and target constraints.
        2. Logical Derivation: Break the problem into atomic units/steps.
        3. Law Verification: Cross-reference every step against the IMMUTABLE LAWS.
        
        CONSTRAINTS:
        - NO conversational filler. Start directly with <think>.
        - After </think>, provide the raw solution or derivation only.
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Solve this problem using the Reasoning Protocol:\n{problem}"}
        ]

        # 6. Network Call (Using the Sharded Router for Lane 1)
        raw_output = await router.invoke(messages, temperature=0.4)
        
        # 7. Post-Processing Cleanup
        clean_solution = raw_output.strip()
        
        # If it's code, ensure we strip any markdown artifacts
        if target_lang != "Agnostic/Math" and "```" in clean_solution:
            clean_solution = re.sub(r"```[a-zA-Z]*\n", "", clean_solution)
            clean_solution = clean_solution.replace("```", "").strip()

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