import random
from schemas.models import AgenticState

async def node_saboteur(state: AgenticState, router):
    """
    Master Adversarial Node: Creates a 'Rejected' version of a perfect solution.
    This provides the 'Distractor' needed for DPO (Preference) training.
    """
    domain = state.get("domain", "General STEM")
    language = state.get("target_language", "Logic")
    
    print(f"--- [Layer 2.5] The Saboteur is engineering a DPO distractor for {domain} ---")

    # 1. Extraction with Safety Guards (Prevents crashes if the solution is missing)
    solution = state.get("final_correct_code") or state.get("proposed_code")
    
    if not solution or len(solution) < 20:
        print("[-] Saboteur Aborted: No valid solution found to sabotage.")
        return {"rca_history": []}

    # 2. Multi-Domain Strategic Sabotage Logic
    # We tailor the 'Type of Flaw' based on the domain to ensure high-quality DPO data
    if language == "Agnostic/Math":
        specific_flaw_instruction = (
            "Introduce a subtle mathematical error: flip a sign (+/-), "
            "miss an integration constant (+C), or swap a numerator/denominator."
        )
    else:
        specific_flaw_instruction = (
            "Introduce a subtle technical bug: an off-by-one error in a loop, "
            "a memory leak, a race condition, or a logic inversion (using < instead of >)."
        )

    system_prompt = f"""
    Role: Master Adversarial STEM Engineer.
    Expertise: {domain}
    
    TASK: You are given a CORRECT solution. You must create a 'REJECTED' version for an AI training dataset.
    
    INSTRUCTION: {specific_flaw_instruction}
    
    CRITICAL RULES:
    1. The flaw must be SUBTLE. It should look 95% correct to a human.
    2. Output ONLY the flawed solution text. 
    3. DO NOT add explanations, labels, or triple backticks unless they are part of the solution.
    """

    # 3. Execution via the Sharded Router
    try:
        rejected = await router.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Correct Solution to Sabotage:\n{solution}"}
        ], temperature=0.7)

        # 4. Return to the Data Bus formatted for the Sanitizer
        # We store it in 'rca_history' so the sanitizer automatically finds it as the 'Rejected' sample.
        return {
            "rca_history": [{
                "failed_code_snapshot": rejected.strip(),
                "mechanical_failure": f"Engineered Adversarial {domain} Flaw",
                "generalized_rule": "Initial logic contained a subtle systemic trap."
            }]
        }
        
    except Exception as e:
        print(f"[!] Saboteur Failure: {e}")
        return {"rca_history": []}