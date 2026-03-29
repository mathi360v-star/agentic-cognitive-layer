import re
from langchain_core.messages import SystemMessage, HumanMessage
from schemas.models import AgenticState

async def evaluate_code(state: AgenticState, router) -> dict:
    """
    The Supreme Judge: Performs a Constitutional Audit of the solution.
    Uses the 'Heavy Lane' for maximum reasoning accuracy.
    """
    print("\n--- [Layer 3] The Supreme Judge is presiding over the Audit ---")
    
    try:
        problem = state.get("problem_statement", "")
        proposed_solution = state.get("proposed_code", "")
        current_domain = state.get("domain", "STEM")
        laws = state.get("fundamental_laws", "Standard STEM Axioms")
        
        # 1. The Constitutional Audit Prompt
        judge_prompt = f"""
        Act as a Universal Constitutional AI Verifier for the domain of {current_domain}. 
        
        THE CONSTITUTION:
        Rule 1 (Logical Consistency): Internal math/logic must be perfect. No contradictions.
        Rule 2 (Constraint Adherence): Must meet every requirement in the problem statement.
        Rule 3 (Physical Integrity): Physical laws ({laws}) must be strictly obeyed.
        Rule 4 (Edge-Case Immunity): Must handle NULL, zero, or boundary conditions.

        PROBLEM: 
        {problem}

        PROPOSED SOLUTION: 
        {proposed_solution}

        FINAL VERDICT PROTOCOL:
        1. Analyze the solution against Rule 1-4.
        2. Output your final decision inside <VERDICT>FLAWLESS</VERDICT> or <VERDICT>REJECTED</VERDICT> tags.
        """
        
        print(f"[*] Supreme Judge is weighing evidence against the {current_domain} Constitution...")
        
        # 2. LANE 2: We use 'heavy=True' to trigger the 70B+ logic lane in your ShardedRouter
        raw_verdict = await router.invoke(
            [{"role": "user", "content": judge_prompt}], 
            temperature=0.0, 
            heavy=True
        )

        # 3. Parsing the Decision
        verdict_match = re.search(r"<VERDICT>(.*?)</VERDICT>", raw_verdict, re.IGNORECASE)
        success = (verdict_match.group(1).strip().upper() == "FLAWLESS") if verdict_match else False

        if success:
            print("[+] SUCCESS! The Supreme Judge ruled the solution FLAWLESS.")
            
            # Determine Difficulty Tier based on how many tries it took (Entropy Proxy)
            iters = state.get("iteration_count", 0) + 1
            if iters == 1:
                difficulty = "Tier 1 (Foundational)"
            elif 2 <= iters <= 3:
                difficulty = "Tier 2 (Applied)"
            else:
                difficulty = "Tier 3 (Edge-Case/High-Entropy)"
                
            return {
                "execution_success": True,
                "final_correct_code": proposed_solution,
                "difficulty_tier": difficulty,
                "audit_feedback": "Valid"
            }
        else:
            print(f"[-] FAILURE! Constitutional Violation detected by Judge.")
            # Strip tags to keep the raw reasoning for the Analyst to study
            clean_reasoning = re.sub(r"<VERDICT>.*?</VERDICT>", "", raw_verdict, flags=re.IGNORECASE).strip()
            return {
                "execution_success": False, 
                "audit_feedback": clean_reasoning[:1500] 
            }
            
    except Exception as e:
        print(f"[!] Supreme Judge System Failure: {e}")
        return {
            "execution_success": False, 
            "audit_feedback": f"Jury System Crash: {str(e)}"
        }