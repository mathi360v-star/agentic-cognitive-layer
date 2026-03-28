import re
from langchain_core.messages import SystemMessage, HumanMessage
from schemas.models import AgenticState
from utils.llm_router import safe_async_invoke, heavy_async_invoke

async def evaluate_code(state: AgenticState) -> AgenticState:
    print("\n--- [Layer 3] The Adversarial Jury is cross-examining ---")
    
    try:
        problem = state.get("problem_statement", "")
        proposed_solution = state.get("proposed_code", "")
        current_domain = state.get("domain", "General Engineering")
        laws = state.get("fundamental_laws", "Standard STEM Axioms")
        
        # ------------------------------------------------------------------
        # PHASE 1: PEDANTIC RED TEAM (Lane 1 - 8B/70B Mix)
        # ------------------------------------------------------------------
        red_team_prompt = f"""
        Role: Pedantic STEM Auditor. Your goal is to find a logical, mathematical, or physical flaw.
        DOMAIN: {current_domain}
        GROUNDING LAWS: {laws}
        
        CRITICAL CHECKS:
        1. Dimensional Analysis: Do the units match?
        2. Boundary Conditions: Does it fail at 0, Infinity, or NULL?
        3. Efficiency: Does it violate the required Big-O complexity?
        
        Output EXACTLY 2 sentences identifying the most likely flaw. If no flaw exists, invent a highly skeptical edge-case scenario.
        """
        
        print("[*] Red Teamer is hunting for vulnerabilities...")
        critique = await safe_async_invoke([
            SystemMessage(content=red_team_prompt),
            HumanMessage(content=f"Solution to Audit:\n{proposed_solution}")
        ], temperature=0.7)

        # ------------------------------------------------------------------
        # PHASE 2: SUPREME JUDGE (Lane 2 - Frontier Models Only)
        # ------------------------------------------------------------------
        judge_prompt = f"""
        Act as a Universal Constitutional AI Verifier. 
        Evaluate this solution for {current_domain}. 
        
        THE CONSTITUTION:
        Rule 1 (Logical Consistency): Internal math/logic must be perfect. No contradictions.
        Rule 2 (Constraint Adherence): Must meet every requirement in the prompt.
        Rule 3 (Dimensional & Physical Integrity): Units must be consistent. Physical laws ({laws}) must be obeyed.
        Rule 4 (Edge-Case Immunity): Must handle NULL, zero, empty sets, or overflow.
        Rule 5 (Adversarial Neutrality): Evaluate the Red Team Critique. If the Red Team is WRONG or being over-skeptical, you MUST REJECT their critique and favor the solution.

        PROBLEM: {problem}
        PROPOSED SOLUTION: {proposed_solution}
        RED TEAM CRITIQUE: {critique}

        FINAL VERDICT PROTOCOL:
        1. Analyze the solution against Rule 1-4.
        2. Specifically debunk or confirm the Red Team Critique.
        3. Output your final decision inside <VERDICT>FLAWLESS</VERDICT> or <VERDICT>REJECTED</VERDICT> tags.
        """
        
        print(f"[*] Supreme Judge is weighing evidence against the {current_domain} Constitution...")
        # Lane 2: 70B+ logic lane
        raw_verdict = await heavy_async_invoke([HumanMessage(content=judge_prompt)], temperature=0.0)

        # ------------------------------------------------------------------
        # PHASE 3: PARSING & ENTROPY LOGIC
        # ------------------------------------------------------------------
        verdict_match = re.search(r"<VERDICT>(.*?)</VERDICT>", raw_verdict, re.IGNORECASE)
        success = (verdict_match.group(1).strip().upper() == "FLAWLESS") if verdict_match else False

        state["execution_success"] = success
        
        if success:
            print("[+] SUCCESS! The Supreme Judge ruled the solution FLAWLESS.")
            
            # DYNAMIC DIFFICULTY (Entropy Proxy)
            iters = state.get("iteration_count", 0) + 1
            if iters == 1:
                difficulty = "Tier 1 (Foundational/Easy)"
            elif 2 <= iters <= 3:
                difficulty = "Tier 2 (Applied/Moderate)"
            else:
                difficulty = "Tier 3 (Edge-Case/Hard - High Entropy)"
                
            state.update({
                "final_correct_code": proposed_solution,
                "difficulty_tier": difficulty,
                "traceback": None
            })
        else:
            print(f"[-] FAILURE! Constitutional Violation detected by Judge.")
            # We strip the Verdict tag to keep the RCA clean for the Analyst
            clean_reasoning = re.sub(r"<VERDICT>.*?</VERDICT>", "", raw_verdict, flags=re.IGNORECASE).strip()
            state["traceback"] = f"RED TEAM: {critique}\nJUDGE ANALYSIS: {clean_reasoning[:1000]}"
            
    except Exception as e:
        print(f"[!] Evaluator Agent Failure: {e}")
        state.update({"execution_success": False, "traceback": f"System Crash in Jury: {e}"})

    return state