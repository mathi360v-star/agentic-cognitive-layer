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
        
        # ------------------------------------------------------------------
        # PHASE 1: RED TEAM
        # ------------------------------------------------------------------
        red_team_prompt = """
        Role: Ruthless Security Auditor. Prove the solution is flawed.
        Check edge cases, time complexity, and physical/math violations.
        CRITICAL: Output STRICTLY 1 to 3 sentences maximum. Be concise.
        """
        
        rt_messages = [
            SystemMessage(content=red_team_prompt),
            HumanMessage(content=f"Problem: {problem}\nSolution:\n{proposed_solution}")
        ]
        
        print("[*] Red Teamer is hunting for vulnerabilities...")
        critique = await safe_async_invoke(rt_messages, temperature=0.7)
        state["red_team_critique"] = critique

        # ------------------------------------------------------------------
        # PHASE 2: SUPREME JUDGE (Universal Constitution)
        # ------------------------------------------------------------------
        judge_prompt = f"""
        Act as a Universal Constitutional AI Verifier. You are evaluating a proposed solution in the domain of: {current_domain}.
        
        THE CONSTITUTION:
        Rule 1 (Logical Consistency): The solution must be internally consistent. Math must compute perfectly; code must compile mentally.
        Rule 2 (Constraint Satisfaction): The solution must strictly adhere to EVERY constraint requested in the prompt.
        Rule 3 (Physical/Mathematical Reality): The solution must not violate fundamental laws of physics, formal mathematics, or computer science theories.
        Rule 4 (Boundary Immunity): The solution must explicitly account for edge cases (e.g., zero-division, NULL pointers).

        THE PROBLEM: 
        {problem}
        
        THE PROPOSED SOLUTION: 
        {proposed_solution}
        
        RED TEAM CRITIQUE TO CONSIDER:
        {critique}

        EVALUATION PROTOCOL:
        1. Evaluate Rule 1.
        2. Evaluate Rule 2.
        3. Evaluate Rule 3.
        4. Evaluate Rule 4.

        FINAL VERDICT:
        You MUST conclude your response with a strict XML tag containing your final decision.
        If ALL rules are satisfied and the Red Team critique is mitigated, output: <VERDICT>FLAWLESS</VERDICT>
        If ANY rule is violated, output: <VERDICT>REJECTED</VERDICT>
        """
        
        judge_messages = [
            HumanMessage(content=judge_prompt)
        ]
        
        print(f"[*] Supreme Judge is weighing the evidence against the {current_domain} Constitution...")
        
        # THE ENTERPRISE FIX: We force the Judge to use the Heavy Lane (70B+ models) 
        # to guarantee high-IQ verification of complex math and physics.
        raw_verdict = await heavy_async_invoke(judge_messages, temperature=0.1)

        # ------------------------------------------------------------------
        # PHASE 3: STRICT XML PARSER & ENTROPY ROUTER
        # ------------------------------------------------------------------
        # Use Regex to hunt specifically for the XML tags to prevent false positives/negatives
        verdict_match = re.search(r"<VERDICT>(.*?)</VERDICT>", raw_verdict, re.IGNORECASE)
        
        # Default to False if the LLM hallucinated the tags entirely
        success = False 
        if verdict_match:
            extracted_verdict = verdict_match.group(1).strip().upper()
            if extracted_verdict == "FLAWLESS":
                success = True

        state["execution_success"] = success
        
        if success:
            print("[+] SUCCESS! The Supreme Judge ruled the solution FLAWLESS.")
            state["traceback"] = None
            
            # THE ENTROPY CALCULATOR
            iters = state.get("iteration_count", 1)
            if iters == 1:
                difficulty = "Tier 1 (Foundational/Easy)"
            elif iters == 2 or iters == 3:
                difficulty = "Tier 2 (Applied/Moderate)"
            else:
                difficulty = "Tier 3 (Edge-Case/Hard - High Entropy)"
                
            state["final_correct_code"] = proposed_solution
            state["difficulty_tier"] = difficulty
            
        else:
            print(f"[-] FAILURE! Constitutional Violation Detected.")
            
            # Clean up the output to save cleanly into our RCA trace
            reasoning = raw_verdict.replace("\n", " ").strip()
            # Remove the XML tag from the traceback so the Analyst doesn't get confused
            reasoning = re.sub(r"<VERDICT>.*?</VERDICT>", "", reasoning, flags=re.IGNORECASE).strip()
            
            if len(reasoning) > 300:
                reasoning = reasoning[:297] + "..."
                
            state["traceback"] = f"Red Team Critique: {critique}\nJudge Evaluation: {reasoning}"
            
    except Exception as e:
        print(f"[!] Evaluator Agent Failure: {e}. Defaulting to rejection.")
        state["execution_success"] = False
        state["traceback"] = f"System Error during Jury AI evaluation: {e}"

    return state