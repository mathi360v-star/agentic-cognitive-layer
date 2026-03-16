import json
import re
import ast
from langchain_core.messages import SystemMessage, HumanMessage
from schemas.models import AgenticState
from utils.llm_router import safe_async_invoke

async def evaluate_code(state: AgenticState) -> AgenticState:
    print("\n--- [Layer 3] The Adversarial Jury is cross-examining ---")
    
    try:
        problem = state.get("problem_statement")
        proposed_solution = state.get("proposed_code")
        
        # PHASE 1: RED TEAM
        red_team_prompt = """
        Role: Ruthless Security Auditor. Prove the solution is flawed.
        Check edge cases, time complexity, and physical math violations.
        CRITICAL: Output STRICTLY 1 to 3 sentences maximum. Be concise.
        """
        
        rt_messages = [
            SystemMessage(content=red_team_prompt),
            HumanMessage(content=f"Problem: {problem}\nSolution:\n{proposed_solution}")
        ]
        
        print("[*] Red Teamer is hunting for vulnerabilities...")
        critique = await safe_async_invoke(rt_messages, temperature=0.7)
        state["red_team_critique"] = critique

        # PHASE 2: SUPREME JUDGE
        judge_prompt = """
        Role: Supreme Judge. Evaluate the solution against the Red Team's attack.
        Output ONLY valid JSON. Use double quotes.
        {
            "is_successful": true,
            "final_verdict_reasoning": "Concise explanation."
        }
        """
        
        judge_messages = [
            SystemMessage(content=judge_prompt),
            HumanMessage(content=f"Problem: {problem}\nSolution: {proposed_solution}\nRed Team: {critique}")
        ]
        
        print("[*] Supreme Judge is weighing the evidence...")
        raw_verdict = await safe_async_invoke(judge_messages, temperature=0.1)

        # PHASE 3: PARSER
        match = re.search(r'\{[\s\S]*\}', raw_verdict)
        if not match: raise ValueError("No dictionary found.")
        
        clean_text = match.group(0)
        try:
            parsed_data = json.loads(clean_text)
        except json.JSONDecodeError:
            parsed_data = ast.literal_eval(clean_text)
            
        success = parsed_data.get("is_successful", False)
        reasoning = parsed_data.get("final_verdict_reasoning", "Unknown.")
        
        state["execution_success"] = success
        if success:
            print("[+] SUCCESS! The Supreme Judge ruled the solution FLAWLESS.")
            state["traceback"] = None
        else:
            print(f"[-] FAILURE! Jury Verdict: {reasoning}")
            state["traceback"] = f"Red Team: {critique}\nVerdict: {reasoning}"
            
    except Exception as e:
        print(f"[!] Evaluator Agent Failure: {e}. Defaulting to rejection.")
        state["execution_success"] = False
        state["traceback"] = f"System Error during Jury AI evaluation: {e}"

    return state