import json
import re
import ast
from langchain_core.messages import SystemMessage, HumanMessage
from schemas.models import AgenticState
from utils.llm_router import safe_async_invoke

async def audit_problem(state: AgenticState) -> AgenticState:
    print("\n--- [Layer 1.5] The Verifier is auditing the problem ---")
    
    if not state.get("problem_statement") or len(state.get("problem_statement")) < 20:
        print("[-] Verifier REJECTED: Professor provided an empty or invalid problem.")
        state["problem_is_valid"] = False
        state["audit_feedback"] = "Problem statement is empty or too short."
        return state

    try:
        system_prompt = """
        Role: Elite Scientific Peer Reviewer. 
        Audit the problem for physical reality and logical correctness.
        Output ONLY valid JSON. Zero conversational text.
        
        {
            "is_valid": true,
            "flaw_reasoning": "Valid"
        }
        """

        human_prompt = f"Language: {state.get('target_language')}\nProblem: {state.get('problem_statement')}\nTests: {state.get('hidden_unit_tests')}"

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]

        # Network Call
        raw_response = await safe_async_invoke(messages, temperature=0.0)

        # Parsing
        match = re.search(r'\{[\s\S]*\}', raw_response)
        if not match: raise ValueError("No JSON object found.")
            
        clean_text = match.group(0)
        try:
            parsed_data = json.loads(clean_text)
        except json.JSONDecodeError:
            parsed_data = ast.literal_eval(clean_text)
        
        state["problem_is_valid"] = parsed_data.get("is_valid", False)
        state["audit_feedback"] = parsed_data.get("flaw_reasoning", "Unknown error.")
        
        if state["problem_is_valid"]:
            print("[+] Verifier APPROVED the problem.")
        else:
            print(f"[-] Verifier REJECTED. Reason: {state['audit_feedback']}")
            
    except Exception as e:
        print(f"[!] Verifier Agent Failure: {e}. Defaulting to rejection.")
        state["problem_is_valid"] = False
        state["audit_feedback"] = "System parsing or API error during verification."

    return state