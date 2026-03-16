import json
import re
import ast
from langchain_core.messages import SystemMessage, HumanMessage
from schemas.models import AgenticState
from utils.llm_router import safe_async_invoke
from memory.vector_store import save_new_heuristic

async def analyze_failure(state: AgenticState) -> AgenticState:
    print("\n--- [Layer 4] The Analyst is performing Root Cause Analysis ---")
    
    try:
        system_prompt = """
        Role: Elite AI Systems Analyst.
        The Scientist's code failed the Adversarial Jury.
        Output ONLY valid JSON. Keep descriptions under 15 words.
        
        {
            "mechanical_failure": "Brief algorithmic flaw.",
            "false_assumption": "Brief incorrect assumption.",
            "generalized_rule": "Brief, domain-agnostic heuristic."
        }
        """

        human_prompt = f"Problem:\n{state.get('problem_statement')}\n\nFailed Code:\n{state.get('proposed_code')}\n\nJury Verdict:\n{state.get('traceback')}"

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]

        # Network Call
        raw_response = await safe_async_invoke(messages, temperature=0.1)

        # Parsing
        match = re.search(r'\{[\s\S]*\}', raw_response)
        if not match: raise ValueError("No JSON object found.")
            
        clean_text = match.group(0)
        try:
            parsed_data = json.loads(clean_text)
        except json.JSONDecodeError:
            parsed_data = ast.literal_eval(clean_text)
        
        # Save to ChromaDB
        save_new_heuristic(
            problem_statement=state.get("problem_statement", ""),
            flawed_assumption=parsed_data.get("false_assumption", "Unknown assumption"),
            generalized_rule=parsed_data.get("generalized_rule", "Verify edge cases.")
        )
        
        if "rca_history" not in state: state["rca_history"] = []
        state["rca_history"].append(parsed_data)
        print("[*] Root Cause Analysis complete. Lesson saved to Vault.")
        
    except Exception as e:
        print(f"[!] Analyst Agent Failure: {e}. Skipping RCA memory update this round.")
        
    return state