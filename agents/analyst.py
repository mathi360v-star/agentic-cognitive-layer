import json
import re
import ast
from langchain_core.messages import SystemMessage, HumanMessage
from schemas.models import AgenticState
# We use safe_async_invoke for speed, but you could use heavy_async_invoke for complex math
from utils.llm_router import safe_async_invoke 
from memory.vector_store import save_new_heuristic

async def analyze_failure(state: AgenticState) -> AgenticState:
    print("\n--- [Layer 4] The Analyst is performing Root Cause Analysis ---")
    
    try:
        # THE ENTERPRISE CONSTITUTIONAL PROMPT
        system_prompt = """
        Role: Principal AI Systems Forensic Engineer.
        Goal: Conduct a post-mortem on a failed engineering solution.
        
        CRITICAL: You must identify which Fundamental Law or Rule was violated.
        Output ONLY valid JSON. Keep descriptions dense and technical.
        
        {
            "mechanical_failure": "The specific logical/syntax bug.",
            "false_assumption": "Why the model thought this would work.",
            "law_violation": "Which Fundamental Law/Axiom was ignored?",
            "generalized_rule": "A high-level heuristic to prevent this forever."
        }
        """

        # We feed the analyst the laws from the Physicist
        laws = state.get("fundamental_laws", "Standard Engineering Principles")
        failed_code = state.get("proposed_code", "")
        
        human_prompt = f"""
        PROBLEM:
        {state.get('problem_statement')}

        FUNDAMENTAL LAWS TO OBEY:
        {laws}

        FAILED CODE SNAPSHOT:
        {failed_code}

        JURY VERDICT / ERROR:
        {state.get('red_team_critique')}
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]

        # Use the standard router for speed
        raw_response = await safe_async_invoke(messages, temperature=0.1)

        # Robust JSON Parsing
        match = re.search(r'\{[\s\S]*\}', raw_response)
        if not match: raise ValueError("Analyst output was not JSON.")
            
        clean_text = match.group(0)
        try:
            parsed_data = json.loads(clean_text)
        except json.JSONDecodeError:
            parsed_data = ast.literal_eval(clean_text)
        
        # --- CRITICAL DPO UPGRADE: Capture the failed code for the Sanitizer ---
        parsed_data["failed_code_snapshot"] = failed_code
        
        # Save to Vector Vault (ChromaDB) for RAG-based learning in Attempt 2
        save_new_heuristic(
            problem_statement=state.get("problem_statement", ""),
            flawed_assumption=parsed_data.get("false_assumption", "Unknown"),
            generalized_rule=parsed_data.get("generalized_rule", "Verify constraints.")
        )
        
        # Append to the history bus
        if "rca_history" not in state or state["rca_history"] is None:
            state["rca_history"] = []
            
        state["rca_history"].append(parsed_data)
        print(f"[*] RCA Complete. Violation: {parsed_data.get('law_violation', 'Logic')}. Trace saved to Vault.")
        
    except Exception as e:
        print(f"[!] Analyst Agent Failure: {e}. Skipping RCA memory update this round.")
        
    return state