import re
from langchain_core.messages import SystemMessage, HumanMessage
from schemas.models import AgenticState
from utils.llm_router import safe_async_invoke
from memory.vector_store import retrieve_past_mistakes

async def propose_solution(state: AgenticState) -> AgenticState:
    print(f"\n--- [Layer 2] The Scientist is analyzing (Attempt {state.get('iteration_count', 0) + 1}) ---")
    
    try:
        problem = state.get("problem_statement", "")
        topic = state.get("current_topic", "General Engineering")
        past_warnings = retrieve_past_mistakes(problem)
        
        system_prompt = f"""
        Role: Elite Engineering AI. 
        Domain: {topic}
        {past_warnings}
        
        Constraints:
        1. Output ONLY the raw solution (code or math).
        2. DO NOT use markdown backticks (```). 
        3. DO NOT include conversational text.
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Solve:\n{problem}")
        ]

        # Network Call
        raw_code = await safe_async_invoke(messages, temperature=0.4)
        
        # Aggressive Markdown Destroyer
        clean_code = re.sub(r"^```[a-zA-Z]*\n", "", raw_code, flags=re.MULTILINE)
        clean_code = re.sub(r"```\n?", "", clean_code, flags=re.MULTILINE).strip()
        
        state["proposed_code"] = clean_code
        print("[*] Solution proposed. Routing to Adversarial Jury...")
        
    except Exception as e:
        print(f"[!] Scientist Agent Failure: {e}")
        state["proposed_code"] = "# SYSTEM API FAILURE. Could not generate solution."
        
    state["iteration_count"] = state.get("iteration_count", 0) + 1
    return state