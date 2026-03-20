import re
from schemas.models import AgenticState
from utils.llm_router import safe_async_invoke
from memory.vector_store import retrieve_past_mistakes

async def propose_solution(state: AgenticState) -> dict:
    iteration = state.get('iteration_count', 0) + 1
    print(f"\n--- [Layer 2] The Scientist is analyzing (Attempt {iteration}) ---")
    
    try:
        # 1. Pull the full context from the Universal Data Bus
        problem = state.get("problem_statement", "")
        topic = state.get("current_topic", "General Engineering")
        target_lang = state.get("target_language", "C/Python")
        
        # 2. Pull the constraints locked in by the Physicist
        fundamental_laws = state.get("fundamental_laws", "Standard logical principles.")
        
        # 3. Pull root cause analysis of past failures
        past_warnings = retrieve_past_mistakes(problem)
        
        system_prompt = f"""
        Role: Elite STEM Problem Solver. 
        Domain: {topic}
        Target Output Format: {target_lang}
        
        IMMUTABLE LAWS YOU MUST OBEY:
        {fundamental_laws}
        
        {past_warnings}
        
        Constraints:
        1. Output ONLY the raw solution. DO NOT include conversational text like "Here is the solution."
        2. If writing code, ensure it handles edge cases, memory safety, and constraints.
        3. If writing a mathematical/physics proof, use clear formal notation (LaTeX is accepted).
        """

        # Using standard dictionary format for messages which works universally across LangChain wrappers
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Solve:\n{problem}"}
        ]

        # Network Call (Using the Fast Lane router)
        raw_solution = await safe_async_invoke(messages, temperature=0.4)
        
        # Domain-Aware Markdown Destroyer
        if target_lang == "Agnostic/Math" or "Physics" in topic:
            # Do NOT strip markdown for Math/Physics to preserve LaTeX ($$, \begin{equation})
            clean_solution = raw_solution.strip()
        else:
            # Aggressively strip backticks for raw Code
            clean_solution = re.sub(r"^```[a-zA-Z]*\n", "", raw_solution, flags=re.MULTILINE)
            clean_solution = re.sub(r"```\n?", "", clean_solution, flags=re.MULTILINE).strip()
        
        print("[*] Solution proposed. Routing to Constitutional Judge...")
        
        # Return only the updated keys for LangGraph state management
        return {
            "proposed_code": clean_solution,
            "iteration_count": iteration
        }
        
    except Exception as e:
        print(f"[!] Scientist Agent Failure: {e}")
        return {
            "proposed_code": "# SYSTEM API FAILURE. Could not generate solution.",
            "iteration_count": iteration
        }