import json
import re
import ast
import random
from langchain_core.messages import SystemMessage, HumanMessage
from schemas.models import AgenticState
from utils.llm_router import safe_async_invoke

# --- STEP 1: THE PERSONA WRAPPER ---
def wrap_instruction(problem_text: str) -> str:
    """Wraps raw problems in different user personas to prevent instruction fragility."""
    if not problem_text:
        return ""
        
    personas = [
        f"Solve this engineering challenge: {problem_text}",
        f"Hey, I'm stuck on this. Can you walk me through the logic? {problem_text}",
        f"Analyze and provide a formal derivation for the following: {problem_text}",
        f"Please implement a solution for this task, focusing on efficiency: {problem_text}"
    ]
    return random.choice(personas)

# --- STEP 2: THE PROFESSOR NODE ---
async def generate_curriculum(state: AgenticState) -> AgenticState:
    print("\n--- [Layer 1] The Professor is designing a new curriculum ---")
    
    try:
        feedback = state.get("audit_feedback", "")
        rejection_warning = f"PREVIOUS PROBLEM REJECTED: {feedback}\nFix the flaws." if feedback else ""
        
        # 1. Multi-Domain Selection
        domains = [
            "Advanced Embedded C & RTOS",
            "Python AI & Machine Learning Algorithms",
            "Calculus & Differential Equations",
            "Quantum Mechanics & Applied Physics",
            "Boolean Algebra & Digital Logic Design",
            "Aerospace & Fluid Dynamics Mathematics"
        ]
        selected_domain = random.choice(domains)
        
        # Set hint for the sanitizer/physicist
        if any(math_word in selected_domain for math_word in ["Physics", "Calculus", "Mathematics"]):
            lang_hint = "Agnostic/Math"
        else:
            lang_hint = "C/Python"
        
        # 2. The Universal JSON Prompt
        system_prompt = f"""
        Role: Elite STEM Curriculum Architect.
        TARGET DOMAIN: {selected_domain}
        Assigned Topic: {state.get("current_topic", "General")}
        {rejection_warning}
        
        Task: Generate a complex reasoning problem. 
        Select a random difficulty tier:
        - 50% chance: Tier 1 (Foundational)
        - 30% chance: Tier 2 (Applied)
        - 20% chance: Tier 3 (Edge-Case)

        Constraints:
        - Output ONLY valid JSON. Zero conversational filler.
        
        {{
            "domain": "{selected_domain}",
            "target_language": "{lang_hint}",
            "problem_statement": "Detailed problem text here.",
            "difficulty_level": "Tier 1/2/3",
            "hidden_unit_tests": "Logic assertions."
        }}
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Generate problem. Attempt: {state.get('iteration_count', 0)}")
        ]

        raw_response = await safe_async_invoke(messages, temperature=0.8)

        # JSON Extraction logic
        match = re.search(r'\{[\s\S]*\}', raw_response)
        if not match: raise ValueError("No JSON dictionary found.")
        
        clean_text = match.group(0)
        try:
            parsed_data = json.loads(clean_text)
        except json.JSONDecodeError:
            parsed_data = ast.literal_eval(clean_text)
            
        # --- STEP 3: APPLY THE WRAPPER ---
        raw_problem = parsed_data.get("problem_statement", "")
        
        # We apply the persona wrapper here before it hits the state bus
        state["problem_statement"] = wrap_instruction(raw_problem)
        
        # Update State Bus
        state["domain"] = parsed_data.get("domain", selected_domain)
        state["target_language"] = parsed_data.get("target_language", lang_hint)
        state["difficulty_level"] = parsed_data.get("difficulty_level", "Tier 2")
        state["hidden_unit_tests"] = parsed_data.get("hidden_unit_tests", "")
        
        # Reset iteration-specific fields
        state["proposed_code"] = None 
        state["execution_success"] = False 
        state["problem_is_valid"] = False 
        
        print(f"[*] Generated {state['difficulty_level']} Problem in {state['domain']}.")
        
    except Exception as e:
        print(f"[!] Professor Agent Failure: {e}")
        state["problem_statement"] = "" 
        state["problem_is_valid"] = False

    return state