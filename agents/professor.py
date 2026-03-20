import json
import re
import ast
import random
from langchain_core.messages import SystemMessage, HumanMessage
from schemas.models import AgenticState
from utils.llm_router import safe_async_invoke

async def generate_curriculum(state: AgenticState) -> AgenticState:
    print("\n--- [Layer 1] The Professor is designing a new curriculum ---")
    
    try:
        feedback = state.get("audit_feedback", "")
        rejection_warning = f"PREVIOUS PROBLEM REJECTED: {feedback}\nFix the flaws." if feedback else ""
        
        # 1. The Multi-Domain Selector
        domains = [
            "Advanced Embedded C & RTOS",
            "Python AI & Machine Learning Algorithms",
            "Calculus & Differential Equations",
            "Quantum Mechanics & Applied Physics",
            "Boolean Algebra & Digital Logic Design",
            "Aerospace & Fluid Dynamics Mathematics"
        ]
        selected_domain = random.choice(domains)
        
        # Determine the target language based on the domain
        if "Physics" in selected_domain or "Calculus" in selected_domain or "Mathematics" in selected_domain:
            lang_hint = "Agnostic/Math"
        else:
            lang_hint = "C/Python"
        
        # 2. The Universal JSON Prompt
        system_prompt = f"""
        Role: Elite STEM Curriculum Architect.
        TARGET DOMAIN: {selected_domain}
        Assigned Topic (Optional Guide): {state.get("current_topic", "General")}
        {rejection_warning}
        
        Crucially, select a random difficulty tier for this problem:
        - 50% chance: Tier 1 (Foundational) - Requires standard application of laws/formulas.
        - 30% chance: Tier 2 (Applied) - Requires bridging two distinct concepts.
        - 20% chance: Tier 3 (Edge-Case) - High complexity, deep logical optimization, or paradox.

        If the domain is coding, focus on algorithmic logic, memory, or concurrency.
        If the domain is physics/math, focus on formal proofs, derivations, or complex physical system modeling.
        
        Constraints:
        - Zero conversational filler. Output ONLY valid JSON.
        - Do NOT solve the problem in the output.
        
        {{
            "domain": "{selected_domain}",
            "target_language": "{lang_hint}",
            "problem_statement": "Detailed mathematical/engineering problem text here.",
            "difficulty_level": "Tier 1/2/3",
            "hidden_unit_tests": "Mathematical proofs or logic assertions to verify the answer later."
        }}
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Generate problem. Attempt: {state.get('iteration_count', 0)}")
        ]

        # Network Call
        raw_response = await safe_async_invoke(messages, temperature=0.8)

        # Parsing
        match = re.search(r'\{[\s\S]*\}', raw_response)
        if not match: raise ValueError("No JSON dictionary found.")
        
        clean_text = match.group(0)
        try:
            parsed_data = json.loads(clean_text)
        except json.JSONDecodeError:
            parsed_data = ast.literal_eval(clean_text)
            
        # Update State Safely
        state["domain"] = parsed_data.get("domain", selected_domain)
        state["target_language"] = parsed_data.get("target_language", lang_hint)
        state["problem_statement"] = parsed_data.get("problem_statement", "")
        state["difficulty_level"] = parsed_data.get("difficulty_level", "Tier 2")
        state["hidden_unit_tests"] = parsed_data.get("hidden_unit_tests", "")
        
        # Reset evaluation flags for the new problem
        state["proposed_code"] = None 
        state["execution_success"] = False 
        state["problem_is_valid"] = False 
        
        print(f"[*] Generated {state['difficulty_level']} Problem in {state['domain']}.")
        
    except Exception as e:
        print(f"[!] Professor Agent Failure: {e}")
        state["problem_statement"] = "" # Forces immediate safe rejection by Layer 1.5
        state["problem_is_valid"] = False

    return state