import json
import re
import ast
from langchain_core.messages import SystemMessage, HumanMessage
from schemas.models import AgenticState
from utils.llm_router import safe_async_invoke

async def generate_curriculum(state: AgenticState) -> AgenticState:
    print("\n--- [Layer 1] The Professor is designing a new curriculum ---")
    
    try:
        feedback = state.get("audit_feedback", "")
        rejection_warning = f"PREVIOUS PROBLEM REJECTED: {feedback}\nFix the flaws." if feedback else ""
        
        system_prompt = f"""
        Role: Elite Professor of Engineering & Computer Science.
        Assigned Topic: {state.get("current_topic", "General Engineering")}
        {rejection_warning}
        
        Constraints:
        - Generate an Elite difficulty problem strictly regarding the Assigned Topic.
        - Zero conversational filler. Output ONLY valid JSON.
        
        {{
            "domain": "Engineering",
            "target_language": "python",
            "problem_statement": "Detailed mathematical/engineering problem.",
            "difficulty_level": "Elite",
            "hidden_unit_tests": "assert function_name(input) == expected_output, 'Error'"
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
            
        state["domain"] = parsed_data.get("domain", "Computer Science")
        state["target_language"] = parsed_data.get("target_language", "python").lower()
        state["problem_statement"] = parsed_data.get("problem_statement", "")
        state["difficulty_level"] = parsed_data.get("difficulty_level", "Elite")
        state["hidden_unit_tests"] = parsed_data.get("hidden_unit_tests", "")
        
        state["proposed_code"] = None 
        state["execution_success"] = False 
        state["problem_is_valid"] = False 
        
        print(f"[*] Generated {state['difficulty_level']} Problem in {state['target_language'].upper()}.")
        
    except Exception as e:
        print(f"[!] Professor Agent Failure: {e}")
        state["problem_statement"] = "" # Forces immediate safe rejection by Layer 1.5
        state["problem_is_valid"] = False

    return state