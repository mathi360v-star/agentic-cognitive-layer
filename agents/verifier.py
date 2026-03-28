import json
import re
import asyncio
from langchain_core.messages import SystemMessage, HumanMessage
from schemas.models import AgenticState
from utils.llm_router import heavy_async_invoke 

async def node_verifier(state: AgenticState) -> AgenticState:
    """The Gatekeeper. Ensures STEM problems are logically sound and properly tiered."""
    print(f"\n--- [Layer 1.5] The Verifier is performing a High-IQ Audit on {state.get('domain', 'STEM')} ---")
    
    problem = state.get("problem_statement", "").strip()
    domain = state.get("domain", "General Engineering")
    tier = state.get("difficulty_tier", "Unknown")
    
    if not problem or len(problem) < 50:
        print("[-] Verifier REJECTED: Problem statement is too shallow for reasoning.")
        return {"problem_is_valid": False, "audit_feedback": "Problem statement too brief."}

    try:
        # THE V3 STEM CONSTITUTION (AUDIT VERSION)
        system_prompt = f"""
        Role: Principal STEM Auditor for Frontier AI Training.
        Expertise: {domain}
        
        Task: Perform an Epistemic Audit on the proposed {tier} problem.
        
        STRICT AUDIT RULES:
        1. NO PARADOXES: The problem must not contain contradictory constraints (e.g., "Sort in O(1) time").
        2. TIER ACCURACY: A {tier} problem must match that difficulty. If it's too easy, flag it.
        3. GROUNDING: The problem must be solvable using fundamental laws of {domain}.
        4. NO VAGUENESS: Variables and expected outputs must be explicitly defined.
        
        OUTPUT FORMAT (STRICT JSON ONLY):
        {{
            "is_valid": true/false,
            "flaw_reasoning": "Detailed explanation if false, 'Valid' if true",
            "suggested_tier": "Tier 1/2/3"
        }}
        """

        human_prompt = f"Problem to Audit:\n{problem}\n\nTarget Domain: {domain}\nStated Tier: {tier}"

        # LANE 2: We use the Heavy Router for guaranteed reasoning quality
        raw_response = await heavy_async_invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ], temperature=0.0)

        # --- ENTERPRISE RECOVERY LOGIC ---
        # 1. Strip markdown
        clean_json = re.sub(r'```json\s*|\s*```', '', raw_response).strip()
        # 2. Extract only the outermost JSON object
        match = re.search(r'(\{.*\})', clean_json, re.DOTALL)
        if not match:
            raise ValueError("No valid JSON structure identified.")
        
        parsed_data = json.loads(match.group(1))
        
        is_valid = parsed_data.get("is_valid", False)
        feedback = parsed_data.get("flaw_reasoning", "Unknown error.")
        actual_tier = parsed_data.get("suggested_tier", tier)

        if is_valid:
            print(f"[+] Verifier APPROVED {domain} ({actual_tier})")
            return {
                "problem_is_valid": True,
                "audit_feedback": "Valid",
                "difficulty_tier": actual_tier # Sync the tier to the actual complexity
            }
        else:
            print(f"[-] Verifier REJECTED: {feedback}")
            return {
                "problem_is_valid": False,
                "audit_feedback": feedback
            }
            
    except Exception as e:
        # If the 70B model fails to output JSON, we retry once with a simpler prompt or fail
        print(f"[!] Verifier Logic Error: {str(e)}. Defaulting to rejection.")
        return {
            "problem_is_valid": False,
            "audit_feedback": f"Verifier Syntax Error: {str(e)}"
        }