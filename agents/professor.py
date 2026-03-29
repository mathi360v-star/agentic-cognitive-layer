import json, re, random
from schemas.models import AgenticState

def wrap_instruction(problem_text):
    personas = [
        f"Solve this engineering challenge: {problem_text}",
        f"Analyze and provide a formal derivation for: {problem_text}",
        f"I'm stuck on this technical task, walk me through the logic: {problem_text}",
        f"Implement a high-efficiency solution for: {problem_text}"
    ]
    return random.choice(personas)

async def generate_curriculum(state: AgenticState, router):
    print("\n--- [Layer 1] The Professor is designing a new curriculum ---")
    
    # 1. State-Safe Tier Selection
    tier_roll = random.random()
    assigned_tier = "Tier 1 (Foundational)"
    if 0.70 < tier_roll < 0.95: assigned_tier = "Tier 2 (Applied)"
    elif tier_roll >= 0.95: assigned_tier = "Tier 3 (Edge-Case/Elite)"

    # 2. Domain Selection
    domains = [
        "Advanced Embedded C & RTOS", "Python AI & RL", 
        "Calculus & Differential Equations", "Quantum Mechanics",
        "Boolean Algebra & Digital Logic", "Aerospace & Fluid Dynamics"
    ]
    selected_domain = random.choice(domains)
    lang_hint = "Agnostic/Math" if any(x in selected_domain for x in ["Calculus", "Quantum", "Dynamics"]) else "C/Python"

    system_prompt = f"""
    Role: Elite STEM Curriculum Architect.
    Domain: {selected_domain} | Difficulty: {assigned_tier}
    Task: Generate a complex reasoning problem in JSON format.
    Format: {{"problem_statement": "...", "target_language": "{lang_hint}"}}
    """

    try:
        raw = await router.invoke([{"role": "system", "content": system_prompt}], temperature=0.8)
        match = re.search(r'\{[\s\S]*\}', raw)
        data = json.loads(match.group(0))
        
        return {
            "problem_statement": wrap_instruction(data["problem_statement"]),
            "domain": selected_domain,
            "difficulty_tier": assigned_tier,
            "target_language": lang_hint,
            "problem_is_valid": True,
            "iteration_count": 0
        }
    except Exception as e:
        print(f"[!] Professor Failure: {e}")
        return {"problem_is_valid": False, "audit_feedback": "Generation Failed."}