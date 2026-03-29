import argparse, asyncio, json, os, random, time
from langgraph.graph import StateGraph, END
from schemas.models import AgenticState
from utils.llm_router import ShardedRouter
from utils.vector_vault import vault

# Import agents
from agents.professor import generate_curriculum
from agents.epistemic_eval import node_epistemic_evaluator
from agents.verifier import node_verifier
from agents.physicist import node_physicist
from agents.scientist import propose_solution
from agents.evaluator import evaluate_code
from agents.analyst import analyze_failure
from agents.saboteur import node_saboteur

def build_agentic_graph(router: ShardedRouter):
    workflow = StateGraph(AgenticState)

    # --- NODE WRAPPER: Fixes the 'missing 1 required positional argument' error ---
    async def wrap_node(func, state):
        # This ensures the router is passed but the graph only sees (state)
        return await func(state, router)

    # 1. Register Nodes
    workflow.add_node("Professor", lambda s: wrap_node(generate_curriculum, s))
    workflow.add_node("Epistemic", lambda s: wrap_node(node_epistemic_evaluator, s))
    workflow.add_node("Verifier", lambda s: wrap_node(node_verifier, s))
    workflow.add_node("Physicist", lambda s: wrap_node(node_physicist, s))
    workflow.add_node("Scientist", lambda s: wrap_node(propose_solution, s))
    workflow.add_node("Evaluator", lambda s: wrap_node(evaluate_code, s))
    workflow.add_node("Analyst", lambda s: wrap_node(analyze_failure, s))
    workflow.add_node("Saboteur", lambda s: wrap_node(node_saboteur, s))

    # 2. Logic Gates (The Safety Circuit)
    def check_safety_and_solvability(state: AgenticState):
        steps = state.get("total_graph_steps", 0)
        state["total_graph_steps"] = steps + 1
        if steps >= 15: return "END"
        if not state.get("problem_is_valid"): return "Professor"
        return "Verifier"

    def check_audit_status(state: AgenticState):
        if state.get("problem_is_valid"): return "Physicist"
        return "Professor"

    def check_execution(state: AgenticState):
        if state.get("execution_success"):
            return "Saboteur" if state.get("iteration_count", 0) <= 1 else "END"
        if state.get("iteration_count", 0) >= 4: return "END"
        return "Analyst"

    # 3. Define Flow
    workflow.set_entry_point("Professor")
    workflow.add_edge("Professor", "Epistemic")
    workflow.add_conditional_edges("Epistemic", check_safety_and_solvability, 
                                  {"Verifier": "Verifier", "Professor": "Professor", "END": END})
    workflow.add_conditional_edges("Verifier", check_audit_status, 
                                  {"Physicist": "Physicist", "Professor": "Professor"})
    workflow.add_edge("Physicist", "Scientist")
    workflow.add_edge("Scientist", "Evaluator")
    workflow.add_conditional_edges("Evaluator", check_execution, 
                                  {"Saboteur": "Saboteur", "Analyst": "Analyst", "END": END})
    workflow.add_edge("Analyst", "Scientist")
    workflow.add_edge("Saboteur", END)

    return workflow.compile()

def harvest_data(state: AgenticState):
    if state.get("execution_success") and state.get("final_correct_code"):
        trace = {
            "problem": state.get("problem_statement"),
            "domain": state.get("domain"),
            "chosen": state.get("final_correct_code"),
            "rejected": state.get("rca_history")[0].get("failed_code_snapshot") if state.get("rca_history") else None,
            "laws": state.get("fundamental_laws"),
            "tier": state.get("difficulty_tier")
        }
        os.makedirs("dataset", exist_ok=True)
        with open("dataset/training_traces.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(trace) + "\n")
            f.flush()
            os.fsync(f.fileno()) # ATOMIC WRITE

async def run_swarm_loop(graph, topic, semaphore):
    async with semaphore:
        print(f"\n>>> [STARTING]: {topic} <<<")
        initial_state = {
            "current_topic": topic, "iteration_count": 0, 
            "total_graph_steps": 0, "rca_history": [], "problem_is_valid": False
        }
        try:
            final_state = await graph.ainvoke(initial_state)
            harvest_data(final_state)
        except Exception as e:
            print(f"[!] Critical Loop Failure: {e}")

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--total-chunks", type=int, default=6)
    args = parser.parse_args()

    router = ShardedRouter(args.chunk, args.total_chunks)
    app = build_agentic_graph(router)
    
    # MASTER CURRICULUM
    domains = [
        "Advanced Engineering", "Applied Physics", 
        "Computational Logic", "Embedded Systems"
    ]
    
    # STRATEGY: Use Semaphore(1) to avoid 429 exhaustion on the free tier
    semaphore = asyncio.Semaphore(1)
    tasks = [run_swarm_loop(app, d, semaphore) for d in domains]
    
    await asyncio.gather(*tasks)
    print("\n--- [SWARM COMPLETE] ---")

if __name__ == "__main__":
    asyncio.run(main())