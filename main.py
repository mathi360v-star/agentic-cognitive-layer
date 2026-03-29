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

    # --- THE FIX: Proper Async Node Factory ---
    # We define local async functions so LangGraph can properly await them.
    async def professor_node(state: AgenticState):
        return await generate_curriculum(state, router)

    async def epistemic_node(state: AgenticState):
        return await node_epistemic_evaluator(state, router)

    async def verifier_node(state: AgenticState):
        return await node_verifier(state, router)

    async def physicist_node(state: AgenticState):
        return await node_physicist(state, router)

    async def scientist_node(state: AgenticState):
        return await propose_solution(state, router)

    async def evaluator_node(state: AgenticState):
        return await evaluate_code(state, router)

    async def analyst_node(state: AgenticState):
        return await analyze_failure(state, router)

    async def saboteur_node(state: AgenticState):
        return await node_saboteur(state, router)

    # 1. Register Nodes using the new async wrappers
    workflow.add_node("Professor", professor_node)
    workflow.add_node("Epistemic", epistemic_node)
    workflow.add_node("Verifier", verifier_node)
    workflow.add_node("Physicist", physicist_node)
    workflow.add_node("Scientist", scientist_node)
    workflow.add_node("Evaluator", evaluator_node)
    workflow.add_node("Analyst", analyst_node)
    workflow.add_node("Saboteur", saboteur_node)

    # 2. Logic Gates (The Safety Circuit)
    def check_safety_and_solvability(state: AgenticState):
        steps = state.get("total_graph_steps", 0)
        # Update state count
        state["total_graph_steps"] = steps + 1
        
        if steps >= 15: 
            print("!!! CIRCUIT BREAKER: Swarm Aborted !!!")
            return "END"
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

# --- ATOMIC DATA HARVESTING ---
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
            os.fsync(f.fileno())

async def run_swarm_loop(graph, topic, semaphore):
    async with semaphore:
        print(f"\n>>> [STARTING]: {topic} <<<")
        initial_state = {
            "current_topic": topic, "iteration_count": 0, 
            "total_graph_steps": 0, "rca_history": [], "problem_is_valid": False
        }
        try:
            # We must use .ainvoke for the async graph
            final_state = await graph.ainvoke(initial_state)
            harvest_data(final_state)
        except Exception as e:
            print(f"[!] Critical Loop Failure: {e}")

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--total-chunks", type=int, default=6)
    args = parser.parse_args()

    # Create the router once
    router = ShardedRouter(args.chunk, args.total_chunks)
    app = build_agentic_graph(router)
    
    # Curriculum domains
    domains = ["Advanced Engineering", "Applied Physics", "Computational Logic", "Embedded Systems"]
    
    # FREE TIER STRATEGY: One at a time to avoid 429 burnout
    semaphore = asyncio.Semaphore(1)
    tasks = [run_swarm_loop(app, d, semaphore) for d in domains]
    
    await asyncio.gather(*tasks)
    print("\n--- [SWARM COMPLETE] ---")

if __name__ == "__main__":
    asyncio.run(main())