import argparse
import asyncio
import json
import os
import random

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
from schemas.models import AgenticState

# Infrastructure Imports
from utils.llm_router import ShardedRouter
from utils.vector_vault import vault

# Agent Imports
from agents.professor import generate_curriculum
from agents.epistemic_eval import node_epistemic_evaluator
from agents.verifier import node_verifier
from agents.physicist import node_physicist
from agents.scientist import propose_solution
from agents.evaluator import evaluate_code
from agents.analyst import analyze_failure
from agents.saboteur import node_saboteur

# ==============================================================
# 1. NODE WRAPPERS (Correct State Update Pattern)
# ==============================================================

async def base_wrapper(state: AgenticState, config: RunnableConfig, func):
    """Increments graph steps correctly and passes the router."""
    router = config["configurable"]["router"]
    # Step increment MUST happen via return value
    steps = state.get("total_graph_steps", 0) + 1
    
    result = await func(state, router)
    # Merge the function result with the step increment
    result["total_graph_steps"] = steps
    return result

# Partial functions to keep the graph clean
professor_node = lambda s, c: base_wrapper(s, c, generate_curriculum)
epistemic_node = lambda s, c: base_wrapper(s, c, node_epistemic_evaluator)
verifier_node  = lambda s, c: base_wrapper(s, c, node_verifier)
physicist_node = lambda s, c: base_wrapper(s, c, node_physicist)
scientist_node = lambda s, c: base_wrapper(s, c, propose_solution)
evaluator_node = lambda s, c: base_wrapper(s, c, evaluate_code)
analyst_node   = lambda s, c: base_wrapper(s, c, analyze_failure)
saboteur_node  = lambda s, c: base_wrapper(s, c, node_saboteur)

# ==============================================================
# 2. ROUTING LOGIC (Read-Only)
# ==============================================================

def check_safety_and_solvability(state: AgenticState):
    steps = state.get("total_graph_steps", 0)
    
    if steps >= 18:
        print("!!! CIRCUIT BREAKER: Swarm Aborted to Prevent Key Burn !!!")
        return "END"
    
    if not state.get("problem_is_valid"): 
        return "Professor"
    return "Verifier"

def check_audit_status(state: AgenticState):
    if state.get("problem_is_valid"): return "Physicist"
    return "Professor"

def check_execution(state: AgenticState):
    if state.get("execution_success"):
        if state.get("iteration_count", 0) <= 1:
            return "Saboteur"
        return "END"
    if state.get("iteration_count", 0) >= 5: 
        return "END"
    return "Analyst"

# ==============================================================
# 3. GRAPH CONSTRUCTION
# ==============================================================

def build_agentic_graph():
    workflow = StateGraph(AgenticState)
    
    workflow.add_node("Professor", professor_node)
    workflow.add_node("Epistemic", epistemic_node)
    workflow.add_node("Verifier", verifier_node)
    workflow.add_node("Physicist", physicist_node)
    workflow.add_node("Scientist", scientist_node)
    workflow.add_node("Evaluator", evaluator_node)
    workflow.add_node("Analyst", analyst_node)
    workflow.add_node("Saboteur", saboteur_node)

    workflow.set_entry_point("Professor")
    workflow.add_edge("Professor", "Epistemic")
    workflow.add_edge("Physicist", "Scientist")
    workflow.add_edge("Scientist", "Evaluator")
    workflow.add_edge("Analyst", "Scientist")
    workflow.add_edge("Saboteur", END)

    workflow.add_conditional_edges("Epistemic", check_safety_and_solvability, 
                                  {"Verifier": "Verifier", "Professor": "Professor", "END": END})
    workflow.add_conditional_edges("Verifier", check_audit_status, 
                                  {"Physicist": "Physicist", "Professor": "Professor"})
    workflow.add_conditional_edges("Evaluator", check_execution, 
                                  {"Saboteur": "Saboteur", "Analyst": "Analyst", "END": END})

    return workflow.compile()

# ==============================================================
# 4. DATA HARVESTING (Index-Safe)
# ==============================================================

def harvest_data(state: AgenticState):
    if state.get("execution_success"):
        # Safe extraction for DPO rejected sample
        history = state.get("rca_history", [])
        rejected_sample = history[0].get("failed_code_snapshot") if history else None
        
        trace = {
            "problem": state.get("problem_statement"),
            "domain": state.get("domain", "Engineering"),
            "chosen": state.get("final_correct_code"),
            "rejected": rejected_sample
        }
        
        os.makedirs("dataset", exist_ok=True)
        with open("dataset/training_traces.jsonl", "a") as f:
            f.write(json.dumps(trace) + "\n")
            f.flush()
            os.fsync(f.fileno())
        print(f"[$$$] Harvested {state.get('domain')} trace.")

async def run_swarm(chunk_id, total_chunks):
    router = ShardedRouter(chunk_id, total_chunks)
    app = build_agentic_graph()
    
    # Sequential topics to prevent 429 burst exhaustion
    topics = ["Advanced Engineering", "Applied Physics", "Computational Logic", "Embedded Systems"]
    
    for topic in topics:
        print(f"\n>>> [CHUNK {chunk_id}] Starting: {topic} <<<")
        # Full initialization to prevent Agent context-starvation
        initial_state = {
            "current_topic": topic,
            "iteration_count": 0,
            "total_graph_steps": 0,
            "rca_history": [],
            "problem_is_valid": False,
            "fundamental_laws": "",
            "audit_feedback": "",
            "domain": "General STEM",
            "target_language": "Python"
        }
        
        try:
            final_state = await app.ainvoke(initial_state, {"configurable": {"router": router}})
            harvest_data(final_state)
        except Exception as e:
            print(f"[!] Critical Loop Failure: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--total-chunks", type=int, default=6)
    args = parser.parse_args()
    asyncio.run(run_swarm(args.chunk, args.total_chunks))