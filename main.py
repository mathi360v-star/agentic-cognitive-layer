import argparse
import asyncio
import json
import os
import traceback
import random 
import time
from langgraph.graph import StateGraph, END
from schemas.models import AgenticState

# --- 1. IMPORT SECTION ---
from agents.professor import generate_curriculum
from agents.epistemic_eval import node_epistemic_evaluator 
from agents.verifier import node_verifier
from agents.physicist import node_physicist 
from agents.scientist import propose_solution
from agents.evaluator import evaluate_code
from agents.analyst import analyze_failure
from agents.saboteur import node_saboteur # IMPORTED

# --- 2. THE MULTI-DOMAIN CURRICULUM ---
MASTER_CURRICULUM = [
    "Advanced Embedded C & RTOS Memory Management",
    "Python AI & Reinforcement Learning Algorithms",
    "Calculus & Differential Equations Proofs",
    "Quantum Mechanics & Physical System Modeling",
    "Digital Logic Design & Boolean Algebra",
    "Aerospace Dynamics & Fluid Mathematics"
]

def load_curriculum():
    return MASTER_CURRICULUM

# --- 3. ROUTING LOGIC (The Brain) ---
def check_epistemic_status(state: AgenticState):
    if state.get("problem_is_valid"): return "Verifier"
    return "Professor"

def check_verification_status(state: AgenticState):
    if state.get("problem_is_valid"): return "Physicist"
    return "Professor" 

def check_execution_status(state: AgenticState):
    """
    DPO STRATEGY: 
    If successful on the first try, we FORCE it to the Saboteur.
    This creates a 'Rejected' version of a perfect answer for DPO training.
    """
    if state.get("execution_success"):
        # If it's the first success (iteration 0 or 1), get DPO data
        if state.get("iteration_count", 0) <= 1:
            return "Saboteur"
        return "success_end"
    
    # Handle Failures
    error_msg = str(state.get("traceback", ""))
    if "Mechanical Failure" in error_msg or "429" in error_msg:
        return "max_retries_end"
        
    if state.get("iteration_count", 0) >= 5:
        return "max_retries_end"
        
    return "analyze_failure"

# --- 4. THE DUAL-STREAM HARVESTER ---
def harvest_training_data(state: AgenticState):
    if state.get("execution_success"):
        trace = {
            "problem_statement": state.get("problem_statement"),
            "domain": state.get("domain", "Engineering"),
            "target_language": state.get("target_language", "python"),
            "fundamental_laws": state.get("fundamental_laws", ""), 
            "difficulty_tier": state.get("difficulty_tier", "Tier 1"),
            "rca_history": state.get("rca_history", []),
            "final_correct_code": state.get("final_correct_code", state.get("proposed_code"))
        }
        
        with open("dataset/training_traces.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(trace) + "\n")
        print(f"[$$$] {state.get('difficulty_tier')} trace harvested!")

# --- 5. THE GRAPH CONSTRUCTION ---
def build_agentic_graph():
    workflow = StateGraph(AgenticState)
    
    # Register Nodes
    workflow.add_node("Professor", generate_curriculum)
    workflow.add_node("Epistemic", node_epistemic_evaluator) 
    workflow.add_node("Verifier", node_verifier)
    workflow.add_node("Physicist", node_physicist) 
    workflow.add_node("Scientist", propose_solution)
    workflow.add_node("Evaluator", evaluate_code)
    workflow.add_node("Analyst", analyze_failure)
    workflow.add_node("Saboteur", node_saboteur) # ADDED

    # Define the Flow
    workflow.set_entry_point("Professor")
    workflow.add_edge("Professor", "Epistemic")
    
    workflow.add_conditional_edges(
        "Epistemic", 
        check_epistemic_status, 
        {"Verifier": "Verifier", "Professor": "Professor"}
    )
    
    workflow.add_conditional_edges(
        "Verifier", 
        check_verification_status, 
        {"Physicist": "Physicist", "Professor": "Professor"}
    )
    
    workflow.add_edge("Physicist", "Scientist")
    workflow.add_edge("Scientist", "Evaluator")
    
    # The Decision Gate
    workflow.add_conditional_edges(
        "Evaluator", 
        check_execution_status, 
        {
            "success_end": END, 
            "Saboteur": "Saboteur", 
            "analyze_failure": "Analyst", 
            "max_retries_end": END
        }
    )
    
    workflow.add_edge("Analyst", "Scientist")
    workflow.add_edge("Saboteur", END) # Saboteur finishes the loop

    return workflow.compile()

# --- 6. EXECUTION ENGINE ---
async def run_single_loop(graph, run_id: int, topic: str, semaphore: asyncio.Semaphore):
    async with semaphore:
        await asyncio.sleep(random.uniform(2, 6)) # Jitter to avoid 429s
        print(f"\n>>> [RUN {run_id}] Domain: {topic} <<<")
        initial_state = {
            "iteration_count": 0, 
            "rca_history": [], 
            "current_topic": topic,
            "problem_is_valid": False 
        }
        try:
            final_state = await graph.ainvoke(initial_state)
            harvest_training_data(final_state)
        except Exception as e:
            print(f"[!] ENGINE FAILURE: {e}")

async def main(chunk_index: int, total_chunks: int):
    app = build_agentic_graph()
    active_curriculum = load_curriculum()
    
    chunk_size = max(1, len(active_curriculum) // total_chunks)
    start_idx = chunk_index * chunk_size
    end_idx = start_idx + chunk_size if chunk_index < total_chunks - 1 else len(active_curriculum)
    assigned_topics = active_curriculum[start_idx:end_idx]
    
    semaphore = asyncio.Semaphore(1) # Safe concurrency for Gemini Free
    
    tasks = [
        run_single_loop(app, start_idx + j, topic, semaphore) 
        for j, topic in enumerate(assigned_topics)
    ]
    
    await asyncio.gather(*tasks, return_exceptions=True)
    print("\n[=== SWARM BATCH COMPLETE ===]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous STEM Swarm")
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--total-chunks", type=int, default=1)
    args = parser.parse_args()
    
    os.makedirs("dataset", exist_ok=True)
    asyncio.run(main(args.chunk, args.total_chunks))
    time.sleep(5) # Final cooldown