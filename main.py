import argparse
import asyncio
import json
import os
import traceback
from langgraph.graph import StateGraph, END
from schemas.models import AgenticState

# Import your agents
from agents.professor import generate_curriculum
from agents.verifier import audit_problem
# ---> NEW: Import the Physicist
from agents.physicist import node_physicist 
from agents.scientist import propose_solution
from agents.evaluator import evaluate_code
from agents.analyst import analyze_failure

MASTER_CURRICULUM = [
    "Optimizing dynamic memory allocation to prevent heap fragmentation in C",
    "Implementing a custom UART communication protocol for embedded microcontrollers",
    "Designing a thread-safe circular buffer for high-speed sensor data in C",
    "Applying multivariable calculus to determine the center of mass of a non-uniform 3D object",
    "Using linear algebra matrix transformations to solve multi-axis inverse kinematics",
    "Calculating magnetic flux density across a complex surface using double integrals",
    "Designing an isolated sandbox environment for an AI Security Gateway",
    "Implementing a self-improving cognitive reasoning loop for an autonomous agent",
    "Architecting a secure voice-biometric authentication pipeline for an AI assistant",
]

def load_curriculum():
    topics_file = "dataset/topics.json"
    if os.path.exists(topics_file):
        with open(topics_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return MASTER_CURRICULUM

def check_verification_status(state: AgenticState):
    # ---> CRITICAL FIX: Route to Physicist instead of Scientist upon approval
    if state.get("problem_is_valid"): return "Physicist"
    return "Professor" 

def check_execution_status(state: AgenticState):
    if state.get("execution_success"): return "success_end"
    error_msg = str(state.get("traceback", ""))
    if "Mechanical Failure" in error_msg or "Cannot connect" in error_msg or "HTTP 401" in error_msg:
        print("[-] Network Error! Skipping RCA to protect memory vault.")
        return "max_retries_end"
    if state.get("iteration_count", 0) >= 5:
        print("[-] Max iterations reached. Aborting this problem.")
        return "max_retries_end"
    return "analyze_failure"

def harvest_training_data(state: AgenticState):
    if state.get("iteration_count", 0) > 1 and state.get("execution_success"):
        # ---> CRITICAL FIX: Ensure the new fields are actually saved to the file
        trace = {
            "problem": state.get("problem_statement"),
            "domain": state.get("domain", "Engineering"),
            "language": state.get("target_language", "python"),
            "fundamental_laws": state.get("fundamental_laws", ""), # NEW
            "difficulty_tier": state.get("difficulty_tier", ""),   # NEW
            "rca_history": state.get("rca_history", []),
            "final_correct_code": state.get("proposed_code")
        }
        with open("dataset/training_traces.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(trace) + "\n")
        print("[$$$] Golden trace harvested and saved to dataset!")

def build_agentic_graph():
    workflow = StateGraph(AgenticState)
    workflow.add_node("Professor", generate_curriculum)
    workflow.add_node("Verifier", audit_problem)
    # ---> NEW: Add the Physicist Node to the Graph
    workflow.add_node("Physicist", node_physicist) 
    workflow.add_node("Scientist", propose_solution)
    workflow.add_node("Evaluator", evaluate_code)
    workflow.add_node("Analyst", analyze_failure)

    workflow.set_entry_point("Professor")
    workflow.add_edge("Professor", "Verifier")
    
    # ---> NEW: Conditional router now points to Physicist
    workflow.add_conditional_edges(
        "Verifier", 
        check_verification_status, 
        {"Physicist": "Physicist", "Professor": "Professor"}
    )
    
    # ---> NEW: The Physicist extracts the laws, then hands off to the Scientist
    workflow.add_edge("Physicist", "Scientist")
    
    workflow.add_edge("Scientist", "Evaluator")
    workflow.add_conditional_edges("Evaluator", check_execution_status, {"success_end": END, "max_retries_end": END, "analyze_failure": "Analyst"})
    workflow.add_edge("Analyst", "Scientist")

    return workflow.compile()

# --------------------------------------------------------------
# THE 10x SEMAPHORE SWARM
# --------------------------------------------------------------
async def run_single_loop(graph, run_id: int, topic: str, semaphore: asyncio.Semaphore):
    """Executes a problem only when the Semaphore grants access."""
    async with semaphore:
        print(f"\n>>> Starting Loop {run_id} | Topic: {topic} <<<")
        initial_state = {"iteration_count": 0, "rca_history": [], "current_topic": topic}
        try:
            final_state = await graph.ainvoke(initial_state)
            harvest_training_data(final_state)
        except Exception as e:
            print(f"[!] CRITICAL ENGINE FAILURE in Loop {run_id} ({topic}): {e}")
            traceback.print_exc() 

async def main(chunk_index: int, total_chunks: int):
    app = build_agentic_graph()
    active_curriculum = load_curriculum()
    
    chunk_size = max(1, len(active_curriculum) // total_chunks)
    start_idx = chunk_index * chunk_size
    end_idx = start_idx + chunk_size if chunk_index < total_chunks - 1 else len(active_curriculum)
    assigned_topics = active_curriculum[start_idx:end_idx]
    
    print(f"[*] Server Booted. Assigned Chunk {chunk_index}. Processing {len(assigned_topics)} topics.")
    
    # Allow exactly 4 concurrent LLM streams per server
    semaphore = asyncio.Semaphore(4)
    
    tasks = [
        run_single_loop(app, start_idx + j, topic, semaphore) 
        for j, topic in enumerate(assigned_topics)
    ]
    
    print("[*] All tasks queued. Semaphore engaged. Commencing high-speed generation...")
    await asyncio.gather(*tasks, return_exceptions=True)
    print("\n[=== CHUNK GENERATION COMPLETELY FINISHED ===]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Autonomous AI Data Swarm")
    parser.add_argument("--chunk", type=int, default=0, help="Which chunk index to process")
    parser.add_argument("--total-chunks", type=int, default=1, help="Total number of chunks")
    args = parser.parse_args()
    
    # GUARANTEE THE FOLDER EXISTS BEFORE THE AI WRITES TO IT
    os.makedirs("dataset", exist_ok=True)
    
    asyncio.run(main(args.chunk, args.total_chunks))