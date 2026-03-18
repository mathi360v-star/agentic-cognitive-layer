import chromadb
import os
import threading

# Ensure the ephemeral path exists on the Linux server
DB_PATH = "./dataset/ephemeral_memory_db"
os.makedirs(DB_PATH, exist_ok=True)

# Initialize the client safely
chroma_client = chromadb.PersistentClient(path=DB_PATH)
memory_collection = chroma_client.get_or_create_collection(name="rca_heuristics")

# THE ENTERPRISE FIX: A Thread Lock to prevent SQLite concurrent write crashes
db_lock = threading.Lock()

def retrieve_past_mistakes(problem_statement: str, n_results: int = 3) -> str:
    """Safely reads from the Vector DB without colliding with writes."""
    try:
        with db_lock:
            if memory_collection.count() == 0:
                return "No prior mistakes recorded. Proceed with standard engineering principles."
                
            results = memory_collection.query(
                query_texts=[problem_statement],
                n_results=min(n_results, memory_collection.count())
            )
            
            warnings = "WARNING - AVOID THESE PAST MISTAKES:\n"
            for idx, doc in enumerate(results['documents'][0]):
                warnings += f"{idx + 1}. {doc}\n"
                
            return warnings
    except Exception as e:
        # If the DB fails, we do not crash the agent. We just proceed without memory.
        print(f"[!] Vector DB Read Error: {e}. Bypassing memory.")
        return "No prior mistakes retrieved. Proceed."

def save_new_heuristic(problem_statement: str, flawed_assumption: str, generalized_rule: str):
    """Safely writes to the Vector DB. Queues simultaneous writes via Thread Lock."""
    try:
        with db_lock:
            doc_id = f"rca_{memory_collection.count() + 1}"
            memory_collection.add(
                documents=[generalized_rule],
                metadatas=[{"problem": problem_statement, "flawed_assumption": flawed_assumption}],
                ids=[doc_id]
            )
            print(f"[+] Memory embedded in Vector Vault: {doc_id}")
    except Exception as e:
        # If two threads collide despite the lock, gracefully skip the write instead of crashing
        print(f"[!] Vector DB Write Collision: {e}. Skipping memory update to preserve server uptime.")