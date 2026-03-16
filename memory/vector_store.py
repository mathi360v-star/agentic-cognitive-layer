import chromadb
import os
import uuid
from tenacity import retry, stop_after_attempt, wait_fixed

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(CURRENT_DIR, "../dataset/ephemeral_memory_db")
os.makedirs(DB_PATH, exist_ok=True)

chroma_client = chromadb.PersistentClient(path=DB_PATH)
memory_collection = chroma_client.get_or_create_collection(name="rca_heuristics")

# Retry decorator catches SQLite Database Locking issues in high-concurrency environments
@retry(stop=stop_after_attempt(5), wait=wait_fixed(0.5))
def retrieve_past_mistakes(problem_statement: str, n_results: int = 3) -> str:
    """Retrieves relevant past failures safely."""
    if memory_collection.count() == 0:
        return "No prior mistakes recorded. Proceed with standard engineering principles."
        
    results = memory_collection.query(
        query_texts=[problem_statement],
        n_results=min(n_results, memory_collection.count())
    )
    
    if not results.get('documents') or not results['documents'][0]:
        return "No prior mistakes recorded. Proceed with standard engineering principles."
    
    warnings = "WARNING - AVOID THESE PAST MISTAKES:\n"
    for idx, doc in enumerate(results['documents'][0]):
        warnings += f"{idx + 1}. {doc}\n"
        
    return warnings

@retry(stop=stop_after_attempt(5), wait=wait_fixed(0.5))
def save_new_heuristic(problem_statement: str, flawed_assumption: str, generalized_rule: str):
    """Saves a new rule. Protected against SQLite lock crashes."""
    doc_id = f"rca_{uuid.uuid4().hex[:8]}" 
    
    memory_collection.add(
        documents=[generalized_rule],
        metadatas=[{"problem": problem_statement, "flawed_assumption": flawed_assumption}],
        ids=[doc_id]
    )
    print(f"[+] Memory embedded in Vector Vault: {doc_id}")