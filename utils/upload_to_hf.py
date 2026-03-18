import os
from huggingface_hub import HfApi

def upload():
    print("\n--- [GATE 3] Secure Vault Transmission ---")
    try:
        api = HfApi()
        
        # Pull securely from GitHub Actions Environment Variables
        username = os.environ.get("HF_USERNAME")
        token = os.environ.get("HF_TOKEN")
        chunk_id = os.environ.get("CHUNK_ID", "unknown")
        
        if not username or not token:
            raise ValueError("Hugging Face credentials missing from GitHub Secrets.")
        
        repo_id = f"{username}/Agentic-Reasoning-Data"
        file_path = f"data/github_chunk_{chunk_id}.jsonl"
        local_path = "dataset/clean_training_data.jsonl"
        
        print(f"[*] Authenticated successfully. Initiating upload to {repo_id} -> {file_path}...")
        
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=file_path,
            repo_id=repo_id,
            repo_type="dataset",
            token=token
        )
        print("[+] PURE PYTHON UPLOAD COMPLETE. Data is securely locked in the vault.")
        
    except Exception as e:
        print(f"[!] CRITICAL UPLOAD ERROR: {e}")
        exit(1) # Forces GitHub Actions to register the failure if the API is down

if __name__ == "__main__":
    upload()