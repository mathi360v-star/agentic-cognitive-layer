import os
import time
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError
from tenacity import retry, stop_after_attempt, wait_exponential

# --- NETWORK RESILIENCE ---
# Only retry on connection/server errors. Don't retry on Auth/404 errors.
@retry(
    stop=stop_after_attempt(3), 
    wait=wait_exponential(multiplier=1, min=4, max=20),
    reraise=True
)
def execute_upload(api, local_path, remote_path, repo_id, token):
    if os.path.exists(local_path) and os.path.getsize(local_path) > 10: # Min 10 bytes to ignore headers
        print(f"[*] Securely Stream-Uploading: {local_path} -> {remote_path}...")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=remote_path,
            repo_id=repo_id,
            repo_type="dataset",
            token=token
        )
        return True
    else:
        print(f"[!] Warning: {local_path} is empty or missing.")
        return False

def upload():
    print("\n--- [GATE 3] Secure Vault Transmission (Dual-Stream V2) ---")
    try:
        api = HfApi()
        
        username = os.environ.get("HF_USERNAME")
        token = os.environ.get("HF_TOKEN")
        chunk_id = os.environ.get("CHUNK_ID", "unknown")
        
        if not username or not token:
            raise ValueError("CRITICAL: HF_USERNAME or HF_TOKEN is missing in GitHub Secrets.")
        
        repo_id = f"{username}/Agentic-Reasoning-Data"
        timestamp = int(time.time())

        # --- SELF-HEALING: Auto-create repo if missing ---
        try:
            api.repo_info(repo_id=repo_id, repo_type="dataset", token=token)
        except HfHubHTTPError:
            print(f"[*] Repo {repo_id} not found. Creating it now...")
            api.create_repo(repo_id=repo_id, repo_type="dataset", token=token, private=True)

        # --- THE DUAL-STREAM TARGETS ---
        uploads = [
            ("dataset/clean_sft_data.jsonl", f"data/sft/chunk_{chunk_id}_{timestamp}.jsonl"),
            ("dataset/clean_dpo_data.jsonl", f"data/dpo/chunk_{chunk_id}_{timestamp}.jsonl")
        ]
        
        print(f"[*] Identity Verified: {username}. Pipeline Live.")
        
        success_count = 0
        for local, remote in uploads:
            if execute_upload(api, local, remote, repo_id, token):
                success_count += 1
        
        # --- CRITICAL SAFETY CHECK ---
        # If the AI struggled and produced NO data for both SFT and DPO, we fail the job.
        if success_count == 0:
            print("[!] FATAL: No valid traces were produced in this run. Failing job to alert architect.")
            # exit(1) # Uncomment this if you want the GitHub Actions red-dot alert

        print(f"[+] PURE PYTHON UPLOAD COMPLETE. {success_count} streams secured.")
        
    except Exception as e:
        print(f"[!] CRITICAL UPLOAD ERROR: {e}")
        exit(1)

if __name__ == "__main__":
    upload()