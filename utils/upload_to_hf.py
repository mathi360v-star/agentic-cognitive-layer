from huggingface_hub import HfApi
import os

def upload_to_hf():
    """Streams the sharded SFT and DPO data to the Hugging Face Vault."""
    
    # 1. Initialize API and Environment Variables
    api = HfApi()
    token = os.getenv("HF_TOKEN")
    username = os.getenv("HF_USERNAME")
    chunk_id = os.getenv("CHUNK_ID", "0")
    
    if not token or not username:
        print("[!] HF_TOKEN or HF_USERNAME missing. Upload aborted.")
        return

    # 2. Define the Repository ID
    repo_id = f"{username}/STEM-Reasoning-V3"
    
    # 3. Process both SFT and DPO data streams
    for folder in ["sft", "dpo"]:
        local_path = f"dataset/clean_{folder}.jsonl"
        
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            remote_path = f"data/{folder}/chunk_{chunk_id}.jsonl"
            
            print(f"[*] Uploading {local_path} to {remote_path}...")
            
            try:
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=remote_path,
                    repo_id=repo_id,
                    repo_type="dataset",
                    token=token
                )
                print(f"[+] Successfully pushed {folder} data to Vault.")
            except Exception as e:
                print(f"[!] Upload failed for {folder}: {e}")
        else:
            print(f"[-] No valid {folder} data found to upload.")

# --- THE CRITICAL FIX: Ensure the script actually executes ---
if __name__ == "__main__":
    upload_to_hf()