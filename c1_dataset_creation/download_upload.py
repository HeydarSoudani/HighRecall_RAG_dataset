import os
from huggingface_hub import HfApi, hf_hub_download

# --- 1) CONFIG ---
HF_TOKEN = os.getenv("HF_TOKEN") 
# or "hf_your_token_here"   # or set env var HF_TOKEN

# === UPLOAD ============================
# --- Corpus
REPO_ID = "HeydarS/enwiki_20251001"
LOCAL_FILE = "data/enwiki_20251001.jsonl"
PATH_IN_REPO = "enwiki_20251001.jsonl"

# api = HfApi(token=HF_TOKEN)
# api.create_repo(
#     repo_id=REPO_ID,
#     repo_type="dataset",
#     private=False,
#     exist_ok=True
# )
# api.upload_file(
#     path_or_fileobj=LOCAL_FILE,
#     path_in_repo=PATH_IN_REPO,
#     repo_id=REPO_ID,
#     repo_type="dataset",
#     commit_message="Add large JSONL"
# )
# print(f"Uploaded to: https://huggingface.co/datasets/{REPO_ID}/blob/main/{PATH_IN_REPO}")



# === DOWNLOAD ===========================
# --- Corpus
# downloaded_path = hf_hub_download(
#     repo_id=REPO_ID,
#     filename=PATH_IN_REPO,
#     repo_type="dataset",
#     token=HF_TOKEN,
#     local_dir="./downloads",                # where to place it
#     local_dir_use_symlinks=False            # make a real copy
# )
# print("Downloaded file at:", downloaded_path)


# python c1_dataset_creation/download_upload.py
