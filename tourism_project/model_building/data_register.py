"""
Data Registration
-----------------
Uploads the local `data` folder (containing tourism.csv) to a Hugging Face
dataset repository, creating the repo first if it doesn't already exist.
"""
import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

REPO_ID = "scrfrob/tourism-package-dataset"
REPO_TYPE = "dataset"

api = HfApi(token=os.getenv("HF_TOKEN"))

try:
    api.repo_info(repo_id=REPO_ID, repo_type=REPO_TYPE)
    print(f"Repo '{REPO_ID}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Repo '{REPO_ID}' not found. Creating it...")
    create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, private=False)
    print(f"Repo '{REPO_ID}' created.")

api.upload_folder(
    folder_path="tourism_project/data",
    repo_id=REPO_ID,
    repo_type=REPO_TYPE,
)
print("Dataset registered on the Hugging Face Hub.")
