"""
Data Preparation
----------------
Loads tourism.csv directly from the Hugging Face dataset repo, cleans it,
splits it into train/test sets, saves them locally, and uploads the splits
back to the Hugging Face dataset repo.
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

REPO_ID = "scrfrob/tourism-package-dataset"
TARGET_COL = "ProdTaken"

api = HfApi(token=os.getenv("HF_TOKEN"))

# Load the dataset directly from the Hugging Face dataset space
df = pd.read_csv(f"hf://datasets/{REPO_ID}/tourism.csv")
print("Loaded dataset:", df.shape)

# Remove unnecessary / identifier columns that carry no predictive signal
unnecessary_cols = ["Unnamed: 0", "CustomerID"]
df.drop(columns=[c for c in unnecessary_cols if c in df.columns], inplace=True)

# Fix a known data-entry typo in the Gender column
if "Gender" in df.columns:
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

print("Cleaned dataset:", df.shape)

# Split into features / target
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# Stratified split since the target is imbalanced
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save locally
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)
print("Saved local train/test splits.")

# Upload the splits back to the Hugging Face dataset repo
for file_path in ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,
        repo_id=REPO_ID,
        repo_type="dataset",
    )
print("Train/test splits uploaded to the Hugging Face Hub.")
