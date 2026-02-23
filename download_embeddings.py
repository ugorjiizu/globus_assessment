"""
download_embeddings.py — Run this ONCE while connected to the internet.

Downloads and caches the sentence-transformers embedding model locally.
After this runs successfully, the app works fully offline forever.
"""

from pathlib import Path
from sentence_transformers import SentenceTransformer

CACHE_DIR = Path(__file__).parent / "models" / "embeddings"
MODEL_NAME = "all-MiniLM-L6-v2"

print(f"Downloading '{MODEL_NAME}' to {CACHE_DIR} ...")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

model = SentenceTransformer(MODEL_NAME, cache_folder=str(CACHE_DIR))

# Quick smoke test
test = model.encode(["test sentence"])
print(f"Done. Model cached and verified (output shape: {test.shape}).")
print("You can now run the app fully offline.")