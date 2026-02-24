"""
config.py — Central configuration for the Globus Bank offline chatbot.
All paths, model settings, and constants live here.
"""

import os

# Must be set before ANY other imports — prevents sentence-transformers
# and HuggingFace from attempting network calls at load time.
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from pathlib import Path

# ── Project root ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent

# ── Data files (sourced directly from provided files) ─────────────────────────
PRODUCTS_TXT_PATH  = BASE_DIR / "data" / "Product_Information.txt"
CUSTOMERS_XLS_PATH = BASE_DIR / "data" / "Globus_AI_Engr_Interview_Data.xlsx"

# ── Model paths ───────────────────────────────────────────────────────────────
MODELS_DIR         = BASE_DIR / "models"

# Place your GGUF model in models/ and set filename here (or via env var)
# Recommended: Llama-3.2-3B-Instruct-Q4_K_M.gguf (~2.0 GB)
#              Phi-3-mini-4k-instruct-q4.gguf      (~2.2 GB)
LLM_MODEL_PATH     = MODELS_DIR / os.getenv("LLM_MODEL_FILE", "Llama-3.2-3B-Instruct-Q4_K_M.gguf")

# ── LLM inference settings ────────────────────────────────────────────────────
LLM_CONTEXT_SIZE   = 4096
LLM_MAX_TOKENS     = 512
LLM_TEMPERATURE    = 0.4
LLM_THREADS        = 4       # Set to your CPU core count
LLM_GPU_LAYERS     = 0       # Set >0 if you have CUDA/Metal GPU

INTENT_MAX_TOKENS  = 20
INTENT_TEMPERATURE = 0.0

# ── Embedding model (sentence-transformers, runs fully offline after first dl) ─
EMBEDDING_MODEL    = "all-MiniLM-L6-v2"
EMBEDDING_CACHE    = MODELS_DIR / "embeddings"

# ── Knowledge base retrieval ──────────────────────────────────────────────────
KB_TOP_K           = 3       # Number of chunks to retrieve per query
KB_CHUNK_SIZE      = 400     # Target characters per chunk

# ── Flask ─────────────────────────────────────────────────────────────────────
FLASK_HOST         = "0.0.0.0"
FLASK_PORT         = 5050
FLASK_DEBUG        = False
SECRET_KEY         = os.getenv("SECRET_KEY", "globus-offline-dev-key")

# ── Session ───────────────────────────────────────────────────────────────────
MAX_HISTORY_TURNS  = 8

# ── Intent categories ─────────────────────────────────────────────────────────
INTENTS = [
    "greeting",
    "general_inquiry",
    "account_information",
    "product_inquiry",
    "card_block_request",
]

# ── User-facing messages ──────────────────────────────────────────────────────
ACCOUNT_NOT_FOUND_MSG = (
    "I couldn't find a Globus Bank account matching that number. "
    "You're welcome to ask about our products and services."
)

ANONYMOUS_RESTRICTION_MSG = (
    "I can only provide general product information for unauthenticated sessions. "
    "Please provide a valid account number for account-specific assistance."
)

CARD_BLOCK_ANONYMOUS_MSG = (
    "Card blocking requires a verified account. "
    "Please provide your account number so I can assist you."
)