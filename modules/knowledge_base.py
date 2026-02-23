"""
modules/knowledge_base.py — Offline semantic retrieval from products.txt.

Pipeline:
  1. Load products.txt and split into logical product sections
  2. Embed all chunks using all-MiniLM-L6-v2 (sentence-transformers)
  3. On each query, embed the query and return top-K most similar chunks

The get_relevant_info() interface is stable — swap the internals for
Chroma or any other backend later without touching other modules.
"""

import re
import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    PRODUCTS_TXT_PATH,
    EMBEDDING_MODEL,
    EMBEDDING_CACHE,
    KB_TOP_K,
    KB_CHUNK_SIZE,
)

# ── Load embedding model once at startup ──────────────────────────────────────
print(f"[kb] Loading embedding model '{EMBEDDING_MODEL}' ...")
_embedder = SentenceTransformer(
    EMBEDDING_MODEL,
    cache_folder=str(EMBEDDING_CACHE),
)
print("[kb] Embedding model ready.")


# ── Document loading ──────────────────────────────────────────────────────────

def _load_txt(path) -> str:
    """Load and normalise the raw .txt file (handles Windows CRLF line endings)."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    # Normalise CRLF → LF and collapse 3+ blank lines to 2
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _split_into_chunks(text: str) -> list[str]:
    """
    Split the product doc into chunks by product section.

    Strategy:
      - Primary split on lines that look like product category headers
        (all-caps lines or lines followed immediately by 'Description:')
      - Secondary split if a chunk exceeds KB_CHUNK_SIZE characters
    """
    # Split on known top-level section boundaries
    # These match lines like "Debit Cards", "Loan Products", "Investment Products", etc.
    sections = re.split(r'\n(?=[A-Z][^\n]{2,50}\n)', text)

    chunks = []
    for section in sections:
        section = section.strip()
        if not section or len(section) < 30:
            continue

        if len(section) <= KB_CHUNK_SIZE:
            chunks.append(section)
        else:
            # Split large sections on numbered items (e.g. "1. Salary Advance Loan")
            sub_sections = re.split(r'\n(?=\d+\. )', section)
            for sub in sub_sections:
                sub = sub.strip()
                if sub:
                    chunks.append(sub)

    return [c for c in chunks if len(c) > 20]


# ── Build index at startup ────────────────────────────────────────────────────

def _build_index(path) -> tuple[list[str], np.ndarray]:
    text   = _load_txt(path)
    chunks = _split_into_chunks(text)
    print(f"[kb] Indexing {len(chunks)} chunks from '{path.name}' ...")
    embeddings = _embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    print("[kb] Knowledge base index ready.")
    return chunks, embeddings


_CHUNKS, _EMBEDDINGS = _build_index(PRODUCTS_TXT_PATH)


# ── Public interface ──────────────────────────────────────────────────────────

def get_relevant_info(query: str, top_k: int = KB_TOP_K) -> str:
    """
    Retrieve the most semantically relevant product documentation chunks.

    Args:
        query:  The user's message or question.
        top_k:  Number of top chunks to return.

    Returns:
        Concatenated relevant chunks as a single string.
    """
    query_vec  = _embedder.encode([query], convert_to_numpy=True)
    norms      = np.linalg.norm(_EMBEDDINGS, axis=1, keepdims=True)
    q_norm     = np.linalg.norm(query_vec)
    similarity = (_EMBEDDINGS @ query_vec.T).flatten() / (norms.flatten() * q_norm + 1e-8)

    top_idx  = np.argsort(similarity)[::-1][:top_k]
    top_chunks = [_CHUNKS[i] for i in top_idx]

    return "\n\n---\n\n".join(top_chunks)
