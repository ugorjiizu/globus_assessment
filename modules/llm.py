"""
modules/llm.py — llama.cpp model loader (singleton).

Loads the GGUF model once at startup and shares the instance across all modules.
Reloading on every request would cost 10–30 seconds — this pattern avoids that.
"""

from llama_cpp import Llama
from config import LLM_MODEL_PATH, LLM_CONTEXT_SIZE, LLM_THREADS, LLM_GPU_LAYERS

_llm_instance: Llama | None = None


def get_llm() -> Llama:
    """Return the shared Llama model instance, loading it on first call."""
    global _llm_instance

    if _llm_instance is not None:
        return _llm_instance

    if not LLM_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"\n[llm] Model not found at: {LLM_MODEL_PATH}\n"
            "Please download a GGUF model and place it in the models/ directory.\n"
            "See README.md for download instructions."
        )

    print(f"[llm] Loading model: {LLM_MODEL_PATH.name} ...")
    _llm_instance = Llama(
        model_path=str(LLM_MODEL_PATH),
        n_ctx=LLM_CONTEXT_SIZE,
        n_threads=LLM_THREADS,
        n_gpu_layers=LLM_GPU_LAYERS,
        verbose=False,
    )
    print("[llm] Model loaded successfully.")
    return _llm_instance


def generate(
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.4,
    stop: list[str] | None = None,
) -> str:
    """
    Run inference on the loaded model.

    Args:
        prompt:      Full formatted prompt string.
        max_tokens:  Maximum tokens to generate.
        temperature: Sampling temperature (0 = deterministic).
        stop:        Stop sequences — generation halts when any is hit.

    Returns:
        Generated text as a stripped string.
    """
    llm = get_llm()
    stop_seqs = stop or ["</s>", "<|end|>", "<|eot_id|>", "User:", "Human:"]

    output = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop_seqs,
        echo=False,
    )
    return output["choices"][0]["text"].strip()
