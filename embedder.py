"""Load the local MiniLM embedder from model/ (same dir as the GGUF)."""

from pathlib import Path

from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parent
EMBEDDER_DIR = REPO_ROOT / "model" / "all-MiniLM-L6-v2"
EMBEDDER_WEIGHTS = EMBEDDER_DIR / "model.safetensors"


def load_embedder() -> SentenceTransformer:
    """Load all-MiniLM-L6-v2 from disk only — no Hugging Face network calls."""
    if not EMBEDDER_WEIGHTS.is_file():
        raise FileNotFoundError(
            f"Local embedder not found at {EMBEDDER_DIR}. "
            "Run: bash download_model.sh"
        )
    return SentenceTransformer(str(EMBEDDER_DIR), local_files_only=True)
