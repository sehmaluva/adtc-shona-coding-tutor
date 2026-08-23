#!/bin/bash
# Downloads Gemma GGUF + MiniLM embedder into model/ for the ADTC 2026 submission.
# Public, unauthenticated download - no credentials required.
# Idempotent + resumable: continues a partial download if interrupted.

MODEL_DIR="model"
MODEL_FILE="gemma-2-2b-it-Q4_K_M.gguf"
MODEL_URL="https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf"
EMBEDDER_DIR="$MODEL_DIR/all-MiniLM-L6-v2"
EMBEDDER_REPO="sentence-transformers/all-MiniLM-L6-v2"

mkdir -p "$MODEL_DIR"
DEST="$MODEL_DIR/$MODEL_FILE"


trap 'echo; echo "Download interrupted. Partial file kept at $DEST for resuming. Re-run the script to continue."; exit 130' INT TERM

echo "Downloading $MODEL_FILE (resumable)..."

wget -c \
     --tries=0 \
     --retry-connrefused \
     --waitretry=5 \
     --timeout=30 \
     -O "$DEST" \
     "$MODEL_URL"
WGET_STATUS=$?

if [ $WGET_STATUS -ne 0 ]; then
    echo "Download failed or was interrupted (wget exit code $WGET_STATUS)."
    echo "Partial file kept at $DEST - re-run the script to resume."
    exit $WGET_STATUS
fi


REMOTE_SIZE=$(wget --spider --server-response -O /dev/null "$MODEL_URL" 2>&1 \
    | grep -i "Content-Length" | tail -1 | awk '{print $2}' | tr -d '\r')
LOCAL_SIZE=$(stat -c%s "$DEST" 2>/dev/null || stat -f%z "$DEST")

if [ -n "$REMOTE_SIZE" ] && [ "$LOCAL_SIZE" != "$REMOTE_SIZE" ]; then
    echo "File size mismatch: got $LOCAL_SIZE bytes, expected $REMOTE_SIZE bytes."
    echo "Download is incomplete. Re-run the script to resume."
    exit 1
fi

echo "Download complete: $DEST"

# --- MiniLM embedder (same MODEL_DIR as Gemma) ---
if [ -f "$EMBEDDER_DIR/model.safetensors" ]; then
    echo "Embedder already present: $EMBEDDER_DIR (skipping)"
else
    echo "Downloading $EMBEDDER_REPO into $EMBEDDER_DIR..."
    if ! python3 -c "import huggingface_hub" 2>/dev/null; then
        echo "huggingface_hub is required to download the embedder."
        echo "Run: pip install -r requirements.txt"
        echo "Then re-run: bash download_model.sh"
        exit 1
    fi
    python3 - <<EOF
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="$EMBEDDER_REPO",
    local_dir="$EMBEDDER_DIR",
)
print("Embedder download complete: $EMBEDDER_DIR")
EOF
    if [ $? -ne 0 ]; then
        echo "Embedder download failed."
        exit 1
    fi
    if [ ! -f "$EMBEDDER_DIR/model.safetensors" ]; then
        echo "Embedder download finished but model.safetensors is missing at $EMBEDDER_DIR"
        exit 1
    fi
fi
