#!/bin/bash
# Downloads the Gemma-2-2b-it GGUF model weights for the ADTC 2026 submission.
# Public, unauthenticated download - no credentials required.
# Idempotent: skips download if the file already exists.

set -e

MODEL_DIR="model"
MODEL_FILE="gemma-2-2b-it-Q4_K_M.gguf"
MODEL_URL="https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf"

mkdir -p "$MODEL_DIR"

if [ -f "$MODEL_DIR/$MODEL_FILE" ]; then
    echo "Model already downloaded at $MODEL_DIR/$MODEL_FILE - skipping."
else
    echo "Downloading $MODEL_FILE..."
    wget -O "$MODEL_DIR/$MODEL_FILE" "$MODEL_URL"
    echo "Download complete: $MODEL_DIR/$MODEL_FILE"
fi
