#!/usr/bin/env bash
# Download the public Qwen3-1.7B Q4_K_M GGUF weights (only used with --full).
set -euo pipefail
DEST="${1:-data}"
mkdir -p "$DEST"
URL="https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf"
echo "downloading Qwen3-1.7B-Q4_K_M.gguf (~1.1 GB) ..."
curl -L --retry 3 -o "$DEST/Qwen3-1.7B-Q4_K_M.gguf" "$URL"
ls -la "$DEST/Qwen3-1.7B-Q4_K_M.gguf"
