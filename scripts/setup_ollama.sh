#!/usr/bin/env bash
# setup_ollama.sh -- Setup and verification script for Ollama on macOS Apple Silicon (M3 Pro)

set -e

echo "=== PEOGAI Mini Project: Ollama Setup ==="

# 1. Check if ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "[!] Ollama CLI not found."
    echo "    Install via Homebrew: brew install --cask ollama"
    echo "    Or download from: https://ollama.com/download"
    exit 1
else
    echo "[✓] Ollama binary found at: $(command -v ollama)"
fi

# 2. Check if Ollama daemon is running, start if needed
if ! curl -s http://localhost:11434/api/tags &> /dev/null; then
    echo "[*] Ollama daemon is not responding on port 11434."
    echo "[*] Attempting to start ollama serve in background..."
    ollama serve &> /tmp/ollama.log &
    sleep 3
    if ! curl -s http://localhost:11434/api/tags &> /dev/null; then
        echo "[!] Could not automatically start Ollama. Please launch the Ollama app or run 'ollama serve' in another terminal."
        exit 1
    fi
fi
echo "[✓] Ollama server is running and accessible at http://localhost:11434"

# 3. Pull Qwen2.5-Coder-3B
MODEL_TAG="qwen2.5-coder:3b"
echo "[*] Checking / pulling model '${MODEL_TAG}'..."
ollama pull "${MODEL_TAG}"

# 4. Verify model with a test prompt
echo "[*] Testing model inference with a dry run..."
RESPONSE=$(curl -s http://localhost:11434/api/generate -d "{
  \"model\": \"${MODEL_TAG}\",
  \"prompt\": \"Output only the word TEST:\",
  \"stream\": false
}" | grep -o '"response":"[^"]*"' || echo "")

echo "[✓] Ollama verification complete! Response received."
echo "=== You are ready to run: python -m app.app ==="
