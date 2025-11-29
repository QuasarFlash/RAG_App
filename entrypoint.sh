#!/bin/bash

# Define the Ollama URL
OLLAMA_HOST_URL="${OLLAMA_BASE_URL:-http://ollama:11434}"

echo "Waiting for Ollama service at $OLLAMA_HOST_URL..."

# Wait for Ollama to be up
until curl -s $OLLAMA_HOST_URL/api/tags > /dev/null; do
    echo "Ollama is not ready... sleeping 5s"
    sleep 5
done

echo "Ollama is online! Pulling required models..."

# 1. Pull Embedding Model
echo "Pulling nomic-embed-text..."
curl -X POST $OLLAMA_HOST_URL/api/pull -d '{"name": "nomic-embed-text"}'

# 2. Pull Default Chat Model (Example: deepseek-r1:7b)
# Note: pulling large models can take time on first run
echo "Pulling gemma3:4b..."
curl -X POST $OLLAMA_HOST_URL/api/pull -d '{"name": "gemma3:latest"}'

echo "Models ready. Starting Streamlit..."

# Start the app
streamlit run app.py --server.address=0.0.0.0