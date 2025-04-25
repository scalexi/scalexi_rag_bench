#!/bin/bash

# Script to rebuild a vector store from an Arabic data source with a new embedding model

# Make sure we're in the project root directory
cd "$(dirname "$0")/.."

# Get the data source path from the user or use default
if [ "$1" != "" ]; then
  DATA_SOURCE="$1"
else
  DATA_SOURCE="data/arabic/data_source.txt"
fi

# Check if the data source exists
if [ ! -f "$DATA_SOURCE" ]; then
  echo "Error: Data source file not found: $DATA_SOURCE"
  echo "Please provide a valid path to a text file as the first argument."
  exit 1
fi

# Check if OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ]; then
  echo "Error: OPENAI_API_KEY environment variable is not set."
  echo "Please set your OpenAI API key using:"
  echo "export OPENAI_API_KEY=your_openai_api_key"
  exit 1
fi

# Check if Ollama is running and start it if needed
if ! curl -s http://localhost:11434/api/version >/dev/null 2>&1; then
  echo "Ollama service is not running. Attempting to start Ollama..."
  if command -v ollama >/dev/null 2>&1; then
    # Start ollama in the background
    ollama serve > /dev/null 2>&1 &
    OLLAMA_PID=$!
    echo "Started Ollama with PID: $OLLAMA_PID"
    # Give it time to start
    sleep 5
    # Check if model is available
    if ! ollama list | grep -q "gemma3:4b"; then
      echo "Pulling required model gemma3:4b..."
      ollama pull gemma3:4b
    fi
  else
    echo "ERROR: Ollama is not installed or not in your PATH."
    echo "Please install Ollama from https://ollama.ai/download"
    exit 1
  fi
else
  echo "Ollama service is running."
  # Check if model is available
  if ! ollama list | grep -q "gemma3:4b"; then
    echo "Pulling required model gemma3:4b..."
    ollama pull gemma3:4b
  fi
fi

echo "Creating a fresh config for the new embedding model..."
CONFIG_PATH="config/arabic_vectorstore_rebuild.yaml"

# Create a new config file with the desired settings
cat > "$CONFIG_PATH" << EOL
experiment_name: Arabic Evaluation with Gemma3
description: Evaluation on Arabic dataset with OpenAI embeddings

retrieval:
  retriever_type: vector
  embedding_model: text-embedding-3-small
  k: 4
  chunk_size: 1000
  chunk_overlap: 200
  search_type: similarity

llm:
  model_name: gemma3:4b
  provider: ollama
  temperature: 0.0
  max_tokens: 512

dataset:
  name: arabic_qa
  language: arabic
  path: data/arabic/qa_dataset.json
  format: json

vectorstore:
  source_path: ${DATA_SOURCE}
  output_dir: ./vectorstores/arabic_qa_new
  path: null  # Will be populated after vector store creation

evaluation_metrics:
  retrieval_metrics:
    - precision_at_k
    - recall_at_k
    - mrr
    - ndcg
  generation_metrics:
    - correctness
    - relevance
    - groundedness
    - coherence
    - conciseness
  system_metrics:
    - latency

output_dir: ./results/arabic_gemma3
EOL

echo "Removing old vector store if it exists..."
rm -rf ./vectorstores/arabic_qa_new

# Create the vector store
echo "Creating vector store from Arabic data source with text-embedding-3-small..."
python scripts/create_vectorstore.py --config "$CONFIG_PATH"

# Check if the vector store was created successfully
if [ $? -ne 0 ]; then
  echo "Failed to create vector store. Exiting."
  exit 1
fi

# Run the evaluation
echo "Running evaluation..."
python examples/05_vector_store_setup.py --config "$CONFIG_PATH" --skip-setup

echo "Done!" 