#!/bin/bash

# Script to create a vector store from an Arabic data source and run evaluation

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

# Create a temporary config based on the original but with the new data source
CONFIG_PATH="config/arabic_vectorstore_config.yaml"
cp config/ollama_local_rag.yaml "$CONFIG_PATH"

# Update the config with the data source path
echo "Updating configuration to use data source: $DATA_SOURCE"
sed -i.bak "s|source_path:.*|source_path: $DATA_SOURCE|g" "$CONFIG_PATH"
rm "${CONFIG_PATH}.bak" 2>/dev/null || true

# Create the vector store
echo "Creating vector store from Arabic data source..."
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