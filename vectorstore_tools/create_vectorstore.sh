#!/bin/bash

# Script to create vector stores from text data
# Usage: ./vectorstore_tools/create_vectorstore.sh <data_file> <config_file>

# Make sure we're in the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

# Check for required arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <data_file> <config_file>"
    echo "Example: $0 data/english/data_source.txt config/llm_cloud_rag.yaml"
    exit 1
fi

DATA_SOURCE="$1"
CONFIG_FILE="$2"

# Check if data source exists
if [ ! -f "$DATA_SOURCE" ]; then
    echo "Error: Data source file not found: $DATA_SOURCE"
    echo "Please provide a valid path to a text file."
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    echo "Please provide a valid path to a YAML config file."
    exit 1
fi

# Check for OpenAI API key if needed
if grep -q "provider: openai" "$CONFIG_FILE"; then
    if [ -z "$OPENAI_API_KEY" ]; then
        echo "Warning: OPENAI_API_KEY environment variable is not set."
        echo "Please set your OpenAI API key using:"
        echo "export OPENAI_API_KEY=your_openai_api_key"
        exit 1
    fi
fi

# Check for Ollama if needed
if grep -q "provider: ollama" "$CONFIG_FILE"; then
    # Extract model name from config
    MODEL_NAME=$(grep "model_name:" "$CONFIG_FILE" | awk '{print $2}')
    
    # Check if Ollama is running
    if ! curl -s http://localhost:11434/api/version >/dev/null 2>&1; then
        echo "Ollama service is not running. Attempting to start Ollama..."
        if command -v ollama >/dev/null 2>&1; then
            # Start ollama in the background
            ollama serve > /dev/null 2>&1 &
            OLLAMA_PID=$!
            echo "Started Ollama with PID: $OLLAMA_PID"
            # Give it time to start
            sleep 5
        else
            echo "ERROR: Ollama is not installed or not in your PATH."
            echo "Please install Ollama from https://ollama.ai/download"
            exit 1
        fi
    fi
    
    # Check if model is available
    if ! ollama list | grep -q "$MODEL_NAME"; then
        echo "Pulling required model $MODEL_NAME..."
        ollama pull "$MODEL_NAME"
    fi
fi

# Update config file with data source path
tmp_file=$(mktemp)
awk -v data="$DATA_SOURCE" '
/vectorstore:/ { print; in_vs=1; next }
in_vs && /source_path:/ { print "  source_path: " data; next }
{ print }
' "$CONFIG_FILE" > "$tmp_file"

# If source_path doesn't exist, add it
if ! grep -q "source_path:" "$tmp_file"; then
    awk '
    /vectorstore:/ { print; print "  source_path: '"$DATA_SOURCE"'"; next }
    { print }
    ' "$tmp_file" > "${tmp_file}.new"
    mv "${tmp_file}.new" "$tmp_file"
fi

# Update the config file
mv "$tmp_file" "$CONFIG_FILE"

echo "Creating vector store from: $DATA_SOURCE"
echo "Using configuration: $CONFIG_FILE"

# Create the vector store
python "$ROOT_DIR/vectorstore_tools/create_vectorstore.py" --config "$CONFIG_FILE"

if [ $? -eq 0 ]; then
    echo "Vector store created successfully!"
    echo "You can now run evaluations with:"
    echo "python rag_tools/rag_evaluation_pipeline.py --config $CONFIG_FILE --skip-setup"
else
    echo "Failed to create vector store."
    exit 1
fi 