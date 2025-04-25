#!/bin/bash

# Script to create a vector store from the dataset and run evaluation

# Make sure we're in the project root directory
cd "$(dirname "$0")"

# Create the vector store
echo "Creating vector store..."
python scripts/create_vectorstore.py --config config/llm_cloud_rag.yaml

# Check if the vector store was created successfully
if [ $? -ne 0 ]; then
  echo "Failed to create vector store. Exiting."
  exit 1
fi

# Get the updated config path
CONFIG_PATH="config/llm_cloud_rag_updated.yaml"
if [ -f "$CONFIG_PATH" ]; then
  echo "Using updated config file: $CONFIG_PATH"
  
  # Make a copy of the updated config to the original location
  echo "Backing up original config and copying updated config..."
  cp config/llm_cloud_rag.yaml config/llm_cloud_rag.yaml.bak
  cp "$CONFIG_PATH" config/llm_cloud_rag.yaml
  echo "Config file updated successfully."
else
  echo "WARNING: Updated config file not found. Vector store path may not be properly configured."
  CONFIG_PATH="config/llm_cloud_rag.yaml"
fi

# Run the evaluation
echo "Running evaluation..."
python examples/05_vector_store_setup.py --config config/llm_cloud_rag.yaml --skip-setup

echo "Done!" 