#!/bin/bash

# Script to create a vector store from data_source.txt and run evaluation

# Make sure we're in the project root directory
cd "$(dirname "$0")"

# Get the data source path from the user or use default
if [ "$1" != "" ]; then
  DATA_SOURCE="$1"
else
  DATA_SOURCE="data/english/data_source.txt"
fi

# Check if the data source exists
if [ ! -f "$DATA_SOURCE" ]; then
  echo "Error: Data source file not found: $DATA_SOURCE"
  echo "Please provide a valid path to a text file as the first argument."
  exit 1
fi

# Create a temporary config based on the original but with the new data source
CONFIG_PATH="config/text_vectorstore_config.yaml"
cp config/llm_cloud_rag.yaml "$CONFIG_PATH"

# Update the config with the data source path
echo "Updating configuration to use data source: $DATA_SOURCE"
sed -i.bak "s|source_path:.*|source_path: $DATA_SOURCE|g" "$CONFIG_PATH"
rm "${CONFIG_PATH}.bak" 2>/dev/null || true

# Create the vector store
echo "Creating vector store from text data source..."
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