# 🔍 Vector Store Tools 

This directory contains utility scripts for the ScaleX RAG Evaluation Benchmark to create and manage vector stores.

## 🚀 Vector Store Creation Methods

There are multiple ways to create vector stores in this repo:

### 1️⃣ Using Shell Scripts

#### `create_vectorstore.sh` 📜

A script for creating vector stores that works with both local and cloud models.

```bash
# Basic usage (from project root)
./vectorstore_tools/create_vectorstore.sh <data_file> <config_file>

# Examples:
# For OpenAI models
./vectorstore_tools/create_vectorstore.sh data/english/data_source.txt config/llm_cloud_rag.yaml

# For local Ollama models 
./vectorstore_tools/create_vectorstore.sh data/arabic/data_source.txt config/ollama_local_rag.yaml
```

**Features:**
- ✅ Validates data files and configurations
- 🔑 Checks for required API keys
- 🖥️ Automatically starts Ollama if needed
- 📝 Updates configuration with the data source path
- 💾 Creates and saves the vector store
- 🔄 Updates the configuration with the vector store path

### 2️⃣ Using Python Scripts Directly

#### `create_vectorstore.py` 🐍

The core Python script for creating vector stores with more control.

```bash
# Direct use with a configuration file (from project root)
python vectorstore_tools/create_vectorstore.py --config <config_file>
```

This script:
- 📂 Loads documents from the source path
- ✂️ Splits text into chunks with specified size and overlap
- 🧮 Embeds text using the configured embedding model
- 💾 Creates a FAISS vector store and saves it

### 3️⃣ Using the RAG Evaluation Pipeline

#### `rag_evaluation_pipeline.py` 🔄

A comprehensive pipeline that includes vector store creation as part of the RAG workflow.

```bash
# Create vector store only (from project root)
python rag_tools/rag_evaluation_pipeline.py --config <config_file> --skip-evaluation

# Create vector store and run evaluation
python rag_tools/rag_evaluation_pipeline.py --config <config_file>

# Use existing vector store (skip setup)
python rag_tools/rag_evaluation_pipeline.py --config <config_file> --skip-setup

# Force rebuild of vector store
python rag_tools/rag_evaluation_pipeline.py --config <config_file> --force-rebuild
```

### 4️⃣ Using Convenient Shell Wrappers

```bash
# For cloud-based models (OpenAI, Anthropic, etc.)
./rag_tools/run_cloud_rag_evaluation.sh [--skip-setup] [--force-rebuild]

# For local models (Ollama)
./rag_tools/run_local_rag_evaluation.sh [--skip-setup] [--force-rebuild]
```

## 📚 Configuration Requirements

Your configuration file needs to include these sections:

```yaml
dataset:
  name: english_qa  # Name for your dataset

retrieval:
  embedding_model: openai  # or 'ollama', etc.
  chunk_size: 1000
  chunk_overlap: 200

vectorstore:
  source_path: data/english/data_source.txt  # Path to your source data
  output_dir: ./vectorstores/english  # Where to save the vector store
```

## 🧰 Other Utility Scripts

| Script | Description |
|--------|-------------|
| `vectorstore_tools/create_vectorstore.py` | Core Python script used to create vector stores |
| `rag_tools/rag_evaluation_pipeline.py` | Complete pipeline for RAG evaluation including vector store creation |
| `vectorstore_tools/cleanup_repo.sh` | Utility script to clean up temporary files and caches |

## 📊 Vector Store Directory Structure

After creation, your vector store will be organized as:

```
vectorstores/
  english_txt/
    english_qa_8e3943dfbb/  # Named with dataset and hash of source
      index.faiss           # FAISS index file
      docstore.json         # Document store mapping
      metadata.json         # Information about the vector store
```

To use your vector store in evaluations, make sure your config file includes the `path` entry:

```yaml
vectorstore:
  path: vectorstores/english_txt/english_qa_8e3943dfbb
```

This will be automatically added when using any of the scripts above.

## 💡 Troubleshooting

If you encounter path-related errors:

1. Make sure you're running commands from the project root directory
2. Use absolute paths if needed
3. Check that your data files exist at the specified locations
4. Verify that your configuration file contains valid YAML

