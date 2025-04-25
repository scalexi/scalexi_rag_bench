# Scripts Directory

This directory contains utility scripts for the ScaleX RAG Evaluation Benchmark.

## Shell Scripts

| Script | Description |
|--------|-------------|
| `create_text_vectorstore.sh` | Creates a vector store from a text file data source and sets up the configuration for evaluation |
| `create_arabic_vectorstore.sh` | Creates a vector store from an Arabic text file data source for evaluation with local models. Automatically checks for and starts Ollama if needed. |
| `rebuild_arabic_vectorstore.sh` | Rebuilds the Arabic vector store with a new embedding model to fix dimension mismatch errors. Use this if you get an AssertionError about dimension mismatches. |
| `run_vectorstore_setup.sh` | Runs the evaluation using an existing vector store configuration |
| `update_repo.sh` | Utility script for updating the repository with the latest changes |
| `cleanup_repo.sh` | Utility script for cleaning up temporary and backup files in the repository |

## Python Scripts

| Script | Description |
|--------|-------------|
| `create_vectorstore.py` | Core script for creating vector stores from various data sources with validation and configuration updating |

## Usage Examples

### Creating a Vector Store from Text

```bash
# From the project root (for English)
./scripts/create_text_vectorstore.sh data/english/data_source.txt

# From the project root (for Arabic - will automatically check requirements)
./scripts/create_arabic_vectorstore.sh data/arabic/data_source.txt
```

### Requirements for Arabic Evaluation

The Arabic evaluation script (`create_arabic_vectorstore.sh`) performs several checks:
- Verifies the OpenAI API key is set
- Checks if Ollama is running and attempts to start it if not
- Verifies the required model (gemma3:4b) is available or pulls it
- Ensures the data source file exists

### Fixing Vector Store Errors

If you encounter an `AssertionError` about dimension mismatches when using the vector store, rebuild it with the new embedding model:

```bash
# Rebuilds the vector store with new embedding model
./scripts/rebuild_arabic_vectorstore.sh data/arabic/data_source.txt
```

This creates a clean vector store with the new embedding model and updates the configuration accordingly.

### Running an Evaluation with an Existing Vector Store

```bash
# From the project root
./scripts/run_vectorstore_setup.sh
```

### Cleaning Up the Repository

```bash
# Clean up backup and temporary files
./scripts/cleanup_repo.sh

# Also remove temporary configuration files
./scripts/cleanup_repo.sh --clean-configs
``` 