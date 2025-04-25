# Configuration Directory

This directory contains YAML configuration files for different evaluation scenarios in the ScaleX RAG Evaluation Benchmark.

## Configuration Files

| File | Description |
|------|-------------|
| `llm_cloud_rag.yaml` | Base configuration for evaluating RAG with OpenAI's GPT models and OpenAI embeddings |
| `ollama_local_rag.yaml` | Configuration for evaluating RAG with local Ollama models and OpenAI embeddings |
| `text_vectorstore_config.yaml` | Configuration specifically designed for text file-based vector stores |

## Configuration Structure

Each configuration file contains sections for:
- Dataset configuration
- LLM settings
- Retrieval parameters
- Vector store settings (REQUIRED)
- Evaluation metrics
- Output directories

### Required Vector Store Configuration

Every configuration file MUST include a `vectorstore` section with at least:
- `source_path`: Path to the data source for creating the vector store
- `output_dir`: Directory to save the vector store

Example:
```yaml
vectorstore:
  source_path: data/english/data_source.txt  # REQUIRED
  output_dir: ./vectorstores/english_txt     # Recommended
  path: null  # Will be populated after vector store creation
```

After creating a vector store, the configuration will be updated with:
```yaml
vectorstore:
  source_path: data/english/data_source.txt
  output_dir: ./vectorstores/english_txt
  path: ./vectorstores/english_txt/english_qa_52ceadd2e9  # Populated by the system
```

### Note on Embedding Models

We currently recommend OpenAI embeddings for both English and multilingual content:

- `text-embedding-3-large`: Higher quality embeddings for production systems
- `text-embedding-3-small`: Cost-effective option with excellent quality

While we previously used HuggingFace models for multilingual support, OpenAI's embedding models now offer superior multilingual capabilities.

## Example Configuration

```yaml
experiment_name: My RAG Evaluation
description: Evaluation of RAG system with my data

# Retrieval settings
retrieval:
  retriever_type: vector
  embedding_model: text-embedding-3-large
  k: 4
  chunk_size: 1000
  chunk_overlap: 200
  search_type: similarity

# LLM settings
llm:
  model_name: gpt-4o
  provider: openai
  temperature: 0.0
  max_tokens: 1024

# Dataset for evaluation
dataset:
  name: my_evaluation_dataset
  language: english
  path: data/my_evaluation_dataset.json
  format: json

# Vector store settings
vectorstore:
  source_path: data/my_knowledge_source.txt
  output_dir: ./vectorstores/my_knowledge

# Evaluation metrics and output settings
evaluation_metrics:
  # ... metrics configuration ...
output_dir: ./results/my_evaluation
```

## Usage

To use a configuration file, specify its path when running evaluation scripts:

```bash
python examples/05_vector_store_setup.py --config config/llm_cloud_rag.yaml
```

Or when creating a vector store:

```bash
python scripts/create_vectorstore.py --config config/text_vectorstore_config.yaml
``` 