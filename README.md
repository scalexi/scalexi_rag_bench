# 🚀 RAG Evaluation Framework

A comprehensive framework for evaluating Retrieval-Augmented Generation (RAG) systems across different models, languages, and datasets.

## 🔍 Overview

This repository provides tools and methodologies to benchmark RAG applications with a focus on:

- Multiple LLM support (commercial and open-source)
- Multilingual evaluation (English, Arabic, and others)
- Comprehensive metrics for retrieval and generation
- Standardized datasets and evaluation protocols

## ✨ Key Features

- **🤖 Model Agnostic**: Test DeepSeek, OpenAI, Cohere, Gemma, and other LLMs
- **🌐 Multilingual Support**: Evaluate RAG performance across languages
- **📊 Comprehensive Metrics**: Measure retrieval quality, generation quality, and overall system performance
- **🧩 Configurable Pipeline**: Mix and match components (embeddings, retrievers, LLMs)
- **🔄 Reproducible Evaluation**: Standardized datasets and evaluation protocols

## 📂 Repository Structure

```
scalexi_rag_bench/
├── config/                  # Configuration files for different evaluation setups
├── data/                    # Sample datasets and data processing utilities
│   ├── english/            # English datasets
│   ├── arabic/             # Arabic datasets
├── scalexi_rag_bench/     # Core evaluation framework
│   ├── config/             # Configuration classes and utilities
│   ├── evaluators/         # Evaluation metrics and utilities
│   ├── models/             # Model adapters and utilities
│   ├── retrievers/         # Retriever implementations
│   ├── utils/              # Utility functions
├── examples/               # Example notebooks and scripts
├── results/                # Results from evaluation runs
├── comparisons/            # Comparison charts and data
└── utils/                  # Utility scripts
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Ollama (for local model running)
- API keys for commercial models

### 🔑 Required API Keys

The framework uses several services that require API keys:

1. **LangSmith API Key** (required for evaluation):
   ```bash
   export LANGCHAIN_API_KEY=your_langsmith_api_key_here
   export LANGCHAIN_PROJECT=your_langsmith_project_name  # Optional
   ```
   LangSmith is used for tracking evaluations, metrics, and experiment results.
   [Sign up here](https://smith.langchain.com/)

2. **Model-specific API Keys** (depending on your configuration):
   ```bash
   # OpenAI
   export OPENAI_API_KEY=your_openai_api_key_here
   
   # Cohere
   export COHERE_API_KEY=your_cohere_api_key_here
   
   # Anthropic (for Claude models)
   export ANTHROPIC_API_KEY=your_anthropic_api_key_here
   ```

You only need the API keys for the specific models you plan to evaluate. Local models (via Ollama) don't require API keys.

### 📦 Installation

```bash
git clone https://github.com/yourusername/scalexi_rag_bench.git
cd scalexi_rag_bench
pip install -r requirements.txt
```

### 🧠 Quick Start

```python
from scalexi_rag_bench import RAGEvaluator
from scalexi_rag_bench.config.config import Config

# Load config for evaluation
config = Config.from_file("config/llm_cloud_rag.yaml")

# Initialize evaluator
evaluator = RAGEvaluator(config)

# Run evaluation
results = evaluator.evaluate()

# Generate report
evaluator.generate_report(results, "results/evaluation_report.html")
```

## 🔬 Evaluation Examples

Here are concrete examples for evaluating different models:

### 1️⃣ Evaluating OpenAI Models in the Cloud

This example demonstrates how to evaluate a RAG system using OpenAI's GPT-4 model.

1. Set up your API key:
```bash
export OPENAI_API_KEY=your_api_key_here
export LANGCHAIN_API_KEY=your_langsmith_api_key_here
```

2. Run the evaluation using the provided script:
```bash
python examples/03_run_llm_cloud_evaluation.py --config config/llm_cloud_rag.yaml
```

The configuration for this evaluation is:

```yaml
experiment_name: English GPT-4 Evaluation
description: Evaluation of GPT-4 on English dataset with OpenAI embeddings

retrieval:
  retriever_type: vector  # Uses FAISS for in-memory vector storage
  embedding_model: text-embedding-3-large
  k: 4
  chunk_size: 1000
  chunk_overlap: 200
  search_type: similarity

llm:
  model_name: gpt-4o
  provider: openai
  temperature: 0.0
  max_tokens: 1024

# ... additional configuration settings
```

### 2️⃣ Evaluating Local Models with Ollama

This example shows how to run a RAG evaluation using local models with Ollama, which is ideal for testing without API costs.

1. Start Ollama and pull the desired model:
```bash
ollama pull gemma3:4b
```
2. Run the evaluation:
```bash
python examples/04_run_ollama_local_evaluation.py --config config/ollama_local_rag.yaml
```

The configuration for Arabic evaluation with a local model:

```yaml
experiment_name: Arabic Evaluation with Gemma3
description: Evaluation on Arabic dataset with multilingual embeddings

retrieval:
  retriever_type: vector
  embedding_model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
  k: 4
  chunk_size: 1000
  chunk_overlap: 200
  search_type: similarity

llm:
  model_name: gemma3:4b
  provider: ollama
  temperature: 0.0
  max_tokens: 512

# ... additional configuration settings
```

### 3️⃣ Custom Evaluations and Advanced Features

For more advanced usage, you can use the detailed LangSmith integration:

```bash
python examples/02_langsmith_evaluation.py --config path/to/custom_config.yaml
```

This allows for:
- ✅ Detailed tracking of evaluation runs
- ✅ Interactive dashboards to analyze results
- ✅ Advanced metrics computation and visualization

### 4️⃣ Comparing Results Across Models

After running evaluations, compare the performance of different models:

```bash
python examples/08_simple_langsmith_comparison.py --experiment-ids exp1_id exp2_id --names "GPT-4" "Gemma3"
```

This will generate:
- 📈 Comparison charts of key metrics
- 📊 Performance tables across all evaluation dimensions
- 🔍 Detailed analysis of strengths and weaknesses

## 🛠️ Customization Options

### Vector Store Options

The framework supports different vector stores for retrieval:

1. **FAISS** (default with `retriever_type: vector`):
   - ⚡ In-memory vector store, faster for smaller datasets
   - 🔄 No persistence between runs by default
   - 👍 Better for quick evaluations and smaller document collections
   
2. **Chroma** (enabled with `retriever_type: chroma`):
   - 💾 Persistent storage by default
   - 📚 Better for larger document collections
   - 🔍 More metadata filtering capabilities
   - 📂 Supports collection management

Example Chroma configuration:
```yaml
retrieval:
  retriever_type: chroma
  embedding_model: text-embedding-3-large
  k: 4
  chunk_size: 1000
  chunk_overlap: 200
  # Chroma-specific settings
  collection_name: my_custom_collection
  persist_directory: ./chroma_db
```

### Hybrid Retrieval

For improved retrieval quality, use hybrid retrieval to combine semantic and keyword search:

```yaml
retrieval:
  retriever_type: hybrid
  embedding_model: text-embedding-3-large
  k: 4
  
  # Hybrid retrieval weights
  hybrid_weights:
    vector: 0.7  # Weight for vector similarity
    bm25: 0.3    # Weight for keyword-based retrieval
```

## 📏 Supported Metrics

### Retrieval Metrics
- 🎯 Precision@K
- 📈 Recall@K
- 🏆 MRR (Mean Reciprocal Rank)
- 📊 NDCG (Normalized Discounted Cumulative Gain)

### Generation Metrics
- ✅ Correctness
- 🔗 Relevance
- 🧱 Groundedness
- 📝 Coherence
- 📌 Conciseness

### System Metrics
- ⏱️ Latency
- 🚀 Throughput
- 💰 Cost

## 🤖 Supported Models

### LLMs
- OpenAI (GPT-4, GPT-3.5)
- DeepSeek Models
- Cohere (Command R+, Command R7B-Arabic)
- Gemma3 Models
- Claude (Opus, Sonnet)
- Llama 3 Models

### Embedding Models
- OpenAI Embeddings
- Sentence Transformers
- BGE Embeddings
- Cohere Embeddings
- E5 Embeddings

## 🌐 Supported Languages
- English
- Arabic
- (More to be added)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- LangChain for RAG components
- LangSmith for evaluation infrastructure
- Hugging Face for model access
- Ollama for local model hosting 
