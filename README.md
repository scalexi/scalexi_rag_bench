# ScaleX RAG Evaluation Benchmark

A comprehensive framework for evaluating Retrieval-Augmented Generation (RAG) systems with support for different LLM providers, retrieval methods, and evaluation metrics.

## Quick Start

```bash
# Create a vector store from text data and run evaluation
./scripts/create_text_vectorstore.sh

# To specify a custom text source
./scripts/create_text_vectorstore.sh path/to/your/text_file.txt
```

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Complete Evaluation Workflow](#complete-evaluation-workflow)
  - [Step 1: Prepare Data Sources](#step-1-prepare-data-sources)
  - [Step 2: Configure Your Evaluation](#step-2-configure-your-evaluation)
  - [Step 3: Create a Vector Store](#step-3-create-a-vector-store)
  - [Step 4: Run the Evaluation](#step-4-run-the-evaluation)
- [Configuration Reference](#configuration-reference)
- [Understanding Metrics](#understanding-metrics)
- [Troubleshooting](#troubleshooting)

## Introduction

The ScaleX RAG Evaluation Benchmark provides tools to evaluate Retrieval-Augmented Generation (RAG) systems. RAG combines the power of external knowledge retrieval with language model generation to create more accurate, up-to-date, and factual responses.

This framework helps you:
- Create vector stores from text, PDF, or JSON sources
- Configure and run comprehensive evaluations
- Measure performance using industry-standard metrics
- Compare different retrieval and generation strategies

## Features

- **Multi-source Vector Store Creation**: Create vector stores from text files, PDFs, or JSON datasets
- **Multiple Retrieval Methods**: Support for dense (embedding-based), sparse (BM25), and hybrid retrieval
- **Comprehensive Metrics**: Evaluate both retrieval quality (MRR, NDCG, precision@k, recall@k) and generation quality (correctness, relevance, groundedness)
- **LangSmith Integration**: Seamless integration with LangSmith for detailed evaluation insights
- **Configurable Pipeline**: Easy adjustment of parameters like chunk size, embedding models, and LLM settings

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/scalexi_rag_bench.git
cd scalexi_rag_bench
```

2. Set up your environment:
```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

3. Set up your API keys:
```bash
# Set OpenAI API key
export OPENAI_API_KEY=your_openai_api_key

# Set LangSmith API key (for evaluation)
export LANGCHAIN_API_KEY=your_langsmith_api_key
export LANGCHAIN_PROJECT=your_langsmith_project
```

### Additional Requirements

#### For OpenAI Embeddings
All configurations in this repository use OpenAI embeddings by default, so an OpenAI API key is required:
```bash
export OPENAI_API_KEY=your_openai_api_key
```

#### For Local LLM Evaluation (Optional)
If you want to use the Ollama-based configurations for local LLM evaluation:
1. Install Ollama from [https://ollama.ai/download](https://ollama.ai/download)
2. Start the Ollama service:
   ```bash
   ollama serve
   ```
3. Pull the required model:
   ```bash
   ollama pull gemma3:4b
   ```

> **Note:** The `create_arabic_vectorstore.sh` script will automatically check if Ollama is running and attempt to start it and pull the required model if needed.

## Complete Evaluation Workflow

### Step 1: Prepare Data Sources

You need two types of data:

1. **Knowledge Source** (for vector store creation):
   - Text file(s) containing domain knowledge (e.g., `data/english/data_source.txt`)
   - PDF documents
   - Any raw knowledge that would be used by your RAG system

2. **Evaluation Dataset** (for testing retrieval and generation quality):
   - JSON file with questions, ground truth answers, and relevant documents
   - Format example:
   ```json
   [
     {
       "question": "What is Retrieval-Augmented Generation?",
       "answer": "Retrieval-Augmented Generation (RAG) is...",
       "relevant_docs": [
         {
           "content": "Retrieval-Augmented Generation (RAG) combines...",
           "metadata": {
             "source": "rag-overview",
             "start_index": 100
           }
         }
       ]
     }
   ]
   ```

### Step 2: Configure Your Evaluation

Create or modify a YAML configuration file (e.g., `config/my_evaluation.yaml`):

```yaml
experiment_name: My RAG Evaluation
description: Evaluation of RAG system with my data

# Retrieval settings
retrieval:
  retriever_type: vector  # Options: vector, bm25, hybrid
  embedding_model: text-embedding-3-large
  k: 4  # Number of documents to retrieve
  chunk_size: 1000
  chunk_overlap: 200
  search_type: similarity  # Options: similarity, mmr

# LLM settings
llm:
  model_name: gpt-4o  # Or any other model
  provider: openai
  temperature: 0.0
  max_tokens: 1024

# Dataset for evaluation
dataset:
  name: my_evaluation_dataset
  language: english
  path: data/my_evaluation_dataset.json
  format: json

# Vector store settings (REQUIRED)
vectorstore:
  source_path: data/my_knowledge_source.txt  # Path to the knowledge source
  output_dir: ./vectorstores/my_knowledge  # Where to store the vector store

# Metrics to evaluate
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
    - cost

# Output directory for results
output_dir: ./results/my_evaluation
```

### Available Configurations

The repository includes several pre-configured evaluation setups:

| Configuration | Description | Use Case |
|---------------|-------------|----------|
| `config/llm_cloud_rag.yaml` | English evaluation with OpenAI GPT models | Cloud-based evaluation with English content |
| `config/ollama_local_rag.yaml` | Arabic evaluation with local Gemma model | Local evaluation with Arabic content |
| `config/text_vectorstore_config.yaml` | Configuration for text file vector stores | Creating vector stores from plain text files |

You can use these as starting points for your own evaluations:

```bash
# For English evaluation with OpenAI
python scripts/create_vectorstore.py --config config/llm_cloud_rag.yaml

# For Arabic evaluation with local models
python scripts/create_vectorstore.py --config config/ollama_local_rag.yaml
```

### Step 3: Create a Vector Store

Create a vector store from your knowledge source:

```bash
python scripts/create_vectorstore.py --config config/my_evaluation.yaml
```

This will:
1. Read your knowledge source (text, PDF, or JSON)
2. Create and save a vector store
3. Update your configuration with the vector store path

When you create a vector store, the system will:
- Create a unique identifier for the vector store based on a hash of the input data
- Save the vector store in a path like: `./vectorstores/english_qa/english_qa_cef5436db0`
- Automatically update your configuration with this path

The path follows this format: `[output_dir]/[dataset_name]_[unique_hash]`

Alternatively, use our convenience script for text files:

```bash
./scripts/create_text_vectorstore.sh data/my_knowledge_source.txt
```

### Step 4: Run the Evaluation

Run the full evaluation:

```bash
python examples/05_vector_store_setup.py --config config/my_evaluation.yaml --skip-setup
```

Or use our convenience script (after creating the vector store):

```bash
./scripts/run_vectorstore_setup.sh
```

The evaluation will:
1. Use your vector store to retrieve documents for each question
2. Generate answers based on retrieved documents
3. Evaluate retrieval and generation quality using specified metrics
4. Generate an HTML report and save results

## Configuration Reference

### Essential Settings

| Setting | Description | Example |
|---------|-------------|---------|
| `experiment_name` | Name of your evaluation experiment | `"English GPT-4 Evaluation"` |
| `vectorstore.source_path` | Path to knowledge source (REQUIRED) | `"data/english/data_source.txt"` |
| `dataset.path` | Path to evaluation dataset | `"data/english/qa_dataset.json"` |
| `llm.model_name` | LLM model to use | `"gpt-4o"` |
| `retrieval.embedding_model` | Embedding model for vector store | `"text-embedding-3-large"` |

### Retrieval Parameters

| Setting | Description | Default |
|---------|-------------|---------|
| `retrieval.chunk_size` | Size of text chunks | `1000` |
| `retrieval.chunk_overlap` | Overlap between chunks | `200` |
| `retrieval.k` | Number of documents to retrieve | `4` |
| `retrieval.search_type` | Search algorithm | `"similarity"` |
| `retrieval.retriever_type` | Type of retriever | `"vector"` |

### Metrics

| Category | Available Metrics |
|----------|------------------|
| Retrieval | `precision_at_k`, `recall_at_k`, `mrr`, `ndcg` |
| Generation | `correctness`, `relevance`, `groundedness`, `coherence`, `conciseness` |
| System | `latency`, `cost` |

## Understanding Metrics

### Retrieval Metrics
- **Precision@k**: Proportion of retrieved documents that are relevant
- **Recall@k**: Proportion of relevant documents that were retrieved
- **MRR (Mean Reciprocal Rank)**: Position of the first relevant document
- **NDCG (Normalized Discounted Cumulative Gain)**: Relevance-weighted ranking metric

### Generation Metrics
- **Correctness**: Factual accuracy of the response
- **Relevance**: How well the response addresses the question
- **Groundedness**: Whether claims in the response are supported by retrieved documents
- **Coherence**: Logical flow and structure of the response
- **Conciseness**: Appropriate length and focus of the response

## Troubleshooting

### Common Issues

**Error: No vector store path configured**
- Make sure you've created a vector store using `scripts/create_vectorstore.py`
- Check that your config file has a valid `vectorstore.source_path`
- After creation, your config should have a `vectorstore.path` that looks like `./vectorstores/english_qa/english_qa_cef5436db0`

**Error: The de-serialization relies loading a pickle file**
- This is fixed in the latest version by using `allow_dangerous_deserialization=True`
- Update your code or use the provided scripts

**Retrieval metrics (MRR, NDCG) show 0.0**
- Your vector store might not contain documents matching the ground truth
- Ensure your knowledge source contains the information needed to answer questions
- Check that your evaluation dataset's `relevant_docs` have proper metadata

### Getting Help

If you encounter issues:
1. Check the error messages for specific problems
2. Review your configuration files for correctness
3. Ensure your data sources are correctly formatted
4. Open an issue on GitHub with details of your problem

## Advanced Usage: LangSmith Integration

This framework integrates with LangSmith to provide detailed insights into your RAG system's performance. Here's how to leverage this integration for deeper analysis:

### Viewing Evaluation Results in LangSmith

After running an evaluation, you'll see a LangSmith URL in the output. This link takes you to a dashboard with detailed metrics and trace information:

```
Evaluation complete. Results saved to ./results/my_evaluation
LangSmith URL: https://smith.langchain.com/projects/p_abc123/datasets
```

The LangSmith dashboard provides:
- Detailed per-question metrics
- Visualizations of retrieval and generation performance
- Trace views showing exactly what documents were retrieved
- Side-by-side comparisons of ground truth vs. generated answers

### Collecting Benchmark Data

To systematically collect benchmark data for multiple configurations:

1. **Create Experiment Variants**:
   ```bash
   # Run evaluations with different parameters
   python examples/05_vector_store_setup.py --config config/chunk_size_500.yaml
   python examples/05_vector_store_setup.py --config config/chunk_size_1000.yaml
   python examples/05_vector_store_setup.py --config config/chunk_size_1500.yaml
   ```

2. **Use Consistent Naming Conventions**:
   ```yaml
   # In your config files
   experiment_name: RAG-ChunkSize-500
   # Then
   experiment_name: RAG-ChunkSize-1000
   # And
   experiment_name: RAG-ChunkSize-1500
   ```

3. **Record Experiment IDs**:
   Each evaluation run produces an experiment ID in the results.json file. Save these IDs for later comparison.

### Comparing Different Benchmarks

You can compare different benchmarks using the LangSmith UI or programmatically:

#### Using LangSmith UI

1. Go to your LangSmith dashboard
2. Click "Datasets" in the left navigation
3. Select multiple experiments using the checkboxes
4. Click "Compare" to generate side-by-side comparisons

This allows you to visually inspect:
- Which configuration performs best across metrics
- Where each configuration excels or falls short
- Specific questions where performance differs significantly

#### Programmatic Comparison

For programmatic comparison, use the LangSmith Python client:

```python
from langsmith import Client

client = Client()

# Get two experiments by ID
exp1 = client.read_experiment(experiment_id="exp_abc123")
exp2 = client.read_experiment(experiment_id="exp_def456")

# Compare average metrics
metrics = ["precision_at_k", "recall_at_k", "correctness", "latency"]
for metric in metrics:
    print(f"{metric}:")
    print(f"  - Config 1: {exp1.mean_scores.get(metric, 'N/A')}")
    print(f"  - Config 2: {exp2.mean_scores.get(metric, 'N/A')}")
    print(f"  - Difference: {exp2.mean_scores.get(metric, 0) - exp1.mean_scores.get(metric, 0)}")
    print()
```

### Deriving Insights

To extract actionable insights from your benchmarks:

1. **Identify Key Performance Drivers**:
   - Compare chunk sizes to find the optimal balance between context window usage and retrieval quality
   - Test different embedding models to identify the best semantic understanding
   - Evaluate retrieval approaches (vector vs. hybrid) to see which works best for your content

2. **Analyze Per-Question Performance**:
   ```python
   # Get detailed run results for specific questions
   example_id = "example_12345"  # ID of a specific question
   
   # Get runs for this example across experiments
   runs1 = client.list_runs(experiment_id=exp1.id, example_id=example_id)
   runs2 = client.list_runs(experiment_id=exp2.id, example_id=example_id)
   
   # Compare the actual retrieved documents
   for run in runs1:
       print(f"Config 1 retrieved documents: {run.inputs.get('context', [])}")
   
   for run in runs2:
       print(f"Config 2 retrieved documents: {run.inputs.get('context', [])}")
   ```

3. **Identify Error Patterns**:
   Use LangSmith to categorize where your RAG system fails:
   - Retrieval failures (relevant documents not found)
   - Generation failures (relevant documents retrieved but answer incorrect)
   - Combined failures (both retrieval and generation issues)

4. **Create Custom Visualizations**:
   ```python
   import matplotlib.pyplot as plt
   import pandas as pd
   
   # Create a DataFrame with metrics from multiple experiments
   experiments = [exp1, exp2, exp3]  # Your experiment objects
   data = []
   
   for exp in experiments:
       exp_data = {
           "name": exp.name,
           "precision": exp.mean_scores.get("precision_at_k", 0),
           "recall": exp.mean_scores.get("recall_at_k", 0),
           "correctness": exp.mean_scores.get("correctness", 0),
           "latency": exp.mean_scores.get("latency", 0)
       }
       data.append(exp_data)
   
   df = pd.DataFrame(data)
   
   # Create visualization
   plt.figure(figsize=(10, 6))
   metrics = ["precision", "recall", "correctness"]
   
   for i, exp_name in enumerate(df["name"]):
       values = df.loc[i, metrics].values
       plt.plot(metrics, values, marker='o', label=exp_name)
   
   plt.title("RAG Performance Comparison")
   plt.ylabel("Score")
   plt.legend()
   plt.grid(True)
   plt.savefig("benchmark_comparison.png")
   ```

By systematically comparing different configurations, you can identify the optimal settings for your specific use case and continually improve your RAG system's performance.
