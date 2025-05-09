# ScaleXI RAG Benchmark Framework - No-Code RAG Evaluation

A low-code, comprehensive framework for evaluating and comparing Retrieval-Augmented Generation (RAG) systems across different models, embedding techniques, and datasets. Configure your evaluation through YAML files without writing a single line of code.

## 🌟 Key Features

- **Low-Code Solution**: Configure your entire RAG evaluation pipeline through simple YAML files
- **Dual Deployment Options**: Run evaluations on cloud-based models or locally with Ollama
- **Comprehensive Metrics**: Evaluate both retrieval quality and generation accuracy
- **Seamless LangSmith Integration**: Sync results directly with LangChain's LangSmith for advanced analytics 
- **Multilingual Support**: Test RAG systems on English, Arabic, and other languages
- **Visualized Reports**: Interactive HTML dashboards for easy result interpretation

Developed by [ScaleXI Innovation](https://scalexi.ai/), specialists in Generative AI and Large Language Models solutions.

## 🌟 Project Structure

```
scalexi_rag_bench/
├── config/                   # Configuration files for RAG evaluations
├── data/                     # Test datasets for evaluation
│   ├── english/              # English language datasets
│   └── arabic/               # Arabic language datasets
├── examples/                 # Example scripts for different evaluation scenarios
│   └── results/              # Results from example evaluations
├── rag_tools/                # Core evaluation scripts and pipelines
│   └── results/              # Results from RAG evaluations
├── results/                  # Evaluation results by model and language
│   ├── langsmith_evaluation/ # Results from LangSmith evaluations
│   └── minimal_evaluation/   # Results from minimal evaluations
├── scalexi_rag_bench/        # Core framework implementation
│   ├── config/               # Configuration handling
│   ├── evaluators/           # Evaluation metrics and implementations
│   ├── models/               # Model adapters for LLMs and embeddings
│   │   ├── embedding_adapters/ # Embedding model implementations
│   │   └── llm_adapters/     # LLM model implementations
│   ├── retrievers/           # Retrieval mechanisms
│   └── utils/                # Utility functions
├── utils/                    # General utility scripts
├── vectorstore_tools/        # Tools for creating and managing vector stores
└── vectorstores/             # Storage for generated vector databases
    ├── english_txt/          # Vector stores for English content
    └── arabic_qa/            # Vector stores for Arabic content
```

## 🚀 Quick Start

The toolkit provides three main ways to run RAG evaluations:

### 1. Cloud-based RAG Evaluation

For evaluating RAG systems using cloud-based models like OpenAI's GPT models:

```bash
./rag_tools/run_cloud_rag_evaluation.sh
```

### 2. Local RAG Evaluation

For evaluating RAG systems using local models via Ollama:

```bash
./rag_tools/run_local_rag_evaluation.sh
```

### 3. Direct Pipeline Access

For more control over the evaluation process:

```bash
python rag_tools/rag_evaluation_pipeline.py --config <config_file>
```

## ⚙️ Cloud vs. Local Evaluation

The toolkit supports two primary modes of evaluation:

### Cloud-based Evaluation
- Uses cloud-based LLMs (OpenAI, Anthropic, etc.)
- Higher accuracy but requires API keys and has associated costs
- Better suited for production-grade evaluations
- Provides advanced metrics including cost tracking

### Local Evaluation
- Uses local models through Ollama
- Free to use and completely private
- Lower resource requirements but may have reduced performance
- Ideal for development and testing

## 📝 Configuration

Configuration files in YAML format control the entire evaluation pipeline:

```yaml
dataset:
  format: json                 # Dataset format
  name: my_dataset             # Dataset name
  path: data/my_dataset.json   # Path to dataset file

description: My RAG Evaluation   # Description of the evaluation

evaluation_metrics:
  generation_metrics:          # Which generation metrics to use
    - correctness
    - relevance
    - groundedness
  retrieval_metrics:           # Which retrieval metrics to use
    - precision_at_k
    - recall_at_k
  system_metrics:              # System performance metrics
    - latency

llm:
  model_name: gpt-4o-mini      # LLM model to use
  provider: openai             # Model provider (openai, anthropic, ollama)
  temperature: 0.0             # Temperature for generation

retrieval:
  chunk_size: 1000             # Size of text chunks
  chunk_overlap: 200           # Overlap between chunks
  embedding_model: text-embedding-3-large  # Embedding model
  k: 4                         # Number of documents to retrieve

vectorstore:
  output_dir: ./vectorstores/my_dataset  # Where to save the vector store
  source_path: data/source.txt           # Source documents to index
```

## 🔍 How It Works

The ScaleXI RAG Benchmark Framework is designed to provide a low-code approach to RAG evaluation with comprehensive insights:

1. **Zero-Code Configuration**
   - Simply define your evaluation parameters in a YAML configuration file
   - No Python coding required - the framework handles all the implementation details
   - Modify parameters and re-run evaluations to quickly iterate and optimize

2. **Preparing Your Knowledge Base**
   - Your source documents (specified in `source_path` in your config) are loaded
   - These can be text files, PDFs, or other supported formats in directories like `data/english/` or `data/arabic/`
   - Documents are split into smaller chunks (controlled by `chunk_size` and `chunk_overlap` in your config)
   - For example, with `chunk_size: 1000` and `chunk_overlap: 200`, a 3000-word document becomes ~4 overlapping chunks

3. **Creating the Vector Database**
   - Each document chunk is converted into a numerical vector using the embedding model (like `text-embedding-3-large`)
   - These vectors capture the semantic meaning of your text
   - Vectors are stored in a searchable database (FAISS) at the location specified in `vectorstore.output_dir`
   - You can see examples of these in the `vectorstores/` directory

4. **Retrieval Process**
   - When a query/question is processed, it's also converted to a vector using the same embedding model
   - The system finds the most similar document chunks by comparing vector similarity
   - The number of chunks retrieved is controlled by the `k` parameter in your config (e.g., `k: 4` retrieves the 4 most relevant chunks)

5. **Generation with Context**
   - The retrieved document chunks are provided as context to the LLM specified in your config (e.g., `gpt-4o-mini`)
   - The LLM generates an answer based on this context and the question
   - The model, temperature, and other generation parameters are controlled through your config file

6. **Evaluation and Reporting**
   - The system compares generated answers against reference answers from your dataset
   - Multiple metrics are calculated based on your configuration (correctness, relevance, etc.)
   - Results are saved to the directory structure in `results/` organized by model and language
   - Interactive HTML reports make it easy to analyze performance
   - Optional seamless integration with LangSmith for deeper analytics

To adapt this to your own data:
- Place your source documents in a directory (similar to `data/english/` or `data/arabic/`)
- Create a test dataset with questions and expected answers (see format examples in existing datasets)
- Update your config file to point to your new data sources and desired output locations
- Run the evaluation using the appropriate script (`run_cloud_rag_evaluation.sh` or `run_local_rag_evaluation.sh`)

All file paths in your config are relative to the project root, making it easy to organize your custom evaluations.

## 🛠️ Setting Up Your Own Evaluation

Follow these steps to set up and run your own RAG evaluation with custom data:

1. **Prepare Your Knowledge Base Documents**
   - Create a new folder for your documents in the `data/` directory:
     ```bash
     mkdir -p data/your_dataset_name
     ```
   - Add your source documents to this folder (supported formats: TXT, PDF, DOCX, CSV, JSON)
   - Example structure:
     ```
     data/
     ├── english/            # Existing datasets
     ├── arabic/             # Existing datasets
     └── your_dataset_name/  # Your new dataset
         ├── document1.pdf
         ├── document2.txt
         └── ...
     ```

2. **Create Your Test Dataset**
   - Create a JSON file with questions and expected answers in the same directory:
   - Save it as `data/your_dataset_name/questions.json`
   - Format:
     ```json
     [
       {
         "question": "What is the main purpose of X?",
         "answer": "The main purpose of X is...",
         "relevant_docs": ["document1.pdf", "document2.txt"]  # Optional
       },
       {
         "question": "When was Y established?",
         "answer": "Y was established in...",
         "relevant_docs": ["document3.txt"]  # Optional
       }
     ]
     ```
   - The `relevant_docs` field is optional but helps evaluate retrieval performance

3. **Create Configuration File**
   - Copy an existing config file as a starting point:
     ```bash
     cp config/llm_cloud_rag.yaml config/your_evaluation.yaml
     ```
   - Update the following key paths:
     ```yaml
     dataset:
       format: json
       name: your_dataset_name
       path: data/your_dataset_name/questions.json

     description: Your Custom RAG Evaluation

     # ... other settings ...

     retrieval:
       chunk_size: 1000  # Adjust based on your document type
       chunk_overlap: 200
       embedding_model: text-embedding-3-large
       k: 4  # Number of chunks to retrieve

     vectorstore:
       output_dir: ./vectorstores/your_dataset_name
       source_path: data/your_dataset_name  # Points to your documents folder
     ```

4. **Create Output Directories**
   - Ensure your results directory exists:
     ```bash
     mkdir -p results/your_evaluation_name
     ```

5. **Run the Evaluation**
   - For cloud-based models (OpenAI, etc.):
     ```bash
     ./rag_tools/run_cloud_rag_evaluation.sh -f --config config/your_evaluation.yaml
     ```
   - For local models using Ollama:
     ```bash
     ./rag_tools/run_local_rag_evaluation.sh -f --config config/your_evaluation.yaml
     ```
   - The `-f` flag forces rebuilding the vector store (needed first time or when documents change)

6. **Access Your Results**
   - Results will be stored in:
     ```
     results/your_evaluation_name/
     ├── metrics.json        # Overall metrics
     ├── results.json        # Detailed results for each question
     ├── evaluation.html     # Interactive HTML report
     └── raw/                # Raw outputs and intermediate data
     ```
   - Visualize results by opening `results/your_evaluation_name/evaluation.html` in a browser

7. **Iterative Improvement**
   - Analyze results to identify patterns of success or failure
   - Adjust parameters in your config (chunk size, overlap, k, etc.)
   - Rerun evaluation to compare performance

## 📊 LangSmith Integration

The framework features seamless integration with LangChain's LangSmith for detailed evaluation tracking and analytics:

1. **Zero-Configuration Integration**
   - Once LangSmith credentials are set up, all evaluations are automatically tracked
   - No additional coding or setup needed - the framework handles all integration points
   - Simply run your evaluations as normal and results flow to LangSmith in real-time

2. **Set Up LangSmith**
   - Sign up for LangSmith at [smith.langchain.com](https://smith.langchain.com)
   - Set environment variables in your `.env` file or shell:
     ```bash
     export LANGCHAIN_API_KEY=your_api_key
     export LANGCHAIN_TRACING_V2=true
     export LANGSMITH_PROJECT=rag-evaluation
     ```

3. **Comprehensive Analytics**
   - Each evaluation run creates a new experiment in LangSmith
   - Track detailed metrics on retrieval quality, generation accuracy, and cost
   - Visualize performance patterns across different models and configurations
   - Identify bottlenecks and optimization opportunities

4. **Compare Evaluations**
   - Easily compare different RAG configurations side-by-side
   - Analyze how changes to embeddings, chunk sizes, or retrieval settings impact performance
   - Make data-driven decisions about your RAG system architecture

The integration provides enterprise-grade analytics capabilities without requiring any additional development work, allowing you to focus on optimizing your RAG systems rather than building evaluation infrastructure.

## 🧰 Available Components and Options

The toolkit supports a wide range of components that can be configured in your evaluation:

### LLM Models

Configure these in the `llm` section of your config:

```yaml
llm:
  model_name: "gpt-4o-mini"  # The model name
  provider: "openai"         # The provider name
  temperature: 0.0           # Temperature for generation
```

Supported providers:
- **OpenAI** (`provider: "openai"`)
  - Models: `gpt-4o`, `gpt-4o-mini`, `gpt-4`, `gpt-3.5-turbo`, etc.
  - Requires `OPENAI_API_KEY` in environment or config

- **Cohere** (`provider: "cohere"`)
  - Models: `command`, `command-light`, etc.
  - Requires `COHERE_API_KEY` in environment or config

- **Ollama** (`provider: "ollama"`)
  - Models: `llama3`, `gemma`, `mistral`, etc.
  - Default base URL: `http://localhost:11434`
  - Perfect for local evaluation without API costs

- **HuggingFace** (`provider: "huggingface"`)
  - Models: Specify any HuggingFace model compatible with AutoModelForCausalLM
  - Example: `meta-llama/Llama-3-8b-chat-hf`, `google/gemma-7b`, etc.
  - Requires appropriate compute resources

### Embedding Models

Configure these in the `retrieval` section of your config:

```yaml
retrieval:
  embedding_model: "text-embedding-3-large"  # The embedding model to use
```

Supported embedding models:
- **OpenAI** (specify `text-embedding-3-large`, `text-embedding-3-small`, or `text-embedding-ada-002`)
- **Cohere** (specify any model with `cohere` in the name)
- **BGE Models** (specify any model with `bge` in the name)
- **HuggingFace** (any other model name is treated as a HuggingFace model)
- **Multilingual E5** (specify models with `multilingual-e5` in the name for multilingual support)

### Retrievers

Configure retriever type in the `retrieval` section:

```yaml
retrieval:
  retriever_type: "vector"  # The type of retriever to use
  search_type: "similarity"  # Search type for vector store
  k: 4  # Number of documents to retrieve
```

Supported retriever types:
- **Vector** (`retriever_type: "vector"`): Standard vector similarity search using FAISS
- **Chroma** (`retriever_type: "chroma"`): ChromaDB-based retriever for persistent vector storage
- **BM25** (`retriever_type: "bm25"`): Keyword-based retrieval using BM25 algorithm
- **Hybrid** (`retriever_type: "hybrid"`): Combines vector and BM25 retrieval (0.5 weight each)

### Vector Stores

Vector stores are automatically created based on your configuration:

```yaml
vectorstore:
  output_dir: "./vectorstores/your_dataset"  # Where to save the vector store
  source_path: "data/your_dataset"          # Documents to index
```

Current implementation supports:
- **FAISS**: Default for vector retrievers, efficient for similarity search
- **Chroma**: Used when `retriever_type: "chroma"`, supports persistence and metadata filtering

### Search Types

For vector retrievers, you can specify different search algorithms:

```yaml
retrieval:
  search_type: "similarity"  # The search type to use
```

Supported search types:
- **Similarity** (`search_type: "similarity"`): Standard cosine similarity search
- **MMR** (`search_type: "mmr"`): Maximum Marginal Relevance for diversity in results

## 🧩 Additional Tools

The toolkit includes several additional scripts:

- **`vectorstore_tools/create_vectorstore.sh`**: Create vector stores separately
- **`update_repo.sh`**: Update the repository to the latest version
- **`utils/`**: Various utilities for working with the framework

## 📚 Advanced Usage

For advanced usage scenarios, refer to the example scripts in the `examples/` directory:

- **Simple RAG Evaluation**: Basic evaluation workflow
- **LangSmith Integration**: Detailed evaluation with LangSmith
- **Custom Evaluators**: Creating your own evaluation metrics

## 📋 Requirements

The toolkit requires Python 3.8+ and the packages listed in `requirements.txt`:

- langchain
- langchain-openai
- langchain-community  
- langchain-chroma
- langchain-core
- langgraph
- langsmith
- openai
- chromadb
- sentence-transformers
- and more

Install dependencies with:
```bash
pip install -r requirements.txt
```

For local evaluations, [Ollama](https://ollama.ai/) must be installed and running.

## 🔧 Installing Local Models with Ollama

For local evaluations, the toolkit uses [Ollama](https://ollama.ai/) to run models on your machine. This section provides guidance on setting up Gemma 3 and other models locally.

### Installing Ollama

1. Download and install Ollama for your platform from [ollama.ai](https://ollama.ai/)
2. Verify installation by running in your terminal:
   ```bash
   ollama --version
   ```

### Installing Gemma 3 Models

Gemma 3 models are Google's latest open language models that offer excellent performance for local use:

1. Pull the Gemma 3 model you want to use:
   ```bash
   # For the 8B model
   ollama pull gemma3:8b
   
   # For the instruction-tuned 8B model (recommended for RAG)
   ollama pull gemma3:8b-instruct
   
   # For the smaller 2B model
   ollama pull gemma3:2b
   
   # For the instruction-tuned 2B model
   ollama pull gemma3:2b-instruct
   ```

2. Verify the model is installed:
   ```bash
   ollama list
   ```

3. Update your configuration file to use Gemma 3:
   ```yaml
   llm:
     model_name: "gemma3:8b-instruct"  # Choose the appropriate model
     provider: "ollama"
     temperature: 0.1
   ```

### Other Recommended Ollama Models

Other high-performing models for local RAG evaluation:

- **Llama 3**: Meta's latest open models
  ```bash
  ollama pull llama3:8b
  ollama pull llama3:8b-instruct
  ```

- **Mistral**: Excellent performance-to-size ratio
  ```bash
  ollama pull mistral:7b
  ollama pull mistral:7b-instruct
  ```

- **Neural Chat**: Optimized for conversational use
  ```bash
  ollama pull neural-chat:7b
  ```

### Ollama Custom Parameters

You can customize model parameters by specifying them in your config:

```yaml
llm:
  model_name: "gemma3:8b-instruct"
  provider: "ollama"
  temperature: 0.1
  base_url: "http://localhost:11434"  # Default Ollama API endpoint
```

### Troubleshooting Local Models

- **Memory Issues**: For large models like Gemma 3 8B, ensure your system has at least 16GB RAM
- **Slow First Run**: The first run will be slower as the model loads into memory
- **CUDA Support**: For GPU acceleration, ensure you have CUDA installed if using NVIDIA GPUs
- **Server Connection**: Verify Ollama is running with `ollama serve` if you encounter connection issues

## 🔐 Environment Setup

Create a `.env` file in the project root with your API keys:

```
OPENAI_API_KEY=your_openai_api_key
LANGCHAIN_API_KEY=your_langchain_api_key
LANGSMITH_API_KEY=your_langsmith_api_key
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🔗 About ScaleXI Innovation

This framework is developed and maintained by [ScaleXI Innovation](https://scalexi.ai/), specialists in Generative AI and Large Language Model solutions. 

ScaleXI Innovation specializes in:
- Generative AI for Business Automation
- AI-Driven Digital Transformation
- Generative AI Consultation
- Enterprise-Grade LLM Solutions

For more information about our services, visit our website at [https://scalexi.ai](https://scalexi.ai) or contact us at info@scalexi.ai.