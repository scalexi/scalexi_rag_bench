# 🔍 RAG Evaluation Framework

A comprehensive framework for evaluating Retrieval-Augmented Generation (RAG) systems across different models, languages, and datasets.

## 📂 Directory Structure

### Core Components

- 📊 **evaluator.py**: Main evaluation orchestrator containing the `RAGEvaluator` class that coordinates the entire evaluation process using LangSmith and LangGraph.

- 🧩 **evaluators/**: Evaluation metrics and methods
  - 📏 `metrics.py`: Implementation of various evaluation metrics for retrieval quality, generation quality, and system performance
  - 🔄 Support for retrieval, generation, and end-to-end system evaluation

- 🔎 **retrievers/**: Document retrieval mechanisms
  - 🗂️ `retriever.py`: Implementation of various retrieval methods (vector search, hybrid search, etc.)
  - 🧠 Adapters for different vector databases and retrieval techniques

- 🤖 **models/**: Model adapters and wrappers
  - 💬 `llm_adapters.py`: Adapters for different language models (OpenAI, Anthropic, etc.)
  - 📊 `embedding_adapters.py`: Adapters for different embedding models
  - 📁 Specialized adapter directories for extended functionality

- ⚙️ **config/**: Configuration management
  - 🛠️ `config.py`: Configuration classes and validation for setting up evaluations

- 🛠️ **utils/**: Utility functions
  - 🧰 Helper functions, data processing, and common utilities

### Data and Examples

- 📚 **data/**: Evaluation datasets
  - 🇺🇸 `english/`: English language datasets
  - 🇦🇪 `arabic/`: Arabic language datasets
  - 🔄 `processors/`: Data processing utilities

- 📝 **examples/**: Usage examples and demonstrations

- 🧪 **experiments/**: Experiment configurations and results

## 🔄 Main Classes

### 📊 evaluator.py

- 🚀 **RAGEvaluator**: Main evaluator class that:
  - Builds and manages the evaluation workflow as a LangGraph
  - Integrates with LangSmith for tracking experiments and metrics
  - Executes the retrieve→generate chain for evaluation
  - Generates HTML reports with result visualizations
  - Manages experiment persistence and serialization

- 📦 **RAGState**: TypedDict for state management in the evaluation graph:
  - Tracks questions, retrieved contexts, generated answers
  - Maintains metadata like timing, token usage, and evaluation details

### ⚙️ config/config.py

- 📝 **Config**: Main configuration class that defines the entire evaluation setup:
  - Manages experiment settings and metadata
  - Provides methods to load/save configurations from YAML
  - Coordinates component-specific configurations

- 🔎 **RetrievalConfig**: Specialized config for retrieval components:
  - Specifies retriever type, parameters, and performance settings
  - Controls chunking and filtering behavior

- 🤖 **LLMConfig**: Settings for language models:
  - Model provider, name, and API credentials
  - Generation parameters (temperature, tokens, etc.)

- 📚 **DatasetConfig**: Dataset specification and parameters:
  - Dataset path, language, and format
  - Processing instructions

- 📏 **EvaluationMetricsConfig**: Metrics selection:
  - Retrieval, generation, and system metrics to track

### 🔍 retrievers/retriever.py

- 🛠️ **BaseRetrieverAdapter**: Abstract base class for all retrievers:
  - Common interface for document retrieval
  - Query processing and standardized output

- 🔢 **VectorRetrieverAdapter**: Dense retrieval adapter:
  - Vector similarity search from embedding spaces
  - Integrates with FAISS or other vector DBs

- 💎 **ChromaRetrieverAdapter**: Chroma DB integration:
  - ChromaDB-specific retrieval implementation
  - Supports metadata filtering and hybrid search

- 📑 **BM25RetrieverAdapter**: Sparse retrieval adapter:
  - Term frequency-based search implementation
  - Lexical matching capabilities

- 🔀 **HybridRetrieverAdapter**: Combined retrieval approach:
  - Weights both dense and sparse retrievers
  - Provides balanced results between semantic and lexical matching

### 💬 models/llm_adapters.py

- 🧩 **BaseLLMAdapter**: Common interface for all language models:
  - Standardized invocation pattern
  - Consistent configuration and prompt handling

- 🌐 **OpenAIAdapter**: Integration with OpenAI models:
  - ChatGPT, GPT-4, and text-davinci models
  - Handles API authentication and rate limiting

- 🔄 **CohereAdapter**: Integration with Cohere models:
  - Command and other Cohere models
  - Response handling and optimization

- 🏠 **OllamaAdapter**: Local model integration:
  - Support for local LLM deployments
  - Customizable base URLs and parameters

- 🤗 **HuggingFaceAdapter**: HuggingFace model support:
  - Integration with Transformers library
  - Optimized loading and inference

### 📊 models/embedding_adapters.py

- 🧮 **BaseEmbeddingAdapter**: Abstract base for all embedding models:
  - Vector representation of text queries and documents
  - Standard embedding interface

- 🌐 **OpenAIEmbeddingAdapter**: OpenAI embedding models:
  - text-embedding-ada-002 and other models
  - Optimized for OpenAI's retrieval approach

- 🔄 **CohereEmbeddingAdapter**: Cohere embedding models:
  - Multilingual embedding support
  - Dimension optimization

- 🤗 **HuggingFaceEmbeddingAdapter**: HuggingFace embedding models:
  - Sentence transformers and other models
  - Local inference options

- 🚀 **BGEEmbeddingAdapter**: BGE embedding models:
  - State-of-the-art embedding performance
  - Specialized for retrieval tasks

### 📏 evaluators/metrics.py

- 🎯 **Retrieval Evaluators**: Measure retrieval performance:
  - Precision, recall, and MRR metrics
  - Document relevance scoring

- ✅ **Generation Evaluators**: Assess answer quality:
  - Correctness, factuality, and relevance metrics
  - LLM-based evaluation methods

- ⏱️ **System Evaluators**: Monitor system performance:
  - Latency and throughput measurements
  - Token usage tracking

## 🚀 Usage

The framework uses a configuration-based approach where you define:
1. The LLM to use for generation
2. The embedding model and retrieval method
3. The evaluation metrics to apply
4. The dataset to evaluate on

Results are tracked in LangSmith and can be exported as reports.

## 📊 Evaluation Metrics

The framework supports multiple evaluation dimensions:
- Retrieval quality (precision, recall, relevance)
- Generation quality (factuality, helpfulness, coherence)
- System metrics (latency, token usage)

## 🔗 Integration

Built on top of:
- LangChain for components
- LangGraph for workflow orchestration
- LangSmith for evaluation tracking
