# 🎯 RAG Evaluation Examples

This directory contains comprehensive examples for running retrieval-augmented generation (RAG) evaluations. Each example is carefully designed to address different evaluation scenarios, from simple testing to complex multi-language comparisons and detailed analytics.

## 📚 Available Examples

### 1. 🌟 Simple Evaluation (`01_simple_evaluation.py`)
A beginner-friendly, self-contained starting point for RAG evaluation without complex dependencies.

**Purpose:**
- **This is an educational script designed to demonstrate RAG concepts, not a production-ready evaluation pipeline**
- Provides a complete end-to-end RAG evaluation pipeline in a single file
- Creates a simple but effective vector store using FAISS for document retrieval
- Demonstrates the core RAG workflow: document retrieval → context integration → answer generation
- Perfect for initial system testing, debugging, and understanding evaluation basics
- Ideal for environments with limited external API access

**How to Run:**
```bash
export OPENAI_API_KEY=your_openai_api_key_here
python 01_simple_evaluation.py
```
The script uses default configurations, creates a test dataset (if needed), and generates an HTML report in the `results/minimal_evaluation` directory. The report displays each question, the retrieved documents, model answer, and ground truth in a clear, side-by-side format for manual comparison and analysis.

### 2. 📊 LangSmith Evaluation (`02_langsmith_evaluation.py`)
A sophisticated evaluation framework leveraging LangSmith's powerful tracing and analytics capabilities.

**Purpose:**
- **This is an educational script that creates its own in-memory vector store for demonstration purposes, not a production-ready evaluation pipeline**
- Creates a small FAISS vector store on the fly from a test dataset (no pre-existing vector store required)
- Implements a complete RAG pipeline where:
  - Documents are embedded using OpenAI embeddings
  - Retrieval uses FAISS similarity search to find relevant documents
  - Generation combines retrieved context with the question for the LLM
- Provides enterprise-grade evaluation with detailed performance metrics through LangSmith
- Implements advanced evaluation metrics like correctness, relevance, and groundedness
- Demonstrates how to capture and analyze each step of the RAG pipeline using LangSmith tracing

**How to Run:**
```bash
export OPENAI_API_KEY=your_openai_api_key_here
export LANGCHAIN_API_KEY=your_langsmith_api_key_here
export LANGCHAIN_PROJECT=my_rag_evaluation  # Optional: custom project name
python 02_langsmith_evaluation.py
```
The script automatically creates a sample dataset, builds an in-memory vector store, and runs the evaluation with default settings. Results will be available in your LangSmith dashboard at smith.langchain.com and also saved locally in the `results/langsmith_evaluation` directory.

