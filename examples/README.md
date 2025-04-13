# 🎯 RAG Evaluation Examples

This directory contains comprehensive examples for running retrieval-augmented generation (RAG) evaluations. Each example is carefully designed to address different evaluation scenarios, from simple testing to complex multi-language comparisons and detailed analytics.

## 📚 Available Examples

### 1. 🌟 Simple Evaluation (`01_simple_evaluation.py`)
A beginner-friendly, self-contained starting point for RAG evaluation without complex dependencies.

**Purpose:**
- Provides a complete end-to-end RAG evaluation pipeline in a single file
- Creates a simple but effective vector store using FAISS for document retrieval
- Evaluates retrieval quality, answer relevance, and overall response correctness
- Perfect for initial system testing, debugging, and understanding evaluation basics
- Ideal for environments with limited external API access

**How to Run:**
```bash
export OPENAI_API_KEY=your_openai_api_key_here
python 01_simple_evaluation.py
```
The script uses default configurations, creates a test dataset (if needed), and generates a comprehensive HTML report in the `results/simple_evaluation` directory with detailed metrics and examples.

### 2. 📊 LangSmith Evaluation (`02_langsmith_evaluation.py`)
A sophisticated evaluation framework leveraging LangSmith's powerful tracing and analytics capabilities.

**Purpose:**
- Provides enterprise-grade evaluation with detailed performance metrics
- Traces every step of the RAG pipeline for pinpoint debugging and optimization
- Implements advanced evaluation metrics like faithfulness, relevance, and context precision/recall
- Enables A/B testing between different RAG configurations
- Supports collaborative analysis with shareable results and annotations
- Tracks improvements over time with version control for experiments

**How to Run:**
```bash
export OPENAI_API_KEY=your_openai_api_key_here
export LANGCHAIN_API_KEY=your_langsmith_api_key_here
export LANGCHAIN_PROJECT=my_rag_evaluation  # Optional: custom project name
python 02_langsmith_evaluation.py
```
The script automatically creates a sample dataset and runs the evaluation with default settings. Results will be available in your LangSmith dashboard at smith.langchain.com and also saved locally in the `results/langsmith_evaluation` directory.

### 3. ☁️ Cloud LLM Evaluation (`03_run_llm_cloud_evaluation.py`)
Evaluate your RAG system using OpenAI's powerful cloud-based language models like GPT-4.

**Purpose:**
- Benchmark your RAG system against state-of-the-art language models
- Evaluate with strong reasoning capabilities for complex question types
- Access sophisticated language understanding for nuanced evaluation
- Test with models that have extensive world knowledge
- Compare performance across different OpenAI model versions
- Ideal for highest-quality evaluation when cost is not a primary concern

**How to Run:**
```bash
export OPENAI_API_KEY=your_openai_api_key_here
python 03_run_llm_cloud_evaluation.py
```
The script uses a default configuration file (`../config/llm_cloud_evaluation.yaml`) that specifies dataset path, model parameters, and output settings. No additional arguments needed. Results are saved in the directory specified in the config file, typically `results/llm_cloud_evaluation`.

### 4. 💻 Local Model Evaluation (`04_run_ollama_local_evaluation.py`)
Evaluate RAG systems using local Ollama models for privacy, cost efficiency, and offline capability.

**Purpose:**
- Run complete evaluations without sending data to external APIs
- Ensure privacy for sensitive or proprietary information
- Eliminate usage costs associated with cloud-based APIs
- Test RAG against various open-source models like Llama, Mistral, or Falcon
- Compare performance across different locally hosted models
- Operate in air-gapped environments without internet connectivity
- Customize evaluation parameters for specific use cases

**How to Run:**
```bash
# First, ensure Ollama is installed and running
# Then, pull your desired model
ollama pull llama2  # or another model of your choice
python 04_run_ollama_local_evaluation.py
```
The script loads configuration from `../config/ollama_local_evaluation.yaml` by default, which contains all necessary parameters for running with Ollama models. Results are saved in the output directory specified in the config, typically `results/ollama_evaluation`.

### 5. 🌍 Language Comparison (`05_langsmith_arabic_english_comparator.py`)
Advanced tool for comparing RAG performance across Arabic and English languages with specialized metrics.

**Purpose:**
- Evaluate multilingual RAG capabilities across Arabic and English simultaneously
- Identify language-specific strengths and weaknesses in retrieval and generation
- Compare performance metrics like relevance, factuality, and completeness between languages
- Analyze token efficiency differences between languages
- Detect potential biases in language handling
- Optimize for consistent performance across languages
- Generate comparative visualizations highlighting cross-language differences

**How to Run:**
```bash
export LANGCHAIN_API_KEY=your_langsmith_api_key_here
python 05_langsmith_arabic_english_comparator.py
```
The script uses default project and experiment IDs for comparison (which can be modified in the script if needed). No command-line arguments required. Comparison results are displayed in the console and can be redirected to a file if needed.

### 6. 📈 LangSmith Project Stats (`06_langsmith_project_stats.py`)
Comprehensive statistics generator for LangSmith projects with aggregated performance metrics.

**Purpose:**
- Generate high-level statistics across all experiments in a project
- Track system-wide performance over time
- Calculate aggregated metrics like average latency, token usage, and error rates
- Monitor cost and resource utilization across experiments
- Identify outliers and performance anomalies
- Support management reporting with executive summaries
- Export statistics for integration with other analytics tools

**How to Run:**
```bash
export LANGCHAIN_API_KEY=your_langsmith_api_key_here
python 06_langsmith_project_stats.py
```
The script uses a default project ID configured within the script. Results are displayed directly in the console and can be saved to the `results/project_stats` directory if needed.

### 7. 📊 Experiment Statistics (`07_langsmith_experiment_stats.py`)
In-depth analytical tool for extracting and visualizing detailed experiment-level statistics.

**Purpose:**
- Deep dive into individual experiment results with fine-grained analysis
- Calculate comprehensive metrics across evaluation dimensions
- Generate visualization-ready data for complex performance graphs
- Compare performance across different question types or categories
- Extract patterns and correlations between input characteristics and performance
- Identify specific areas for RAG pipeline improvement
- Support data-driven optimization decisions with statistical evidence

**How to Run:**
```bash
export LANGCHAIN_API_KEY=your_langsmith_api_key_here
python 07_langsmith_experiment_stats.py
```
The script uses default project and experiment IDs defined in the script itself. Statistics are displayed in a formatted console output and can be saved to the `results/experiment_stats` directory for further analysis.

### 8. 🔄 Simple LangSmith Comparison (`08_simple_langsmith_comparison.py`)
Streamlined tool for head-to-head comparison between two LangSmith experiments.

**Purpose:**
- Directly compare two different RAG configurations or versions
- Highlight performance differences across all evaluation metrics
- Calculate statistical significance of improvements
- Generate side-by-side comparisons for key performance indicators
- Visualize performance deltas with intuitive charts
- Identify wins, losses, and ties across evaluation dimensions
- Support evidence-based decisions on which configuration to adopt

**How to Run:**
```bash
export LANGCHAIN_API_KEY=your_langsmith_api_key_here
python 08_simple_langsmith_comparison.py
```
The script uses default project and experiment IDs configured within the script. It produces a clean comparison table showing metrics from both experiments. Results can be saved to the `results/comparisons` directory if specified.

## 🔑 Prerequisites

### 🌟 For Simple Evaluation
```bash
pip install -r requirements.txt  # Installs OpenAI, FAISS, and basic evaluation dependencies
export OPENAI_API_KEY=your_openai_api_key_here
```

### 📊 For LangSmith-based Evaluations
```bash
pip install -r requirements.txt langsmith  # Ensures LangSmith integration
export OPENAI_API_KEY=your_openai_api_key_here
export LANGCHAIN_API_KEY=your_langsmith_api_key_here  # Get this from smith.langchain.com
export LANGCHAIN_PROJECT=your_project_name  # Optional: custom project identifier
```

### 💻 For Local Evaluation
1. Install Ollama from https://ollama.ai (available for Mac, Linux, and Windows)
2. Pull desired models:
```bash
ollama pull llama2  # Base model
ollama pull mistral  # Alternative model
# Check available models at ollama.ai/library
```
3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start Guide

1. **Choose your evaluation type:**
   - 🌟 **New to RAG?** Start with `01_simple_evaluation.py` for a self-contained example
   - 📊 **Need detailed metrics?** Use LangSmith examples for comprehensive analysis
   - 🌍 **Working with multiple languages?** Use the language comparator for cross-lingual evaluation
   - 💻 **Need privacy?** Use local evaluation with Ollama to keep data on-premises
   - 🔄 **Comparing configurations?** Use the comparison tools to see performance differences

2. **Set up your environment:**
   ```bash
   pip install -r requirements.txt
   export OPENAI_API_KEY=your_key  # If using OpenAI models
   export LANGCHAIN_API_KEY=your_key  # If using LangSmith for tracking
   ```

3. **Run your chosen example:**
   ```bash
   cd examples
   python <example_filename>.py
   ```
   
   No command-line arguments are needed! All examples use sensible defaults configured within each script.

4. **Modify configurations (if needed):**
   - For configuration-based examples (03, 04): Edit the YAML files in the `config/` directory
   - For script-based examples: Modify the default project/experiment IDs in the script itself

5. **Analyze the results:**
   - Check generated reports in the `results/` directory
   - View traces and metrics in LangSmith dashboard
   - Compare configurations to identify improvements

## 📋 Output and Results

All evaluation results are saved in the `results/` directory at the root of the project, organized by evaluation type:

- 📑 **Simple evaluation:** `results/simple_evaluation/`
  - HTML reports with per-question analysis and overall metrics
  - JSON files with raw evaluation data
  - Timestamped files for comparing multiple runs

- 📊 **LangSmith evaluations:** `results/langsmith_evaluation/`
  - Local copies of evaluation metadata and results
  - Links to access detailed results in LangSmith platform
  - Experiment IDs and timestamps for future reference

- 💻 **Local evaluations:** `results/ollama_evaluation/`
  - HTML reports similar to simple evaluation
  - Per-model performance metrics
  - Raw evaluation data for custom analysis

- ☁️ **Cloud evaluations:** `results/llm_cloud_evaluation/`
  - Comprehensive reports on cloud LLM performance
  - Detailed metrics categorized by question type
  - Error analysis and improvement suggestions

- 📈 **Comparison results:** `results/comparisons/`
  - Side-by-side comparison reports
  - Visualizations of performance differences
  - Statistical significance analysis

The `results/` directory structure makes it easy to locate evaluation outputs and compare different runs over time. All reports include timestamps to track performance changes as your RAG system evolves.

## 🔍 Need Help?

- 📚 **Documentation:** Check each file's docstrings for detailed documentation
- 🐛 **Debugging:** Look for logging output during execution (set `LOG_LEVEL=DEBUG` for more detail)
- 🔧 **Configuration:** Review the default configurations in the scripts or YAML files
- 🤝 **Community:** Join our community forum for questions and discussions
- 🔄 **Updates:** Check the GitHub repository regularly for new examples and improvements

For detailed documentation on each example, refer to the comments and docstrings within each file. 