#!/usr/bin/env python
"""RAG Evaluation Pipeline: Setup vector stores and evaluate RAG systems.

This script provides a comprehensive pipeline for RAG (Retrieval-Augmented Generation) evaluation:
1. Creating vector stores from source data (documents, text files, etc.)
2. Evaluating RAG systems with various retrieval, generation, and system metrics
3. Generating detailed reports with evaluation results

The pipeline supports both cloud-based LLMs (OpenAI, Anthropic, etc.) and local models (Ollama)
and is designed to be used in production environments.
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path to import from scalexi_rag_bench
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

from scalexi_rag_bench import RAGEvaluator
from scalexi_rag_bench.config.config import Config
from scalexi_rag_bench.vectorstores import create_vectorstore
from scalexi_rag_bench.models import get_embedding_model


# Emoji definitions for better visual logging
EMOJI = {
    "rocket": "🚀",
    "gear": "⚙️",
    "check": "✅",
    "error": "❌",
    "warning": "⚠️",
    "info": "ℹ️",
    "file": "📄",
    "database": "🗃️",
    "search": "🔍",
    "chart": "📊",
    "done": "✨",
    "config": "🔧",
    "loading": "⏳"
}

# ANSI color codes
class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def print_header(text, emoji_key="info"):
    """Print a visually distinct header with emoji and color."""
    emoji = EMOJI.get(emoji_key, "")
    border = "=" * (len(text) + 10)
    print(f"\n{Colors.BOLD}{Colors.BLUE}{border}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}    {emoji} {text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{border}{Colors.END}\n")


def print_step(text, emoji_key="gear"):
    """Print a step with emoji and color."""
    emoji = EMOJI.get(emoji_key, "")
    print(f"\n{Colors.CYAN}{emoji} {text}{Colors.END}")


def print_success(text):
    """Print a success message with check emoji and green color."""
    print(f"{Colors.GREEN}{EMOJI['check']} {text}{Colors.END}")


def print_error(text):
    """Print an error message with error emoji and red color."""
    print(f"{Colors.RED}{EMOJI['error']} {text}{Colors.END}")


def print_warning(text):
    """Print a warning message with warning emoji and yellow color."""
    print(f"{Colors.YELLOW}{EMOJI['warning']} {text}{Colors.END}")


def print_info(text):
    """Print an info message with info emoji and purple color."""
    print(f"{Colors.PURPLE}{EMOJI['info']} {text}{Colors.END}")


def print_config_section(title, config_section, indent=0):
    """Print a configuration section with indentation and formatting."""
    indent_str = "  " * indent
    print(f"{indent_str}{Colors.CYAN}{title}:{Colors.END}")
    
    for key, value in vars(config_section).items():
        if key.startswith('_'):
            continue
            
        if hasattr(value, '__dict__') and not isinstance(value, (str, int, float, bool, list, dict)):
            print_config_section(key, value, indent + 1)
        elif isinstance(value, list):
            print(f"{indent_str}  {Colors.YELLOW}{key}:{Colors.END} {', '.join(str(item) for item in value)}")
        else:
            print(f"{indent_str}  {Colors.YELLOW}{key}:{Colors.END} {value}")


def display_config(config, config_path):
    """Display configuration details in a structured, readable format."""
    print_header("Configuration Details", "config")
    print_step(f"Config file: {Colors.BOLD}{config_path}{Colors.END}", "file")
    
    # Display key configuration sections
    for section_name in dir(config):
        if not section_name.startswith('_') and section_name not in ('to_dict', 'to_file', 'from_file'):
            section = getattr(config, section_name)
            if hasattr(section, '__dict__'):
                print_config_section(section_name, section)
    
    print()  # Add a blank line after config display


def setup_vectorstore(config_path: str, force_rebuild: bool = False) -> str:
    """Set up a vector store from the dataset.
    
    Args:
        config_path: Path to configuration file
        force_rebuild: Whether to force rebuilding the vector store even if it exists
        
    Returns:
        str: Path to the created vector store
    """
    # Load configuration
    print_step(f"Loading configuration from {Colors.BOLD}{config_path}{Colors.END}", "config")
    try:
        config = Config.from_file(config_path)
        print_info(f"Configuration loaded successfully")
        
        # Display configuration details
        display_config(config, config_path)
        
    except Exception as e:
        print_error(f"Error loading configuration: {e}")
        return None
    
    # Check if vectorstore configuration exists
    if not hasattr(config, 'vectorstore'):
        print_error("Configuration must include vectorstore section")
        return None
    
    # Check if vector store already exists and can be reused
    if hasattr(config.vectorstore, 'path') and config.vectorstore.path:
        existing_path = Path(config.vectorstore.path)
        if not existing_path.is_absolute():
            existing_path = project_root / existing_path
        
        if existing_path.exists() and not force_rebuild:
            print_step(f"Existing vector store found at: {Colors.BOLD}{existing_path}{Colors.END}", "database")
            print_info("Using existing vector store. Use --force-rebuild to recreate it if needed.")
            return str(existing_path)
        elif existing_path.exists() and force_rebuild:
            print_step(f"Existing vector store found at: {Colors.BOLD}{existing_path}{Colors.END}", "database")
            print_warning("Force rebuild enabled - recreating vector store")
    
    if not hasattr(config.vectorstore, 'source_path') or not config.vectorstore.source_path:
        print_error("Missing vectorstore.source_path in configuration")
        print_warning("Please specify a data source path in the config file, e.g.:")
        print(f"{Colors.YELLOW}vectorstore:\n  source_path: data/english/data_source.txt\n  output_dir: ./vectorstores/english{Colors.END}")
        return None
    
    # Fix path resolution: make relative paths absolute relative to project root
    source_path = Path(config.vectorstore.source_path)
    if not source_path.is_absolute():
        source_path = project_root / source_path
        print_info(f"Resolved source path: {Colors.BOLD}{source_path}{Colors.END}")
    
    # Check if source path exists
    if not source_path.exists():
        print_error(f"Source data path does not exist: {Colors.BOLD}{source_path}{Colors.END}")
        print_warning("Please provide a valid path to a text file, PDF, or JSON dataset")
        return None
    
    # Get embedding model
    print_step(f"Initializing embedding model: {Colors.BOLD}{config.retrieval.embedding_model}{Colors.END}", "loading")
    embedding_model = get_embedding_model(config.retrieval.embedding_model)
    print_info("Embedding model initialized successfully")
    
    # Set default output directory if not specified
    output_dir = getattr(config.vectorstore, 'output_dir', f"./vectorstores/{config.dataset.name}")
    
    # Make output dir absolute if it's relative
    if not Path(output_dir).is_absolute():
        output_dir = str(project_root / output_dir)
        print_info(f"Resolved output directory: {Colors.BOLD}{output_dir}{Colors.END}")
    
    # Create vector store
    print_step(f"Creating vector store from: {Colors.BOLD}{source_path}{Colors.END}", "database")
    print_info(f"Chunk size: {Colors.BOLD}{config.retrieval.chunk_size}{Colors.END}, Overlap: {Colors.BOLD}{config.retrieval.chunk_overlap}{Colors.END}")
    
    vectorstore_path = create_vectorstore(
        source_path=str(source_path),
        embedding_model=embedding_model._embeddings,
        output_dir=output_dir,
        chunk_size=config.retrieval.chunk_size,
        chunk_overlap=config.retrieval.chunk_overlap,
        dataset_name=config.dataset.name
    )
    
    if vectorstore_path:
        print_success(f"Vector store created successfully at: {Colors.BOLD}{vectorstore_path}{Colors.END}")
        
        # Update the configuration with the new vector store path
        config.vectorstore.path = vectorstore_path
        
        # Save the updated configuration
        updated_config_path = config_path.replace(".yaml", "_with_vectorstore.yaml")
        config.to_file(updated_config_path)
        print_step(f"Updated configuration saved to: {Colors.BOLD}{updated_config_path}{Colors.END}", "config")
        
        # Also update the original config file
        config.to_file(config_path)
        print_step(f"Original configuration updated at: {Colors.BOLD}{config_path}{Colors.END}", "config")
        
        return vectorstore_path
    else:
        print_error("Failed to create vector store")
        return None


def run_evaluation(config_path: str) -> None:
    """Run evaluation with the specified configuration.
    
    Args:
        config_path: Path to configuration file
    """
    # Load configuration
    print_step(f"Loading configuration from {Colors.BOLD}{config_path}{Colors.END}", "config")
    try:
        config = Config.from_file(config_path)
        print_info(f"Configuration loaded successfully")
        
        # Display configuration details
        display_config(config, config_path)
        
    except Exception as e:
        print_error(f"Error loading configuration: {e}")
        return
    
    # Check if vector store path is configured
    if not hasattr(config, 'vectorstore') or not config.vectorstore:
        print_error("No vector store configuration found.")
        print_warning("Please run the setup_vectorstore function first to create a vector store.")
        return
    
    if not config.vectorstore.path:
        print_error("No vector store path configured in the configuration file.")
        print_warning("Please run the setup_vectorstore function first to create a vector store.")
        return
    
    # Fix path resolution for vector store path
    vectorstore_path = Path(config.vectorstore.path)
    if not vectorstore_path.is_absolute():
        vectorstore_path = project_root / vectorstore_path
        config.vectorstore.path = str(vectorstore_path)
        print_info(f"Resolved vector store path: {Colors.BOLD}{config.vectorstore.path}{Colors.END}")
        
    print_step(f"Using vector store at: {Colors.BOLD}{config.vectorstore.path}{Colors.END}", "database")
        
    # Verify that the vector store exists
    if not os.path.exists(config.vectorstore.path):
        print_error(f"Vector store path {Colors.BOLD}{config.vectorstore.path}{Colors.END} does not exist.")
        print_warning("Please run the setup_vectorstore function first.")
        return
    
    # Fix dataset path if it's relative
    if hasattr(config, 'dataset') and hasattr(config.dataset, 'path'):
        dataset_path = Path(config.dataset.path)
        if not dataset_path.is_absolute():
            dataset_path = project_root / dataset_path
            config.dataset.path = str(dataset_path)
            print_step(f"Resolved dataset path: {Colors.BOLD}{config.dataset.path}{Colors.END}", "file")
    
    # Initialize evaluator
    print_step(f"Initializing evaluator for {Colors.BOLD}{config.experiment_name}{Colors.END}", "gear")
    
    # Show metrics being used
    if hasattr(config, 'evaluation_metrics'):
        metrics = []
        if hasattr(config.evaluation_metrics, 'retrieval_metrics'):
            metrics.extend([f"{Colors.CYAN}Retrieval:{Colors.END} " + ", ".join(config.evaluation_metrics.retrieval_metrics)])
        if hasattr(config.evaluation_metrics, 'generation_metrics'):
            metrics.extend([f"{Colors.CYAN}Generation:{Colors.END} " + ", ".join(config.evaluation_metrics.generation_metrics)])
        if hasattr(config.evaluation_metrics, 'system_metrics'):
            metrics.extend([f"{Colors.CYAN}System:{Colors.END} " + ", ".join(config.evaluation_metrics.system_metrics)])
        
        if metrics:
            print_info("Evaluation metrics:")
            for metric in metrics:
                print(f"  {Colors.PURPLE}•{Colors.END} {metric}")
    
    evaluator = RAGEvaluator(config)
    print_info("Evaluator initialized successfully")
    
    # Run evaluation
    print_step(f"Running evaluation on {Colors.BOLD}{config.dataset.path}{Colors.END}", "search")
    print_info(f"This may take some time, please be patient...")
    
    results = evaluator.evaluate()
    
    # Fix path for output directory
    output_dir = Path(config.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
        
    # Generate report
    report_path = os.path.join(str(output_dir), "report.html")
    print_step(f"Generating report at {Colors.BOLD}{report_path}{Colors.END}", "chart")
    evaluator.generate_report(results, report_path)
    
    print_header("Evaluation Complete", "done")
    print_success(f"Results saved to {Colors.BOLD}{output_dir}{Colors.END}")
    print_success(f"Report available at {Colors.BOLD}{report_path}{Colors.END}")
    
    if results.get('langsmith_url'):
        print_success(f"LangSmith URL: {Colors.BOLD}{results.get('langsmith_url')}{Colors.END}")


def main(config_path: str, skip_setup: bool = False, force_rebuild: bool = False, skip_evaluation: bool = False) -> None:
    """Run the full RAG pipeline: setup vector store and run evaluation.
    
    Args:
        config_path: Path to configuration file
        skip_setup: Whether to skip vector store setup
        force_rebuild: Whether to force rebuilding the vector store even if it exists
        skip_evaluation: Whether to skip the evaluation step (only setup vector store)
    """
    # Resolve config path
    config_path_obj = Path(config_path)
    if not config_path_obj.is_absolute():
        # If config path is relative to current directory, make it absolute
        if config_path_obj.exists():
            config_path = str(config_path_obj.absolute())
        # Otherwise try relative to project root
        else:
            project_config_path = project_root / config_path_obj
            if project_config_path.exists():
                config_path = str(project_config_path)
    
    if not Path(config_path).exists():
        print_error(f"Configuration file not found: {Colors.BOLD}{config_path}{Colors.END}")
        return
    
    if not skip_setup:
        print_header("Stage 1: Vector Store Setup", "database")
        vectorstore_path = setup_vectorstore(config_path, force_rebuild)
        if not vectorstore_path:
            print_error("Vector store setup failed. Please fix the configuration and try again.")
            return
        
        # Use the updated config for evaluation
        updated_config_path = config_path.replace(".yaml", "_with_vectorstore.yaml")
        if os.path.exists(updated_config_path):
            config_path = updated_config_path
    
    if not skip_evaluation:
        print_header("Stage 2: RAG Evaluation", "search")
        run_evaluation(config_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Pipeline: Setup vector stores and evaluate RAG systems")
    parser.add_argument(
        "--config",
        type=str,
        default=str(project_root / "config/llm_cloud_rag.yaml"),
        help="Path to configuration YAML file (default: config/llm_cloud_rag.yaml)"
    )
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Skip vector store setup and use existing configuration"
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuilding the vector store even if it already exists"
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip the evaluation step (only setup vector store)"
    )
    
    args = parser.parse_args()
    
    # Check if terminal supports colors
    color_supported = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
    if not color_supported:
        # Disable colors
        for attr in dir(Colors):
            if not attr.startswith('__'):
                setattr(Colors, attr, '')
    
    # Display startup message
    print_header("RAG Evaluation Pipeline Starting", "rocket")
    print_step(f"Config: {Colors.BOLD}{args.config}{Colors.END}", "config")
    print_step(f"Skip setup: {Colors.BOLD}{args.skip_setup}{Colors.END}", "info")
    if args.force_rebuild:
        print_step(f"Force rebuild: {Colors.BOLD}Yes{Colors.END}", "warning")
    if args.skip_evaluation:
        print_step(f"Skip evaluation: {Colors.BOLD}Yes{Colors.END}", "info")
    
    main(args.config, args.skip_setup, args.force_rebuild, args.skip_evaluation) 