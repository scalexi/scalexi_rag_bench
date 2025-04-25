#!/usr/bin/env python
"""Example script demonstrating vector store setup and RAG evaluation with proper ground truth retrieval."""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path to import from scalexi_rag_bench
sys.path.append(str(Path(__file__).parent.parent))

from scalexi_rag_bench import RAGEvaluator
from scalexi_rag_bench.config.config import Config
from scalexi_rag_bench.vectorstores import create_vectorstore
from scalexi_rag_bench.models import get_embedding_model


def setup_vectorstore(config_path: str) -> str:
    """Set up a vector store from the dataset.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        str: Path to the created vector store
    """
    # Load configuration
    print(f"Loading configuration from {config_path}")
    try:
        config = Config.from_file(config_path)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return None
    
    # Check if vectorstore configuration exists
    if not hasattr(config, 'vectorstore'):
        print("Error: Configuration must include vectorstore section")
        return None
    
    if not hasattr(config.vectorstore, 'source_path') or not config.vectorstore.source_path:
        print("Error: Missing vectorstore.source_path in configuration")
        print("Please specify a data source path in the config file, e.g.:")
        print("""
vectorstore:
  source_path: data/english/data_source.txt
  output_dir: ./vectorstores/english
        """)
        return None
    
    # Check if source path exists
    source_path = Path(config.vectorstore.source_path)
    if not source_path.exists():
        print(f"Error: Source data path does not exist: {source_path}")
        print("Please provide a valid path to a text file, PDF, or JSON dataset")
        return None
    
    # Get embedding model
    print(f"Initializing embedding model: {config.retrieval.embedding_model}")
    embedding_model = get_embedding_model(config.retrieval.embedding_model)
    
    # Set default output directory if not specified
    output_dir = getattr(config.vectorstore, 'output_dir', f"./vectorstores/{config.dataset.name}")
    
    # Create vector store
    print(f"Creating vector store from: {config.vectorstore.source_path}")
    vectorstore_path = create_vectorstore(
        source_path=config.vectorstore.source_path,
        embedding_model=embedding_model._embeddings,
        output_dir=output_dir,
        chunk_size=config.retrieval.chunk_size,
        chunk_overlap=config.retrieval.chunk_overlap,
        dataset_name=config.dataset.name
    )
    
    if vectorstore_path:
        print(f"\nVector store created successfully at: {vectorstore_path}")
        
        # Update the configuration with the new vector store path
        config.vectorstore.path = vectorstore_path
        
        # Save the updated configuration
        updated_config_path = config_path.replace(".yaml", "_updated.yaml")
        config.to_file(updated_config_path)
        print(f"Updated configuration saved to: {updated_config_path}")
        
        # Also update the original config file
        config.to_file(config_path)
        print(f"Original configuration updated at: {config_path}")
        
        return vectorstore_path
    else:
        print("Failed to create vector store")
        return None


def run_evaluation(config_path: str) -> None:
    """Run evaluation with the specified configuration.
    
    Args:
        config_path: Path to configuration file
    """
    # Load configuration
    print(f"Loading configuration from {config_path}")
    try:
        config = Config.from_file(config_path)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return
    
    # Check if vector store path is configured
    if not hasattr(config, 'vectorstore') or not config.vectorstore:
        print("Error: No vector store configuration found.")
        print("Please run the setup_vectorstore function first to create a vector store.")
        return
    
    if not config.vectorstore.path:
        print("Error: No vector store path configured in the configuration file.")
        print("Please run the setup_vectorstore function first to create a vector store.")
        return
        
    print(f"Using vector store at: {config.vectorstore.path}")
        
    # Verify that the vector store exists
    if not os.path.exists(config.vectorstore.path):
        print(f"Error: Vector store path {config.vectorstore.path} does not exist.")
        print("Please run the setup_vectorstore function first.")
        return
    
    # Initialize evaluator
    print(f"Initializing evaluator for {config.experiment_name}")
    evaluator = RAGEvaluator(config)
    
    # Run evaluation
    print(f"Running evaluation on {config.dataset.path}")
    results = evaluator.evaluate()
    
    # Generate report
    report_path = os.path.join(config.output_dir, "report.html")
    print(f"Generating report at {report_path}")
    evaluator.generate_report(results, report_path)
    
    print(f"Evaluation complete. Results saved to {config.output_dir}")
    print(f"Report available at {report_path}")
    print(f"LangSmith URL: {results.get('langsmith_url', 'Not available')}")


def main(config_path: str, skip_setup: bool = False) -> None:
    """Run the full process of setting up vector store and running evaluation.
    
    Args:
        config_path: Path to configuration file
        skip_setup: Whether to skip vector store setup
    """
    if not skip_setup:
        vectorstore_path = setup_vectorstore(config_path)
        if not vectorstore_path:
            print("Vector store setup failed. Please fix the configuration and try again.")
            return
        
        # Use the updated config for evaluation
        updated_config_path = config_path.replace(".yaml", "_updated.yaml")
        if os.path.exists(updated_config_path):
            config_path = updated_config_path
    
    run_evaluation(config_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG evaluation with vector store setup")
    parser.add_argument(
        "--config",
        type=str,
        default="../config/llm_cloud_rag.yaml",
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Skip vector store setup and use existing configuration"
    )
    
    args = parser.parse_args()
    main(args.config, args.skip_setup) 