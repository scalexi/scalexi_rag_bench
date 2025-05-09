#!/usr/bin/env python
"""Script to create vector stores from source data."""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path to import module
sys.path.append(str(Path(__file__).parent.parent))

from scalexi_rag_bench.vectorstores import create_vectorstore
from scalexi_rag_bench.models import get_embedding_model
from scalexi_rag_bench.config.config import Config


def main(config_path: str):
    """Create vector store based on configuration.
    
    Args:
        config_path: Path to configuration file
    """
    # Load configuration
    print(f"Loading configuration from {config_path}")
    try:
        config = Config.from_file(config_path)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Validate configuration
    if not hasattr(config, 'vectorstore'):
        print("Error: Configuration must include vectorstore section")
        sys.exit(1)
    
    if not hasattr(config.vectorstore, 'source_path') or not config.vectorstore.source_path:
        print("Error: Configuration must include vectorstore.source_path")
        print("Please specify a data source path in the config file, e.g.:")
        print("""
vectorstore:
  source_path: data/english/data_source.txt
  output_dir: ./vectorstores/english
        """)
        sys.exit(1)
    
    # Check if source path exists
    source_path = Path(config.vectorstore.source_path)
    if not source_path.exists():
        print(f"Error: Source data path does not exist: {source_path}")
        print("Please provide a valid path to a text file, PDF, or JSON dataset")
        sys.exit(1)
    
    # Get embedding model
    print(f"Initializing embedding model: {config.retrieval.embedding_model}")
    embedding_model = get_embedding_model(config.retrieval.embedding_model)
    
    # Create output directory for vector stores
    default_output_dir = os.path.join("vectorstores", config.dataset.name)
    output_dir = getattr(config.vectorstore, 'output_dir', default_output_dir)
    
    # Create vector store
    print(f"Creating vector store from: {config.vectorstore.source_path}")
    vectorstore_path = create_vectorstore(
        source_path=config.vectorstore.source_path,
        embedding_model=embedding_model,
        output_dir=output_dir,
        chunk_size=config.retrieval.chunk_size,
        chunk_overlap=config.retrieval.chunk_overlap,
        dataset_name=config.dataset.name
    )
    
    if vectorstore_path:
        # Create updated configuration with the vector store path
        config.vectorstore.path = vectorstore_path
        
        # Save updated configuration
        updated_config_path = config_path.replace(".yaml", "_updated.yaml")
        config.to_file(updated_config_path)
        print(f"\nUpdated configuration saved to: {updated_config_path}")
        
        # Also update the original config file
        config.to_file(config_path)
        print(f"Original configuration updated at: {config_path}")
        
        # Suggest configuration updates
        print("\nVector store created successfully!")
        print(f"Path: {vectorstore_path}")
        print("\nConfiguration has been updated with the following:")
        print(f"""
vectorstore:
  path: {vectorstore_path}
  source_path: {config.vectorstore.source_path}
  output_dir: {output_dir}
""")
    else:
        print("Failed to create vector store")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create vector store from source data")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )
    
    args = parser.parse_args()
    main(args.config) 