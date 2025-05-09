#!/usr/bin/env python
"""Example script to run evaluation with Arabic models with local LLM provider, namely Ollama."""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path to import from scalexi_rag_bench
sys.path.append(str(Path(__file__).parent.parent))

from scalexi_rag_bench import RAGEvaluator
from scalexi_rag_bench.config.config import Config


def main(config_path: str):
    """Run evaluation with specified configuration.
    
    Args:
        config_path: Path to configuration file
    """
    print(f"Loading configuration from {config_path}")
    config = Config.from_file(config_path)
    
    print(f"Initializing evaluator for {config.experiment_name}")
    evaluator = RAGEvaluator(config)
    
    print(f"Running evaluation on {config.dataset.path}")
    results = evaluator.evaluate()
    
    # Generate report
    report_path = os.path.join(config.output_dir, "report.html")
    print(f"Generating report at {report_path}")
    evaluator.generate_report(results, report_path)
    
    print(f"Evaluation complete. Results saved to {config.output_dir}")


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Run RAG evaluation for Arabic")
        parser.add_argument(
        "--config",
        type=str,
        default="config/ollama_local_rag_with_vectorstore.yaml",
        help="Path to configuration YAML file"
        )
    
        args = parser.parse_args()
        main(args.config) 
    except Exception as e:
        print(f"[Error]run from the root directory as: python examples/04_run_ollama_local_evaluation.py --config config/ollama_local_evaluation.yaml")
        print(f"[Error]Error: {e}")
        sys.exit(1)
