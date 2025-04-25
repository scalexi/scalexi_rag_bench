"""Main evaluator for RAG systems."""

import os
import json
import time
from typing import Any, Dict, List, Optional
from langsmith import Client
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict, Annotated

from scalexi_rag_bench.config.config import Config
from scalexi_rag_bench.models import get_llm, get_embedding_model
from scalexi_rag_bench.retrievers import get_retriever
from scalexi_rag_bench.evaluators import (
    get_retrieval_evaluators,
    get_generation_evaluators,
    get_system_evaluators
)


class RAGState(TypedDict):
    """State for RAG evaluation."""
    
    question: str
    context: List[Any]  # Documents retrieved
    answer: str
    metadata: Dict[str, Any]  # Additional metadata for evaluation


class RAGEvaluator:
    """Main evaluator for RAG systems."""
    
    def __init__(self, config: Config):
        """Initialize the evaluator.
        
        Args:
            config: Configuration for the evaluation
        """
        self.config = config
        self.experiment_name = config.experiment_name
        self.langsmith_client = Client()
        
        # Initialize components based on config
        self.llm = get_llm(config.llm)
        self.embedding_model = get_embedding_model(config.retrieval.embedding_model)
        
        # Get vector store path from config if available
        vectorstore_path = None
        if hasattr(config, 'vectorstore') and config.vectorstore and hasattr(config.vectorstore, 'path'):
            vectorstore_path = config.vectorstore.path
            if vectorstore_path and not os.path.exists(vectorstore_path):
                print(f"Warning: Vector store path {vectorstore_path} does not exist. Falling back to empty vector store.")
                vectorstore_path = None
        
        # Initialize retriever with the vector store path
        self.retriever = get_retriever(config.retrieval, self.embedding_model, vectorstore_path)
        
        # Set up evaluation metrics
        self.retrieval_evaluators = get_retrieval_evaluators(config.evaluation_metrics.retrieval_metrics)
        self.generation_evaluators = get_generation_evaluators(config.evaluation_metrics.generation_metrics)
        self.system_evaluators = get_system_evaluators(config.evaluation_metrics.system_metrics)
        
        # Set up output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Set up the RAG graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the RAG graph.
        
        Returns:
            StateGraph: The RAG graph
        """
        # Define retrieval node
        def retrieve(state: RAGState) -> Dict:
            start_time = time.time()
            retrieved_docs = self.retriever.retrieve(state["question"])
            retrieval_time = time.time() - start_time
            
            return {
                "context": retrieved_docs,
                "metadata": {"retrieval_time": retrieval_time}
            }
        
        # Define generation node
        def generate(state: RAGState) -> Dict:
            start_time = time.time()
            
            # Prepare context for generation
            context = "\n\n".join(doc.page_content for doc in state["context"])
            
            # Generate answer
            answer = self.llm.invoke({
                "question": state["question"],
                "context": context
            })
            
            generation_time = time.time() - start_time
            
            return {
                "answer": answer,
                "metadata": {**state["metadata"], "generation_time": generation_time}
            }
        
        # Build graph
        graph_builder = StateGraph(RAGState)
        graph_builder.add_node("retrieve", retrieve)
        graph_builder.add_node("generate", generate)
        
        # Add edges
        graph_builder.add_edge(START, "retrieve")
        graph_builder.add_edge("retrieve", "generate")
        
        return graph_builder.compile()
    
    def evaluate(self, dataset_path: Optional[str] = None) -> Dict[str, Any]:
        """Run the evaluation.
        
        Args:
            dataset_path: Path to the dataset. If None, use the path from config.
            
        Returns:
            Dict: Evaluation results
        """
        # Load dataset
        dataset_path = dataset_path or self.config.dataset.path
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        with open(dataset_path, "r") as f:
            dataset = json.load(f)
        
        # Create LangSmith dataset if it doesn't exist
        langsmith_dataset_name = f"{self.experiment_name}-{self.config.dataset.name}"
        
        try:
            langsmith_dataset = self.langsmith_client.read_dataset(dataset_name=langsmith_dataset_name)
        except:
            langsmith_dataset = self.langsmith_client.create_dataset(dataset_name=langsmith_dataset_name)
            
            # Add examples to dataset
            examples = []
            for item in dataset:
                examples.append({
                    "inputs": {"question": item["question"]},
                    "outputs": {
                        "answer": item.get("answer", ""),
                        "relevant_docs": item.get("relevant_docs", [])
                    }
                })
            
            self.langsmith_client.create_examples(
                dataset_id=langsmith_dataset.id,
                examples=examples
            )
        
        # Define target function for evaluation
        def target(inputs: Dict) -> Dict:
            result = self.graph.invoke({"question": inputs["question"]})
            return {
                "answer": result["answer"],
                "context": result["context"],
                "metadata": result["metadata"]
            }
        
        # Run evaluation
        experiment_results = self.langsmith_client.evaluate(
            target,
            data=langsmith_dataset_name,
            evaluators=self.retrieval_evaluators + self.generation_evaluators + self.system_evaluators,
            experiment_prefix=self.experiment_name,
            metadata={
                "model": self.config.llm.model_name,
                "embedding_model": self.config.retrieval.embedding_model,
                "language": self.config.dataset.language,
                "vectorstore": getattr(self.config.vectorstore, 'path', 'none') if hasattr(self.config, 'vectorstore') else 'none'
            }
        )
        
        # Convert experiment results to a serializable format
        serializable_results = {
            "experiment_id": getattr(experiment_results, "id", None),
            "experiment_name": getattr(experiment_results, "name", None),
            "dataset_id": getattr(experiment_results, "dataset_id", None),
            "langsmith_url": getattr(experiment_results, "url", None),
            "run_ids": list(getattr(experiment_results, "run_ids", [])) if hasattr(experiment_results, "run_ids") else [],
        }
        
        # Try to extract scores and metrics
        if hasattr(experiment_results, "mean_scores"):
            serializable_results["mean_scores"] = experiment_results.mean_scores
        
        if hasattr(experiment_results, "feedback_counts"):
            serializable_results["feedback_counts"] = experiment_results.feedback_counts
        
        # Save results
        results_path = os.path.join(self.config.output_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(serializable_results, f, indent=2)
        
        return serializable_results
    
    def generate_report(self, results: Dict[str, Any], output_path: str) -> None:
        """Generate a report from evaluation results.
        
        Args:
            results: Evaluation results
            output_path: Path to save the report
        """
        # Create report directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Extract metrics from results
        mean_scores = results.get("mean_scores", {})
        
        # Create a simple table with the results
        metrics_table = ""
        for metric, value in mean_scores.items():
            metrics_table += f"<tr><td>{metric}</td><td>{value:.4f}</td></tr>"
        
        # Get vector store info
        vectorstore_info = "None"
        if hasattr(self.config, 'vectorstore') and self.config.vectorstore and hasattr(self.config.vectorstore, 'path'):
            vectorstore_info = self.config.vectorstore.path
        
        # Basic HTML report for now, can be enhanced later
        html_content = f"""
        <html>
            <head>
                <title>RAG Evaluation Report: {self.experiment_name}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #333; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                </style>
            </head>
            <body>
                <h1>RAG Evaluation Report: {self.experiment_name}</h1>
                <h2>Configuration</h2>
                <table>
                    <tr><th>Setting</th><th>Value</th></tr>
                    <tr><td>Model</td><td>{self.config.llm.model_name}</td></tr>
                    <tr><td>Embedding Model</td><td>{self.config.retrieval.embedding_model}</td></tr>
                    <tr><td>Dataset</td><td>{self.config.dataset.name}</td></tr>
                    <tr><td>Language</td><td>{self.config.dataset.language}</td></tr>
                    <tr><td>Vector Store</td><td>{vectorstore_info}</td></tr>
                </table>
                
                <h2>Evaluation Metrics</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    {metrics_table}
                </table>
                
                <h2>LangSmith Results</h2>
                <p>View detailed results on LangSmith: <a href="{results.get('langsmith_url', '#')}" target="_blank">Open in LangSmith</a></p>
            </body>
        </html>
        """
        
        # Write the report to the output file
        with open(output_path, "w") as f:
            f.write(html_content)
        
        print(f"Report generated at {output_path}") 