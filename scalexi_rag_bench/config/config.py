"""Configuration management for RAG evaluation."""

import os
import yaml
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class RetrievalConfig(BaseModel):
    """Configuration for retrieval components."""
    
    retriever_type: str = Field(
        ..., description="Type of retriever (e.g., 'vector', 'hybrid', 'bm25')"
    )
    embedding_model: str = Field(
        ..., description="Embedding model to use for vector retrieval"
    )
    k: int = Field(4, description="Number of documents to retrieve")
    chunk_size: int = Field(1000, description="Size of document chunks")
    chunk_overlap: int = Field(200, description="Overlap between document chunks")
    
    # Optional parameters based on retriever type
    similarity_top_k: Optional[int] = Field(None, description="Number of documents for similarity search")
    search_type: Optional[str] = Field("similarity", description="Search type for vector retrieval")
    filters: Optional[Dict[str, Any]] = Field(None, description="Filters to apply during retrieval")
    
    class Config:
        extra = "allow"


class LLMConfig(BaseModel):
    """Configuration for LLM components."""
    
    model_name: str = Field(..., description="Name of the LLM model")
    provider: str = Field(..., description="Provider of the LLM (e.g., 'openai', 'deepseek', 'cohere')")
    temperature: float = Field(0.0, description="Temperature for text generation")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens for generation")
    top_p: Optional[float] = Field(None, description="Top-p sampling parameter")
    api_key: Optional[str] = Field(None, description="API key for the model provider")
    
    class Config:
        extra = "allow"


class DatasetConfig(BaseModel):
    """Configuration for dataset."""
    
    name: str = Field(..., description="Name of the dataset")
    language: str = Field(..., description="Language of the dataset")
    path: str = Field(..., description="Path to the dataset")
    format: str = Field(..., description="Format of the dataset (e.g., 'json', 'csv', 'huggingface')")
    
    class Config:
        extra = "allow"


class EvaluationMetricsConfig(BaseModel):
    """Configuration for evaluation metrics."""
    
    retrieval_metrics: List[str] = Field(
        ["precision_at_k", "recall_at_k"], 
        description="Metrics for evaluating retrieval quality"
    )
    generation_metrics: List[str] = Field(
        ["correctness", "relevance", "groundedness"], 
        description="Metrics for evaluating generation quality"
    )
    system_metrics: List[str] = Field(
        ["latency"], 
        description="Metrics for evaluating system performance"
    )
    
    class Config:
        extra = "allow"


class Config(BaseModel):
    """Main configuration for RAG evaluation."""
    
    experiment_name: str = Field(..., description="Name of the experiment")
    description: Optional[str] = Field(None, description="Description of the experiment")
    
    retrieval: RetrievalConfig = Field(..., description="Configuration for retrieval")
    llm: LLMConfig = Field(..., description="Configuration for LLM")
    dataset: DatasetConfig = Field(..., description="Configuration for dataset")
    evaluation_metrics: EvaluationMetricsConfig = Field(
        ..., description="Configuration for evaluation metrics"
    )
    
    output_dir: str = Field("./results", description="Directory to save results")
    
    @classmethod
    def from_file(cls, file_path: str) -> "Config":
        """Load configuration from a YAML file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        with open(file_path, "r") as f:
            config_dict = yaml.safe_load(f)
        
        return cls(**config_dict)
    
    def to_file(self, file_path: str) -> None:
        """Save configuration to a YAML file."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, "w") as f:
            yaml.dump(self.dict(), f, default_flow_style=False)
    
    class Config:
        extra = "allow" 