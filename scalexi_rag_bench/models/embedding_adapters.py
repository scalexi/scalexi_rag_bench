"""Adapters for embedding models."""

import os
from typing import Dict, List, Optional, Any, Union

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import (
    HuggingFaceEmbeddings,
    CohereEmbeddings,
    HuggingFaceBgeEmbeddings
)


class BaseEmbeddingAdapter:
    """Base adapter for embedding models."""
    
    def __init__(self, model_name: str, **kwargs):
        """Initialize embedding adapter.
        
        Args:
            model_name: Name of the embedding model
            **kwargs: Additional arguments for embedding model
        """
        self.model_name = model_name
        self.kwargs = kwargs
        self._embeddings = self._init_embeddings()
    
    def _init_embeddings(self) -> Embeddings:
        """Initialize embedding model.
        
        Returns:
            Embeddings: Embedding model
        """
        raise NotImplementedError("Embedding adapter must implement _init_embeddings")
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query text.
        
        Args:
            text: Text to embed
            
        Returns:
            List[float]: Embedding
        """
        return self._embeddings.embed_query(text)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents.
        
        Args:
            texts: Texts to embed
            
        Returns:
            List[List[float]]: Embeddings
        """
        return self._embeddings.embed_documents(texts)


class OpenAIEmbeddingAdapter(BaseEmbeddingAdapter):
    """Adapter for OpenAI embedding models."""
    
    def _init_embeddings(self) -> OpenAIEmbeddings:
        """Initialize OpenAI embedding model.
        
        Returns:
            OpenAIEmbeddings: OpenAI embedding model
        """
        api_key = self.kwargs.get("api_key") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided")
        
        return OpenAIEmbeddings(
            model=self.model_name,
            api_key=api_key
        )


class CohereEmbeddingAdapter(BaseEmbeddingAdapter):
    """Adapter for Cohere embedding models."""
    
    def _init_embeddings(self) -> CohereEmbeddings:
        """Initialize Cohere embedding model.
        
        Returns:
            CohereEmbeddings: Cohere embedding model
        """
        api_key = self.kwargs.get("api_key") or os.environ.get("COHERE_API_KEY")
        if not api_key:
            raise ValueError("Cohere API key not provided")
        
        return CohereEmbeddings(
            model=self.model_name,
            cohere_api_key=api_key
        )


class HuggingFaceEmbeddingAdapter(BaseEmbeddingAdapter):
    """Adapter for HuggingFace embedding models."""
    
    def _init_embeddings(self) -> HuggingFaceEmbeddings:
        """Initialize HuggingFace embedding model.
        
        Returns:
            HuggingFaceEmbeddings: HuggingFace embedding model
        """
        return HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={
                "device": self.kwargs.get("device", "cpu")
            }
        )


class BGEEmbeddingAdapter(BaseEmbeddingAdapter):
    """Adapter for BGE embedding models."""
    
    def _init_embeddings(self) -> HuggingFaceBgeEmbeddings:
        """Initialize BGE embedding model.
        
        Returns:
            HuggingFaceBgeEmbeddings: BGE embedding model
        """
        return HuggingFaceBgeEmbeddings(
            model_name=self.model_name,
            model_kwargs={
                "device": self.kwargs.get("device", "cpu")
            }
        )


def get_embedding_model(model_name: str, **kwargs) -> BaseEmbeddingAdapter:
    """Get embedding adapter based on model name.
    
    Args:
        model_name: Name of the embedding model
        **kwargs: Additional arguments for embedding model
        
    Returns:
        BaseEmbeddingAdapter: Embedding adapter
    """
    model_name_lower = model_name.lower()
    
    if "openai" in model_name_lower or "text-embedding" in model_name_lower:
        return OpenAIEmbeddingAdapter(model_name, **kwargs)
    elif "cohere" in model_name_lower:
        return CohereEmbeddingAdapter(model_name, **kwargs)
    elif "bge" in model_name_lower:
        return BGEEmbeddingAdapter(model_name, **kwargs)
    elif "multilingual-e5" in model_name_lower:
        # Special case for multilingual e5 models
        return HuggingFaceEmbeddingAdapter(model_name, **kwargs)
    else:
        # Default to HuggingFace embeddings
        return HuggingFaceEmbeddingAdapter(model_name, **kwargs) 