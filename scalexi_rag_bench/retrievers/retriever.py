"""Retriever implementations for RAG evaluation."""

import os
from typing import Any, Dict, List, Optional, Union

from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever

from scalexi_rag_bench.config.config import RetrievalConfig
from scalexi_rag_bench.models.embedding_adapters import BaseEmbeddingAdapter
from scalexi_rag_bench.vectorstores import load_vectorstore, create_retriever_from_vectorstore


class BaseRetrieverAdapter:
    """Base adapter for retrievers."""
    
    def __init__(self, config: RetrievalConfig, embedding_model: BaseEmbeddingAdapter, vectorstore_path: Optional[str] = None):
        """Initialize retriever adapter.
        
        Args:
            config: Configuration for retriever
            embedding_model: Embedding model for vector retrieval
            vectorstore_path: Path to vector store (if available)
        """
        self.config = config
        self.embedding_model = embedding_model
        self.vectorstore_path = vectorstore_path
        self._retriever = self._init_retriever()
    
    def _init_retriever(self) -> BaseRetriever:
        """Initialize retriever.
        
        Returns:
            BaseRetriever: Retriever
        """
        raise NotImplementedError("Retriever adapter must implement _init_retriever")
    
    def retrieve(self, query: str) -> List[Any]:
        """Retrieve documents for query.
        
        Args:
            query: Query for retrieval
            
        Returns:
            List[Any]: Retrieved documents
        """
        return self._retriever.invoke(query)


class VectorRetrieverAdapter(BaseRetrieverAdapter):
    """Adapter for vector retrievers."""
    
    def _init_retriever(self) -> BaseRetriever:
        """Initialize vector retriever.
        
        Returns:
            BaseRetriever: Vector retriever
        """
        # If vectorstore_path is provided, load the vectorstore
        if self.vectorstore_path and os.path.exists(self.vectorstore_path):
            print(f"Loading vector store from {self.vectorstore_path}")
            vector_store = load_vectorstore(self.vectorstore_path, self.embedding_model._embeddings)
        else:
            # For fallback or demo purposes
            print("No vector store path provided, using in-memory store with empty documents")
            vector_store = self._get_empty_vector_store()
        
        return vector_store.as_retriever(
            search_type=self.config.search_type,
            search_kwargs={
                "k": self.config.k,
                **(self.config.filters or {})
            }
        )
    
    def _get_empty_vector_store(self) -> VectorStore:
        """Get empty vector store for initialization.
        
        Returns:
            VectorStore: Empty vector store
        """
        from langchain_core.documents import Document
        
        # Create a minimal document for initialization
        empty_docs = [Document(page_content="Initial document for vector store")]
        
        # Use the internal _embeddings object instead of the adapter itself
        return FAISS.from_documents(documents=empty_docs, embedding=self.embedding_model._embeddings)


class ChromaRetrieverAdapter(BaseRetrieverAdapter):
    """Adapter for Chroma vector retriever."""
    
    def _init_retriever(self) -> BaseRetriever:
        """Initialize Chroma retriever.
        
        Returns:
            BaseRetriever: Chroma retriever
        """
        # If vectorstore_path is provided, use it
        if self.vectorstore_path and os.path.exists(self.vectorstore_path):
            print(f"Loading Chroma vector store from {self.vectorstore_path}")
            vector_store = Chroma(
                persist_directory=self.vectorstore_path,
                embedding_function=self.embedding_model._embeddings
            )
        else:
            # Fallback to in-memory store
            print("No vector store path provided, using in-memory Chroma store")
            vector_store = Chroma(
                collection_name="rag_evaluation",
                embedding_function=self.embedding_model._embeddings
            )
        
        return vector_store.as_retriever(
            search_type=self.config.search_type,
            search_kwargs={
                "k": self.config.k,
                **(self.config.filters or {})
            }
        )


class BM25RetrieverAdapter(BaseRetrieverAdapter):
    """Adapter for BM25 retriever."""
    
    def _init_retriever(self) -> BM25Retriever:
        """Initialize BM25 retriever.
        
        Returns:
            BM25Retriever: BM25 retriever
        """
        # In a real implementation, this would load documents from a path
        # For now, we'll use an empty BM25 retriever
        # TODO: Implement document loading from vectorstore_path for BM25
        return BM25Retriever.from_documents(
            documents=[],
            tokenizer=None,  # Default tokenizer
            k=self.config.k
        )


class HybridRetrieverAdapter(BaseRetrieverAdapter):
    """Adapter for hybrid retrievers."""
    
    def _init_retriever(self) -> BaseRetriever:
        """Initialize hybrid retriever.
        
        Returns:
            BaseRetriever: Hybrid retriever
        """
        from langchain_community.retrievers import EnsembleRetriever
        
        # Initialize vector retriever
        if self.vectorstore_path and os.path.exists(self.vectorstore_path):
            print(f"Loading vector store from {self.vectorstore_path}")
            vector_store = load_vectorstore(self.vectorstore_path, self.embedding_model._embeddings)
        else:
            # Fallback to empty store
            print("No vector store path provided, using in-memory store with empty documents")
            vector_store = self._get_empty_vector_store()
        
        vector_retriever = vector_store.as_retriever(
            search_type=self.config.search_type,
            search_kwargs={"k": self.config.k}
        )
        
        # Set up BM25 retriever
        bm25_retriever = BM25Retriever.from_documents(
            documents=[],  # TODO: Load documents for BM25
            tokenizer=None,
            k=self.config.k
        )
        
        # Create ensemble retriever
        # Weight of 0.5 for each retriever
        return EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.5, 0.5]
        )
    
    def _get_empty_vector_store(self) -> VectorStore:
        """Get empty vector store for initialization.
        
        Returns:
            VectorStore: Empty vector store
        """
        from langchain_core.documents import Document
        
        # Create a minimal document for initialization
        empty_docs = [Document(page_content="Initial document for vector store")]
        
        # Use the internal _embeddings object instead of the adapter itself
        return FAISS.from_documents(documents=empty_docs, embedding=self.embedding_model._embeddings)


def get_retriever(config: RetrievalConfig, embedding_model: BaseEmbeddingAdapter, 
                 vectorstore_path: Optional[str] = None) -> BaseRetrieverAdapter:
    """Get retriever adapter based on configuration.
    
    Args:
        config: Configuration for retriever
        embedding_model: Embedding model for vector retrieval
        vectorstore_path: Path to vector store (if available)
        
    Returns:
        BaseRetrieverAdapter: Retriever adapter
    """
    retriever_type = config.retriever_type.lower()
    
    if retriever_type == "vector":
        return VectorRetrieverAdapter(config, embedding_model, vectorstore_path)
    elif retriever_type == "chroma":
        return ChromaRetrieverAdapter(config, embedding_model, vectorstore_path)
    elif retriever_type == "bm25":
        return BM25RetrieverAdapter(config, embedding_model, vectorstore_path)
    elif retriever_type == "hybrid":
        return HybridRetrieverAdapter(config, embedding_model, vectorstore_path)
    else:
        raise ValueError(f"Unsupported retriever type: {retriever_type}") 