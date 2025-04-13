"""Model adapters and utilities."""

from scalexi_rag_bench.models.llm_adapters import get_llm
from scalexi_rag_bench.models.embedding_adapters import get_embedding_model

__all__ = ["get_llm", "get_embedding_model"] 