"""Adapters for LLM models."""

import os
from typing import Dict, Optional, Any

from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatCohere
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from scalexi_rag_bench.config.config import LLMConfig


class BaseLLMAdapter:
    """Base adapter for LLM models."""
    
    def __init__(self, config: LLMConfig):
        """Initialize LLM adapter.
        
        Args:
            config: Configuration for LLM
        """
        self.config = config
        self._llm = self._init_llm()
    
    def _init_llm(self) -> Any:
        """Initialize LLM model.
        
        Returns:
            LLM: LLM model
        """
        raise NotImplementedError("LLM adapter must implement _init_llm")
    
    def invoke(self, inputs: Dict[str, str]) -> str:
        """Invoke LLM with inputs.
        
        Args:
            inputs: Inputs for LLM
            
        Returns:
            str: Generated output
        """
        raise NotImplementedError("LLM adapter must implement invoke")


class OpenAIAdapter(BaseLLMAdapter):
    """Adapter for OpenAI LLM models."""
    
    def _init_llm(self) -> ChatOpenAI:
        """Initialize OpenAI LLM model.
        
        Returns:
            ChatOpenAI: OpenAI LLM model
        """
        api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided")
        
        return ChatOpenAI(
            model=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            api_key=api_key
        )
    
    def invoke(self, inputs: Dict[str, str]) -> str:
        """Invoke OpenAI LLM with inputs.
        
        Args:
            inputs: Inputs for LLM
            
        Returns:
            str: Generated output
        """
        question = inputs["question"]
        context = inputs.get("context", "")
        
        prompt = f"""
        Answer the following question based on the provided context. If the information is not in the context, say so.
        
        Context: {context}
        
        Question: {question}
        
        Answer:
        """
        
        response = self._llm.invoke(prompt)
        return response.content


class CohereAdapter(BaseLLMAdapter):
    """Adapter for Cohere LLM models."""
    
    def _init_llm(self) -> ChatCohere:
        """Initialize Cohere LLM model.
        
        Returns:
            ChatCohere: Cohere LLM model
        """
        api_key = self.config.api_key or os.environ.get("COHERE_API_KEY")
        if not api_key:
            raise ValueError("Cohere API key not provided")
        
        return ChatCohere(
            model=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            cohere_api_key=api_key
        )
    
    def invoke(self, inputs: Dict[str, str]) -> str:
        """Invoke Cohere LLM with inputs.
        
        Args:
            inputs: Inputs for LLM
            
        Returns:
            str: Generated output
        """
        question = inputs["question"]
        context = inputs.get("context", "")
        
        prompt = f"""
        Answer the following question based on the provided context. If the information is not in the context, say so.
        
        Context: {context}
        
        Question: {question}
        
        Answer:
        """
        
        response = self._llm.invoke(prompt)
        return response.content


class OllamaAdapter(BaseLLMAdapter):
    """Adapter for Ollama LLM models."""
    
    def _init_llm(self) -> Ollama:
        """Initialize Ollama LLM model.
        
        Returns:
            Ollama: Ollama LLM model
        """
        return Ollama(
            model=self.config.model_name,
            temperature=self.config.temperature,
            base_url=self.config.base_url if hasattr(self.config, 'base_url') else "http://localhost:11434"
        )
    
    def invoke(self, inputs: Dict[str, str]) -> str:
        """Invoke Ollama LLM with inputs.
        
        Args:
            inputs: Inputs for LLM
            
        Returns:
            str: Generated output
        """
        question = inputs["question"]
        context = inputs.get("context", "")
        
        prompt = f"""
        Answer the following question based on the provided context. If the information is not in the context, say so.
        
        Context: {context}
        
        Question: {question}
        
        Answer:
        """
        
        response = self._llm.invoke(prompt)
        return response


class HuggingFaceAdapter(BaseLLMAdapter):
    """Adapter for HuggingFace LLM models."""
    
    def _init_llm(self) -> HuggingFacePipeline:
        """Initialize HuggingFace LLM model.
        
        Returns:
            HuggingFacePipeline: HuggingFace LLM model
        """
        model_name = self.config.model_name
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True
        )
        
        hf_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            temperature=self.config.temperature,
            max_new_tokens=self.config.max_tokens or 512,
            top_p=self.config.top_p or 0.95,
            repetition_penalty=1.1
        )
        
        return HuggingFacePipeline(pipeline=hf_pipeline)
    
    def invoke(self, inputs: Dict[str, str]) -> str:
        """Invoke HuggingFace LLM with inputs.
        
        Args:
            inputs: Inputs for LLM
            
        Returns:
            str: Generated output
        """
        question = inputs["question"]
        context = inputs.get("context", "")
        
        prompt = f"""
        Answer the following question based on the provided context. If the information is not in the context, say so.
        
        Context: {context}
        
        Question: {question}
        
        Answer:
        """
        
        response = self._llm.invoke(prompt)
        return response


def get_llm(config: LLMConfig) -> BaseLLMAdapter:
    """Get LLM adapter based on configuration.
    
    Args:
        config: Configuration for LLM
        
    Returns:
        BaseLLMAdapter: LLM adapter
    """
    provider = config.provider.lower()
    
    if provider == "openai":
        return OpenAIAdapter(config)
    elif provider == "cohere":
        return CohereAdapter(config)
    elif provider == "ollama":
        return OllamaAdapter(config)
    elif provider == "huggingface":
        return HuggingFaceAdapter(config)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}") 