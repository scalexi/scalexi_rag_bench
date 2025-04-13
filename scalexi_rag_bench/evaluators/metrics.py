"""Evaluation metrics for RAG systems."""

import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from typing_extensions import TypedDict, Annotated

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


# Structured output types for evaluators
class CorrectnessGrade(TypedDict):
    """Grade for correctness evaluator."""
    
    explanation: Annotated[str, "Explain your reasoning for the score"]
    correct: Annotated[bool, "True if the answer is correct, False otherwise."]


class RelevanceGrade(TypedDict):
    """Grade for relevance evaluator."""
    
    explanation: Annotated[str, "Explain your reasoning for the score"]
    relevant: Annotated[bool, "True if the answer addresses the question, False otherwise."]


class GroundednessGrade(TypedDict):
    """Grade for groundedness evaluator."""
    
    explanation: Annotated[str, "Explain your reasoning for the score"]
    grounded: Annotated[bool, "True if the answer is grounded in the documents, False otherwise"]


class PrecisionGrade(TypedDict):
    """Grade for precision evaluator."""
    
    explanation: Annotated[str, "Explain your reasoning for the score"]
    precision: Annotated[float, "Proportion of retrieved documents that are relevant"]


class RecallGrade(TypedDict):
    """Grade for recall evaluator."""
    
    explanation: Annotated[str, "Explain your reasoning for the score"]
    recall: Annotated[float, "Proportion of relevant documents that were retrieved"]


class MRRGrade(TypedDict):
    """Grade for MRR evaluator."""
    
    explanation: Annotated[str, "Explain your reasoning for the score"]
    mrr: Annotated[float, "Mean Reciprocal Rank value"]


class NDCGGrade(TypedDict):
    """Grade for NDCG evaluator."""
    
    explanation: Annotated[str, "Explain your reasoning for the score"]
    ndcg: Annotated[float, "Normalized Discounted Cumulative Gain value"]


class CoherenceGrade(TypedDict):
    """Grade for coherence evaluator."""
    
    explanation: Annotated[str, "Explain your reasoning for the score"]
    coherence: Annotated[int, "Score from 1-5 for coherence of the answer"]


class ConcisenessGrade(TypedDict):
    """Grade for conciseness evaluator."""
    
    explanation: Annotated[str, "Explain your reasoning for the score"]
    conciseness: Annotated[int, "Score from 1-5 for conciseness of the answer"]


# Retrieval Metrics
def create_precision_at_k_evaluator() -> Callable:
    """Create precision@k evaluator.
    
    Returns:
        Callable: precision@k evaluator function
    """
    precision_instructions = """You are a teacher grading a quiz. 
    You will be given a QUESTION, a list of GROUND TRUTH RELEVANT DOCUMENTS (with start_index or content), and a list of RETRIEVED DOCUMENTS. 
    Determine the proportion of retrieved documents that are relevant (match any ground truth document). 
    Explain your reasoning step-by-step. 
    Precision: A float between 0 and 1 representing the proportion of relevant documents."""
    
    precision_llm = ChatOpenAI(
        model="gpt-4o", 
        temperature=0
    ).with_structured_output(PrecisionGrade)
    
    def precision_at_k(inputs: Dict, outputs: Dict, reference_outputs: Dict) -> Dict:
        if "relevant_docs" not in reference_outputs:
            return {
                "score": 0.0,
                "explanation": "No ground truth relevant documents provided"
            }
        
        ground_truth_docs = reference_outputs.get("relevant_docs", [])
        retrieved_docs = outputs.get("context", [])
        
        # Format documents for LLM evaluation
        ground_truth_str = str(ground_truth_docs)
        retrieved_str = "\n\n".join(
            f"Doc {i+1}: " + (doc.page_content if hasattr(doc, "page_content") else str(doc))
            for i, doc in enumerate(retrieved_docs)
        )
        
        answer = f"""QUESTION: {inputs.get('question', '')}
        GROUND TRUTH RELEVANT DOCUMENTS: {ground_truth_str}
        RETRIEVED DOCUMENTS: {retrieved_str}"""
        
        grade = precision_llm.invoke([
            {"role": "system", "content": precision_instructions},
            {"role": "user", "content": answer}
        ])
        
        return {
            "score": grade["precision"],
            "explanation": grade["explanation"]
        }
    
    return precision_at_k


def create_recall_at_k_evaluator() -> Callable:
    """Create recall@k evaluator.
    
    Returns:
        Callable: recall@k evaluator function
    """
    recall_instructions = """You are a teacher grading a quiz. 
    You will be given a QUESTION, a list of GROUND TRUTH RELEVANT DOCUMENTS (with start_index or content), and a list of RETRIEVED DOCUMENTS. 
    Determine the proportion of ground truth relevant documents that were retrieved. 
    Explain your reasoning step-by-step. 
    Recall: A float between 0 and 1 representing the proportion of ground truth documents that were retrieved."""
    
    recall_llm = ChatOpenAI(
        model="gpt-4o", 
        temperature=0
    ).with_structured_output(RecallGrade)
    
    def recall_at_k(inputs: Dict, outputs: Dict, reference_outputs: Dict) -> Dict:
        if "relevant_docs" not in reference_outputs:
            return {
                "score": 0.0,
                "explanation": "No ground truth relevant documents provided"
            }
        
        ground_truth_docs = reference_outputs.get("relevant_docs", [])
        retrieved_docs = outputs.get("context", [])
        
        # Format documents for LLM evaluation
        ground_truth_str = str(ground_truth_docs)
        retrieved_str = "\n\n".join(
            f"Doc {i+1}: " + (doc.page_content if hasattr(doc, "page_content") else str(doc))
            for i, doc in enumerate(retrieved_docs)
        )
        
        answer = f"""QUESTION: {inputs.get('question', '')}
        GROUND TRUTH RELEVANT DOCUMENTS: {ground_truth_str}
        RETRIEVED DOCUMENTS: {retrieved_str}"""
        
        grade = recall_llm.invoke([
            {"role": "system", "content": recall_instructions},
            {"role": "user", "content": answer}
        ])
        
        return {
            "score": grade["recall"],
            "explanation": grade["explanation"]
        }
    
    return recall_at_k


def create_mrr_evaluator() -> Callable:
    """Create MRR (Mean Reciprocal Rank) evaluator.
    
    Returns:
        Callable: MRR evaluator function
    """
    mrr_instructions = """You are a teacher grading a quiz. 
    You will be given a QUESTION, a list of GROUND TRUTH RELEVANT DOCUMENTS (with start_index or content), and a list of RETRIEVED DOCUMENTS. 
    Calculate the Mean Reciprocal Rank (MRR), which is the reciprocal of the rank of the first relevant document in the retrieved list.
    If no relevant document is retrieved, the score is 0.
    Explain your reasoning step-by-step. 
    MRR: A float between 0 and 1."""
    
    mrr_llm = ChatOpenAI(
        model="gpt-4o", 
        temperature=0
    ).with_structured_output(MRRGrade)
    
    def mrr(inputs: Dict, outputs: Dict, reference_outputs: Dict) -> Dict:
        if "relevant_docs" not in reference_outputs:
            return {
                "score": 0.0,
                "explanation": "No ground truth relevant documents provided"
            }
        
        ground_truth_docs = reference_outputs.get("relevant_docs", [])
        retrieved_docs = outputs.get("context", [])
        
        # Format documents for LLM evaluation
        ground_truth_str = str(ground_truth_docs)
        retrieved_str = "\n\n".join(
            f"Doc {i+1}: " + (doc.page_content if hasattr(doc, "page_content") else str(doc))
            for i, doc in enumerate(retrieved_docs)
        )
        
        answer = f"""QUESTION: {inputs.get('question', '')}
        GROUND TRUTH RELEVANT DOCUMENTS: {ground_truth_str}
        RETRIEVED DOCUMENTS: {retrieved_str}"""
        
        grade = mrr_llm.invoke([
            {"role": "system", "content": mrr_instructions},
            {"role": "user", "content": answer}
        ])
        
        return {
            "score": grade["mrr"],
            "explanation": grade["explanation"]
        }
    
    return mrr


def create_ndcg_evaluator() -> Callable:
    """Create NDCG (Normalized Discounted Cumulative Gain) evaluator.
    
    Returns:
        Callable: NDCG evaluator function
    """
    ndcg_instructions = """You are a teacher grading a quiz. 
    You will be given a QUESTION, a list of GROUND TRUTH RELEVANT DOCUMENTS (with start_index or content), and a list of RETRIEVED DOCUMENTS. 
    Calculate the Normalized Discounted Cumulative Gain (NDCG), which measures the ranking quality.
    Assume all ground truth documents have relevance score of 1, and non-relevant documents have 0.
    Explain your reasoning step-by-step. 
    NDCG: A float between 0 and 1."""
    
    ndcg_llm = ChatOpenAI(
        model="gpt-4o", 
        temperature=0
    ).with_structured_output(NDCGGrade)
    
    def ndcg(inputs: Dict, outputs: Dict, reference_outputs: Dict) -> Dict:
        if "relevant_docs" not in reference_outputs:
            return {
                "score": 0.0,
                "explanation": "No ground truth relevant documents provided"
            }
        
        ground_truth_docs = reference_outputs.get("relevant_docs", [])
        retrieved_docs = outputs.get("context", [])
        
        # Format documents for LLM evaluation
        ground_truth_str = str(ground_truth_docs)
        retrieved_str = "\n\n".join(
            f"Doc {i+1}: " + (doc.page_content if hasattr(doc, "page_content") else str(doc))
            for i, doc in enumerate(retrieved_docs)
        )
        
        answer = f"""QUESTION: {inputs.get('question', '')}
        GROUND TRUTH RELEVANT DOCUMENTS: {ground_truth_str}
        RETRIEVED DOCUMENTS: {retrieved_str}"""
        
        grade = ndcg_llm.invoke([
            {"role": "system", "content": ndcg_instructions},
            {"role": "user", "content": answer}
        ])
        
        return {
            "score": grade["ndcg"],
            "explanation": grade["explanation"]
        }
    
    return ndcg


# Generation Metrics
def create_correctness_evaluator() -> Callable:
    """Create correctness evaluator.
    
    Returns:
        Callable: Correctness evaluator function
    """
    correctness_instructions = """You are a teacher grading a quiz. 
    You will be given a QUESTION, the GROUND TRUTH ANSWER, and the STUDENT ANSWER. 
    Grade based on factual accuracy relative to the ground truth. 
    A correct answer must be factually accurate and not contain conflicting statements. 
    Explain your reasoning step-by-step. 
    Correctness: True if the answer meets all criteria, False otherwise."""
    
    correctness_llm = ChatOpenAI(
        model="gpt-4o", 
        temperature=0
    ).with_structured_output(CorrectnessGrade)
    
    def correctness(inputs: Dict, outputs: Dict, reference_outputs: Dict) -> Dict:
        if "answer" not in reference_outputs:
            return {
                "score": 0.0,
                "explanation": "No ground truth answer provided"
            }
        
        answers = f"""QUESTION: {inputs.get('question', '')}
        GROUND TRUTH ANSWER: {reference_outputs.get('answer', '')}
        STUDENT ANSWER: {outputs.get('answer', '')}"""
        
        grade = correctness_llm.invoke([
            {"role": "system", "content": correctness_instructions},
            {"role": "user", "content": answers}
        ])
        
        return {
            "score": 1.0 if grade["correct"] else 0.0,
            "explanation": grade["explanation"]
        }
    
    return correctness


def create_relevance_evaluator() -> Callable:
    """Create relevance evaluator.
    
    Returns:
        Callable: Relevance evaluator function
    """
    relevance_instructions = """You are a teacher grading a quiz. 
    You will be given a QUESTION and a STUDENT ANSWER. 
    Ensure the answer is relevant and addresses the question. 
    Relevance: True if the answer addresses the question, False otherwise. 
    Explain your reasoning step-by-step."""
    
    relevance_llm = ChatOpenAI(
        model="gpt-4o", 
        temperature=0
    ).with_structured_output(RelevanceGrade)
    
    def relevance(inputs: Dict, outputs: Dict) -> Dict:
        answer = f"""QUESTION: {inputs.get('question', '')}
        STUDENT ANSWER: {outputs.get('answer', '')}"""
        
        grade = relevance_llm.invoke([
            {"role": "system", "content": relevance_instructions},
            {"role": "user", "content": answer}
        ])
        
        return {
            "score": 1.0 if grade["relevant"] else 0.0,
            "explanation": grade["explanation"]
        }
    
    return relevance


def create_groundedness_evaluator() -> Callable:
    """Create groundedness evaluator.
    
    Returns:
        Callable: Groundedness evaluator function
    """
    grounded_instructions = """You are a teacher grading a quiz. 
    You will be given a set of RETRIEVED DOCUMENTS and a STUDENT ANSWER. 
    Ensure the answer is grounded in the documents and does not contain hallucinated information. 
    Grounded: True if the answer is supported by the documents, False otherwise. 
    Explain your reasoning step-by-step."""
    
    grounded_llm = ChatOpenAI(
        model="gpt-4o", 
        temperature=0
    ).with_structured_output(GroundednessGrade)
    
    def groundedness(inputs: Dict, outputs: Dict) -> Dict:
        retrieved_docs = outputs.get("context", [])
        retrieved_str = "\n\n".join(
            f"Doc {i+1}: " + (doc.page_content if hasattr(doc, "page_content") else str(doc))
            for i, doc in enumerate(retrieved_docs)
        )
        
        answer = f"""RETRIEVED DOCUMENTS: {retrieved_str}
        STUDENT ANSWER: {outputs.get('answer', '')}"""
        
        grade = grounded_llm.invoke([
            {"role": "system", "content": grounded_instructions},
            {"role": "user", "content": answer}
        ])
        
        return {
            "score": 1.0 if grade["grounded"] else 0.0,
            "explanation": grade["explanation"]
        }
    
    return groundedness


def create_coherence_evaluator() -> Callable:
    """Create coherence evaluator.
    
    Returns:
        Callable: Coherence evaluator function
    """
    coherence_instructions = """You are a teacher grading a quiz. 
    You will be given a STUDENT ANSWER. 
    Evaluate the coherence of the answer on a scale of 1-5, where:
    1: Incoherent, disjointed, hard to follow
    2: Somewhat coherent but with significant issues
    3: Moderately coherent with some organizational issues
    4: Mostly coherent with minor issues
    5: Completely coherent, well-organized, easy to follow
    Explain your reasoning step-by-step."""
    
    coherence_llm = ChatOpenAI(
        model="gpt-4o", 
        temperature=0
    ).with_structured_output(CoherenceGrade)
    
    def coherence(inputs: Dict, outputs: Dict) -> Dict:
        answer = f"""STUDENT ANSWER: {outputs.get('answer', '')}"""
        
        grade = coherence_llm.invoke([
            {"role": "system", "content": coherence_instructions},
            {"role": "user", "content": answer}
        ])
        
        return {
            "score": grade["coherence"] / 5.0,  # Normalize to 0-1
            "explanation": grade["explanation"]
        }
    
    return coherence


def create_conciseness_evaluator() -> Callable:
    """Create conciseness evaluator.
    
    Returns:
        Callable: Conciseness evaluator function
    """
    conciseness_instructions = """You are a teacher grading a quiz. 
    You will be given a QUESTION and a STUDENT ANSWER. 
    Evaluate the conciseness of the answer on a scale of 1-5, where:
    1: Extremely verbose, repetitive, with much irrelevant content
    2: Somewhat verbose with unnecessary information
    3: Moderately concise but could be more efficient
    4: Mostly concise with minimal excess content
    5: Perfectly concise, direct, no wasted words
    Explain your reasoning step-by-step."""
    
    conciseness_llm = ChatOpenAI(
        model="gpt-4o", 
        temperature=0
    ).with_structured_output(ConcisenessGrade)
    
    def conciseness(inputs: Dict, outputs: Dict) -> Dict:
        content = f"""QUESTION: {inputs.get('question', '')}
        STUDENT ANSWER: {outputs.get('answer', '')}"""
        
        grade = conciseness_llm.invoke([
            {"role": "system", "content": conciseness_instructions},
            {"role": "user", "content": content}
        ])
        
        return {
            "score": grade["conciseness"] / 5.0,  # Normalize to 0-1
            "explanation": grade["explanation"]
        }
    
    return conciseness


# System Metrics
def create_latency_evaluator() -> Callable:
    """Create latency evaluator.
    
    Returns:
        Callable: Latency evaluator function
    """
    def latency(inputs: Dict, outputs: Dict) -> Dict:
        # Check if we have timing information in metadata
        metadata = outputs.get("metadata", {})
        retrieval_time = metadata.get("retrieval_time", 0)
        generation_time = metadata.get("generation_time", 0)
        
        total_time = retrieval_time + generation_time
        
        explanation = f"Total latency: {total_time:.2f}s (Retrieval: {retrieval_time:.2f}s, Generation: {generation_time:.2f}s)"
        
        return {
            "score": total_time,  # Raw time value in seconds
            "explanation": explanation
        }
    
    return latency


def create_cost_evaluator() -> Callable:
    """Create cost evaluator based on token usage.
    
    Returns:
        Callable: Cost evaluator function
    """
    # Cost per 1K tokens (approximate values)
    COST_PER_1K_TOKENS = {
        "gpt-4o": 0.01,  # $0.01 per 1K tokens
        "gpt-4": 0.03,   # $0.03 per 1K tokens
        "gpt-3.5-turbo": 0.0015,  # $0.0015 per 1K tokens
    }
    
    def cost(inputs: Dict, outputs: Dict) -> Dict:
        # This is a placeholder implementation
        # In real use, you would track token usage and calculate actual costs
        metadata = outputs.get("metadata", {})
        model_name = metadata.get("model_name", "unknown")
        tokens_used = metadata.get("tokens_used", 0)
        
        if tokens_used == 0:
            # Estimate based on answer length
            answer = outputs.get("answer", "")
            tokens_used = len(answer.split()) * 1.3  # Rough estimate
        
        model_cost_per_1k = COST_PER_1K_TOKENS.get(model_name, 0.01)  # Default to gpt-4o price
        estimated_cost = (tokens_used / 1000) * model_cost_per_1k
        
        explanation = f"Estimated cost: ${estimated_cost:.6f} based on approximately {tokens_used:.0f} tokens at ${model_cost_per_1k} per 1K tokens"
        
        return {
            "score": estimated_cost,
            "explanation": explanation
        }
    
    return cost


# Evaluator factories
def get_retrieval_evaluators(metric_names: List[str]) -> List[Callable]:
    """Get retrieval evaluators based on metric names.
    
    Args:
        metric_names: Names of retrieval metrics
        
    Returns:
        List[Callable]: List of retrieval evaluator functions
    """
    evaluators = []
    
    for metric in metric_names:
        if metric == "precision_at_k":
            evaluators.append(create_precision_at_k_evaluator())
        elif metric == "recall_at_k":
            evaluators.append(create_recall_at_k_evaluator())
        elif metric == "mrr":
            evaluators.append(create_mrr_evaluator())
        elif metric == "ndcg":
            evaluators.append(create_ndcg_evaluator())
    
    return evaluators


def get_generation_evaluators(metric_names: List[str]) -> List[Callable]:
    """Get generation evaluators based on metric names.
    
    Args:
        metric_names: Names of generation metrics
        
    Returns:
        List[Callable]: List of generation evaluator functions
    """
    evaluators = []
    
    for metric in metric_names:
        if metric == "correctness":
            evaluators.append(create_correctness_evaluator())
        elif metric == "relevance":
            evaluators.append(create_relevance_evaluator())
        elif metric == "groundedness":
            evaluators.append(create_groundedness_evaluator())
        elif metric == "coherence":
            evaluators.append(create_coherence_evaluator())
        elif metric == "conciseness":
            evaluators.append(create_conciseness_evaluator())
    
    return evaluators


def get_system_evaluators(metric_names: List[str]) -> List[Callable]:
    """Get system evaluators based on metric names.
    
    Args:
        metric_names: Names of system metrics
        
    Returns:
        List[Callable]: List of system evaluator functions
    """
    evaluators = []
    
    for metric in metric_names:
        if metric == "latency":
            evaluators.append(create_latency_evaluator())
        elif metric == "cost":
            evaluators.append(create_cost_evaluator())
    
    return evaluators 