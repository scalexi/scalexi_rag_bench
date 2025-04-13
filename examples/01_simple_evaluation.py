#!/usr/bin/env python
"""Minimal example script to demonstrate RAG evaluation without complex dependencies."""

import os
import sys
import json
import time
from pathlib import Path

# Add parent directory to path to import necessary modules
sys.path.append(str(Path(__file__).parent.parent))

# Directly import the classes we need without going through from scalexi_rag_bench package
from scalexi_rag_bench.config.config import Config
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS


def create_simple_dataset():
    """Create a simple dataset if it doesn't exist."""
    dataset_path = Path(__file__).parent.parent / "data/english/qa_dataset.json"
    
    # Check if dataset exists
    if dataset_path.exists():
        print(f"Using existing dataset at {dataset_path}")
        return str(dataset_path)
    
    # Create directory if it doesn't exist
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Simple dataset with two QA pairs
    simple_dataset = [
        {
            "question": "What is Retrieval-Augmented Generation?",
            "answer": "Retrieval-Augmented Generation (RAG) is an AI framework that enhances language models by retrieving relevant information from external sources before generating responses.",
            "relevant_docs": [
                {
                    "content": "Retrieval-Augmented Generation (RAG) combines retrieval and generation approaches in NLP. An LLM first retrieves relevant information and then uses it to generate accurate responses.",
                    "metadata": {
                        "source": "rag-overview",
                        "start_index": 100
                    }
                }
            ]
        },
        {
            "question": "What are the main components of a RAG system?",
            "answer": "A RAG system typically consists of three main components: a retriever, an indexing system, and a generator.",
            "relevant_docs": [
                {
                    "content": "RAG architectures generally comprise three key components: the retriever, the indexing system, and the generator.",
                    "metadata": {
                        "source": "rag-architecture",
                        "start_index": 780
                    }
                }
            ]
        }
    ]
    
    # Save dataset
    with open(dataset_path, "w") as f:
        json.dump(simple_dataset, f, indent=2)
    
    print(f"Created simple dataset at {dataset_path}")
    return str(dataset_path)


def simple_rag_evaluation():
    """Run a simplified RAG evaluation."""
    # Check for OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: Missing OPENAI_API_KEY environment variable")
        print("Please set this variable before running the script.")
        sys.exit(1)
    
    # Create simple dataset
    dataset_path = create_simple_dataset()
    
    # Load dataset
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    
    print("\nLoaded dataset with the following questions:")
    for item in dataset:
        print(f"- {item['question']}")
    
    # Initialize OpenAI embedding model
    print("\nInitializing OpenAI embeddings...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Create a simple vector store from documents
    print("Creating vector store from documents...")
    docs = []
    for item in dataset:
        for doc in item.get("relevant_docs", []):
            docs.append(Document(
                page_content=doc["content"],
                metadata=doc["metadata"]
            ))
    
    vector_store = FAISS.from_documents(documents=docs, embedding=embeddings)
    
    # Initialize OpenAI LLM
    print("Initializing OpenAI LLM...")
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    # Simple evaluation loop
    print("\nRunning evaluation on", len(dataset), "questions...\n")
    results = []
    
    for i, item in enumerate(dataset):
        question = item["question"]
        ground_truth = item["answer"]
        
        print(f"Question {i+1}: {question}")
        
        # Retrieve documents
        retrieved_docs = vector_store.similarity_search(question, k=2)
        retrieved_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
        
        # Generate answer
        prompt = f"""
        Answer the following question based on the provided context. If the information is not in the context, say so.
        
        Context: {retrieved_content}
        
        Question: {question}
        
        Answer:
        """
        
        answer = llm.invoke(prompt).content
        
        print(f"Answer: {answer}")
        print(f"Ground truth: {ground_truth}")
        print("-" * 50)
        
        # Store result
        results.append({
            "question": question,
            "answer": answer,
            "ground_truth": ground_truth,
            "retrieved_docs": [doc.page_content for doc in retrieved_docs]
        })
    
    # Save results
    output_dir = Path(__file__).parent.parent / "results/minimal_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for filenames
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Save JSON results
    with open(output_dir / f"results_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Generate HTML report
    html_report = generate_html_report(results)
    
    # Save HTML report
    with open(output_dir / f"report_{timestamp}.html", "w") as f:
        f.write(html_report)
    
    print(f"\nEvaluation complete. Results saved to {output_dir / f'results_{timestamp}.json'}")
    print(f"HTML report available at {output_dir / f'report_{timestamp}.html'}")


def generate_html_report(results):
    """Generate a simple HTML report from evaluation results."""
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Simple RAG Evaluation Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
        }
        h1, h2, h3 {
            color: #333;
        }
        .question {
            margin-bottom: 30px;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            background-color: #f9f9f9;
        }
        .question h3 {
            margin-top: 0;
            border-bottom: 1px solid #ddd;
            padding-bottom: 10px;
        }
        .metadata {
            font-size: 0.9em;
            color: #666;
            margin-bottom: 10px;
        }
        .docs {
            background-color: #f0f0f0;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 15px;
        }
        .answer, .ground-truth {
            padding: 10px;
            border-radius: 5px;
        }
        .answer {
            background-color: #e6f7ff;
            border-left: 4px solid #1890ff;
        }
        .ground-truth {
            background-color: #f6ffed;
            border-left: 4px solid #52c41a;
        }
        .summary {
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 30px;
        }
    </style>
</head>
<body>
    <h1>Simple RAG Evaluation Report</h1>
    <div class="summary">
        <h2>Evaluation Summary</h2>
        <p>Timestamp: """ + time.strftime("%Y-%m-%d %H:%M:%S") + """</p>
        <p>Total Questions: """ + str(len(results)) + """</p>
    </div>
    """
    
    # Add each question and answer
    for i, result in enumerate(results):
        html += f"""
    <div class="question">
        <h3>Question {i+1}: {result['question']}</h3>
        
        <h4>Retrieved Documents:</h4>
        <div class="docs">
            <ul>
        """
        
        for j, doc in enumerate(result['retrieved_docs']):
            html += f"""
                <li>Document {j+1}: {doc}</li>
            """
            
        html += """
            </ul>
        </div>
        
        <h4>Model Answer:</h4>
        <div class="answer">
            <p>{}</p>
        </div>
        
        <h4>Ground Truth:</h4>
        <div class="ground-truth">
            <p>{}</p>
        </div>
    </div>
    """.format(result['answer'], result['ground_truth'])
    
    html += """
</body>
</html>
    """
    
    return html


if __name__ == "__main__":
    simple_rag_evaluation() 