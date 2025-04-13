#!/usr/bin/env python
"""Utility script for creating evaluation datasets."""

import os
import json
import argparse
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional
from pathlib import Path

# Document type for QA dataset
QA_Item = Dict[str, Any]


def scrape_webpage(url: str) -> str:
    """Scrape the content of a webpage.
    
    Args:
        url: URL of the webpage
        
    Returns:
        str: HTML content of the webpage
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    return response.text


def extract_text_from_html(html: str) -> List[Dict[str, Any]]:
    """Extract text from HTML content.
    
    Args:
        html: HTML content
        
    Returns:
        List[Dict[str, Any]]: List of extracted text chunks with metadata
    """
    soup = BeautifulSoup(html, "html.parser")
    
    # Remove scripts, styles, etc.
    for element in soup(["script", "style", "head", "header", "footer", "nav"]):
        element.extract()
    
    # Extract text from different content sections
    chunks = []
    start_index = 0
    
    # Extract headings and paragraphs
    for element in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p"]):
        text = element.get_text().strip()
        if text:
            # Get the tag name to use as source
            tag_name = element.name
            
            chunks.append({
                "content": text,
                "metadata": {
                    "source": tag_name,
                    "start_index": start_index
                }
            })
            
            start_index += len(text) + 1  # +1 for the newline
    
    return chunks


def prepare_qa_dataset(url: str, questions: List[Dict[str, str]]) -> List[QA_Item]:
    """Prepare QA dataset from a URL and a list of questions.
    
    Args:
        url: URL to scrape
        questions: List of questions with answers
        
    Returns:
        List[QA_Item]: QA dataset
    """
    # Scrape the webpage
    html = scrape_webpage(url)
    
    # Extract text chunks
    chunks = extract_text_from_html(html)
    
    # Prepare dataset
    dataset = []
    
    for qa_item in questions:
        question = qa_item["question"]
        answer = qa_item["answer"]
        
        # For simplicity, randomly assign relevant documents
        # In a real implementation, this would be done more carefully
        relevant_docs = []
        
        dataset.append({
            "question": question,
            "answer": answer,
            "relevant_docs": relevant_docs
        })
    
    return dataset


def save_dataset(dataset: List[QA_Item], output_path: str) -> None:
    """Save dataset to file.
    
    Args:
        dataset: Dataset to save
        output_path: Path to save the dataset
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)


def main(url: str, questions_path: str, output_path: str):
    """Create QA dataset from URL and questions.
    
    Args:
        url: URL to scrape
        questions_path: Path to questions JSON file
        output_path: Path to save the dataset
    """
    # Load questions
    with open(questions_path, "r") as f:
        questions = json.load(f)
    
    # Prepare dataset
    dataset = prepare_qa_dataset(url, questions)
    
    # Save dataset
    save_dataset(dataset, output_path)
    
    print(f"Dataset created with {len(dataset)} QA pairs and saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create QA dataset from a URL")
    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="URL to scrape"
    )
    parser.add_argument(
        "--questions",
        type=str,
        required=True,
        help="Path to questions JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataset.json",
        help="Path to save the dataset"
    )
    
    args = parser.parse_args()
    main(args.url, args.questions, args.output) 