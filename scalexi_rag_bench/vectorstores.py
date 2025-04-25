"""Vector store creation and management for RAG evaluation."""

import os
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import hashlib

from langchain_community.document_loaders import (
    TextLoader, 
    PyPDFLoader, 
    DirectoryLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


def create_vectorstore(
    source_path: str,
    embedding_model: Embeddings,
    output_dir: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    file_types: Optional[List[str]] = None,
    dataset_name: Optional[str] = None
) -> str:
    """Create a vector store from source documents.
    
    Args:
        source_path: Path to the source document or directory
        embedding_model: Embedding model to use
        output_dir: Directory to save the vector store
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        file_types: List of file types to process (e.g., ["txt", "pdf"])
        dataset_name: Optional name for the dataset. If not provided, will be derived from source_path
        
    Returns:
        str: Path to the created vector store
    """
    # Set default file types if not provided
    if file_types is None:
        file_types = ["txt", "pdf", "json"]
    
    # Get dataset name if not provided
    if dataset_name is None:
        dataset_name = Path(source_path).stem
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a hash of the source path and parameters to avoid reprocessing
    hasher = hashlib.md5()
    hasher.update(f"{source_path}_{chunk_size}_{chunk_overlap}".encode())
    source_hash = hasher.hexdigest()[:10]
    
    # Vector store path
    vectorstore_path = os.path.join(output_dir, f"{dataset_name}_{source_hash}")
    
    # Check if vector store already exists
    if os.path.exists(vectorstore_path):
        print(f"Vector store already exists at {vectorstore_path}")
        return vectorstore_path
    
    # Load documents based on source type
    print(f"Loading documents from {source_path}")
    documents = []
    
    # Check if source path exists
    source_path = Path(source_path)
    if not source_path.exists():
        print(f"Error: Source path {source_path} does not exist")
        return ""
    
    print(f"Source path type: {source_path.is_file() and 'file' or source_path.is_dir() and 'directory' or 'unknown'}")
    if source_path.is_file():
        # Single file
        print(f"Processing single file with suffix: {source_path.suffix.lower()}")
        if source_path.suffix.lower() == '.pdf':
            loader = PyPDFLoader(str(source_path))
            documents = loader.load()
        elif source_path.suffix.lower() == '.txt':
            loader = TextLoader(str(source_path))
            documents = loader.load()
        elif source_path.suffix.lower() == '.json':
            # Special handling for JSON files
            print(f"Loading JSON file: {source_path}")
            with open(source_path, 'r') as f:
                data = json.load(f)
            
            print(f"JSON data type: {type(data)}")
            if isinstance(data, list):
                print(f"JSON list length: {len(data)}")
                print(f"First item type: {type(data[0]) if data else 'N/A'}")
            
            # Check if it's a dataset with relevant_docs
            if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                print(f"Processing list of dictionaries, items: {len(data)}")
                doc_count = 0
                for i, item in enumerate(data):
                    if "relevant_docs" in item:
                        doc_list = item.get("relevant_docs", [])
                        print(f"Item {i} has {len(doc_list)} relevant_docs")
                        for doc in doc_list:
                            if "content" in doc and "metadata" in doc:
                                documents.append(Document(
                                    page_content=doc["content"],
                                    metadata=doc["metadata"]
                                ))
                                doc_count += 1
                print(f"Extracted {doc_count} documents from relevant_docs fields")
            # If not a standard format, just extract text
            else:
                documents.append(Document(
                    page_content=json.dumps(data),
                    metadata={"source": str(source_path)}
                ))
                print("Extracted 1 document from JSON data (non-standard format)")
    elif source_path.is_dir():
        # Directory
        for file_type in file_types:
            if file_type == "pdf":
                loader = DirectoryLoader(str(source_path), glob=f"**/*.{file_type}", loader_cls=PyPDFLoader)
                documents.extend(loader.load())
            elif file_type == "txt":
                loader = DirectoryLoader(str(source_path), glob=f"**/*.{file_type}", loader_cls=TextLoader)
                documents.extend(loader.load())
            elif file_type == "json":
                # Process JSON files one by one
                json_files = list(source_path.glob(f"**/*.{file_type}"))
                print(f"Found {len(json_files)} JSON files in directory")
                for json_file in json_files:
                    with open(json_file, 'r') as f:
                        try:
                            data = json.load(f)
                            # Check if it's a dataset with relevant_docs
                            if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                                doc_count = 0
                                for item in data:
                                    if "relevant_docs" in item:
                                        for doc in item["relevant_docs"]:
                                            if "content" in doc and "metadata" in doc:
                                                documents.append(Document(
                                                    page_content=doc["content"],
                                                    metadata=doc["metadata"]
                                                ))
                                                doc_count += 1
                                print(f"Extracted {doc_count} documents from {json_file}")
                            # If not a standard format, just extract text
                            else:
                                documents.append(Document(
                                    page_content=json.dumps(data),
                                    metadata={"source": str(json_file)}
                                ))
                                print(f"Extracted 1 document from {json_file} (non-standard format)")
                        except json.JSONDecodeError:
                            print(f"Error parsing JSON file: {json_file}")
    
    print(f"Loaded {len(documents)} documents")
    
    # Split documents
    if documents:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        split_docs = text_splitter.split_documents(documents)
        print(f"Split into {len(split_docs)} chunks")
        
        # Create vector store
        vectorstore = FAISS.from_documents(split_docs, embedding_model)
        vectorstore.save_local(vectorstore_path)
        print(f"Vector store created and saved to {vectorstore_path}")
        
        # Save metadata for reference
        metadata = {
            "source_path": str(source_path),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "document_count": len(documents),
            "chunk_count": len(split_docs),
            "dataset_name": dataset_name
        }
        
        with open(os.path.join(vectorstore_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        return vectorstore_path
    else:
        print("No documents loaded, vector store creation failed")
        return ""


def load_vectorstore(
    vectorstore_path: str,
    embedding_model: Embeddings
) -> FAISS:
    """Load a vector store from disk.
    
    Args:
        vectorstore_path: Path to the vector store
        embedding_model: Embedding model to use
        
    Returns:
        FAISS: Loaded vector store
    """
    if not os.path.exists(vectorstore_path):
        raise FileNotFoundError(f"Vector store not found at {vectorstore_path}")
    
    vectorstore = FAISS.load_local(vectorstore_path, embedding_model, allow_dangerous_deserialization=True)
    print(f"Vector store loaded from {vectorstore_path}")
    
    return vectorstore


def create_retriever_from_vectorstore(
    vectorstore_path: str,
    embedding_model: Embeddings,
    k: int = 4,
    search_type: str = "similarity"
):
    """Create a retriever from a vector store.
    
    Args:
        vectorstore_path: Path to the vector store
        embedding_model: Embedding model to use
        k: Number of documents to retrieve
        search_type: Search type (similarity, mmr)
        
    Returns:
        Retriever: The retriever
    """
    vectorstore = load_vectorstore(vectorstore_path, embedding_model)
    
    search_kwargs = {"k": k}
    if search_type == "mmr":
        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs=search_kwargs)
    else:
        retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    
    return retriever 