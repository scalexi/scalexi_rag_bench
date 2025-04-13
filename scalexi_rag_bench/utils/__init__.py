"""Utility functions for RAG evaluation."""

from typing import List, Dict, Any

def load_json_file(file_path: str) -> Any:
    """Load JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Any: Loaded JSON content
    """
    import json
    
    with open(file_path, "r") as f:
        return json.load(f)


def save_json_file(data: Any, file_path: str) -> None:
    """Save data to JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save JSON file
    """
    import json
    import os
    
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        str: Formatted duration
    """
    if seconds < 1:
        return f"{seconds * 1000:.2f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.2f}s"


__all__ = [
    "load_json_file",
    "save_json_file",
    "format_duration"
] 