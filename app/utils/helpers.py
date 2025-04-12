"""
Helper utility functions for the YouTube video summarization application.
"""

import os
import json
import re
import time
from typing import Dict, Any, Optional
from pathlib import Path


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a string to be used as a filename.

    Args:
        filename: The filename to sanitize

    Returns:
        Sanitized filename
    """
    # Remove invalid characters
    sanitized = re.sub(r'[\\/*?:"<>|]', "_", filename)
    # Replace spaces with underscores
    sanitized = sanitized.replace(" ", "_")
    # Limit length
    if len(sanitized) > 100:
        sanitized = sanitized[:100]
    return sanitized


def get_timestamp() -> str:
    """
    Get the current timestamp in a readable format.

    Returns:
        Formatted timestamp string
    """
    return time.strftime("%Y%m%d_%H%M%S")


def save_json(data: Dict[str, Any], filepath: str, pretty: bool = True) -> None:
    """
    Save data to a JSON file.

    Args:
        data: Data to save
        filepath: Path to save the file
        pretty: Whether to format the JSON for readability
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        if pretty:
            json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            json.dump(data, f, ensure_ascii=False)


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load data from a JSON file.

    Args:
        filepath: Path to the JSON file

    Returns:
        Loaded JSON data
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def ensure_dir(directory: str) -> None:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        directory: Directory path to ensure exists
    """
    os.makedirs(directory, exist_ok=True)


def get_file_extension(filepath: str) -> str:
    """
    Get the extension of a file.

    Args:
        filepath: Path to the file

    Returns:
        File extension (without the dot)
    """
    return Path(filepath).suffix.lstrip('.')


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix