"""Utility functions for DocuRAG system."""

import hashlib
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    logger.debug(f"Set random seeds to {seed}")


def get_device() -> torch.device:
    """Get the appropriate device for computations.
    
    Returns:
        torch.device: Available device (cuda/mps/cpu)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS device")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    
    return device


def calculate_file_hash(file_path: Union[str, Path]) -> str:
    """Calculate SHA-256 hash of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        str: Hexadecimal hash string
    """
    sha256_hash = hashlib.sha256()
    
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    
    return sha256_hash.hexdigest()


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        str: Formatted size string
    """
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = int(np.floor(np.log(size_bytes) / np.log(1024)))
    p = np.power(1024, i)
    s = round(size_bytes / p, 2)
    
    return f"{s} {size_names[i]}"


def validate_file_extension(file_path: Union[str, Path], allowed_extensions: List[str]) -> bool:
    """Validate file extension against allowed extensions.
    
    Args:
        file_path: Path to the file
        allowed_extensions: List of allowed extensions (with dots)
        
    Returns:
        bool: True if extension is allowed
    """
    file_extension = Path(file_path).suffix.lower()
    return file_extension in [ext.lower() for ext in allowed_extensions]


def safe_filename(filename: str) -> str:
    """Create a safe filename by removing/replacing problematic characters.
    
    Args:
        filename: Original filename
        
    Returns:
        str: Safe filename
    """
    # Replace problematic characters
    unsafe_chars = '<>:"/\\|?*'
    safe_name = filename
    
    for char in unsafe_chars:
        safe_name = safe_name.replace(char, '_')
    
    # Remove leading/trailing dots and spaces
    safe_name = safe_name.strip('. ')
    
    # Ensure filename is not empty
    if not safe_name:
        safe_name = "unnamed_file"
    
    return safe_name


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split a list into chunks of specified size.
    
    Args:
        lst: List to split
        chunk_size: Size of each chunk
        
    Returns:
        List[List[Any]]: List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def timing_decorator(func):
    """Decorator to measure function execution time.
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        logger.debug(f"{func.__name__} executed in {execution_time:.2f}s")
        
        return result
    
    return wrapper


def extract_metadata_from_filename(filename: str) -> Dict[str, Any]:
    """Extract metadata from filename patterns.
    
    Args:
        filename: Name of the file
        
    Returns:
        Dict[str, Any]: Extracted metadata
    """
    metadata = {
        "filename": filename,
        "extension": Path(filename).suffix.lower(),
        "name_without_ext": Path(filename).stem
    }
    
    # Try to extract date patterns (YYYY-MM-DD or YYYY_MM_DD)
    import re
    date_pattern = r'(\d{4}[-_]\d{2}[-_]\d{2})'
    date_match = re.search(date_pattern, filename)
    if date_match:
        metadata["date_in_filename"] = date_match.group(1).replace('_', '-')
    
    # Extract version patterns (v1.0, version_2, etc.)
    version_pattern = r'[v|version][\s_-]?(\d+\.?\d*)'
    version_match = re.search(version_pattern, filename, re.IGNORECASE)
    if version_match:
        metadata["version"] = version_match.group(1)
    
    return metadata


def calculate_text_statistics(text: str) -> Dict[str, int]:
    """Calculate basic text statistics.
    
    Args:
        text: Input text
        
    Returns:
        Dict[str, int]: Text statistics
    """
    words = text.split()
    sentences = text.split('.')
    paragraphs = text.split('\n\n')
    
    return {
        "char_count": len(text),
        "word_count": len(words),
        "sentence_count": len([s for s in sentences if s.strip()]),
        "paragraph_count": len([p for p in paragraphs if p.strip()]),
        "avg_word_length": sum(len(word) for word in words) / len(words) if words else 0
    }


def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and special characters.
    
    Args:
        text: Raw text
        
    Returns:
        str: Cleaned text
    """
    import re
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,;:!?()-]', '', text)
    
    # Remove multiple consecutive punctuation
    text = re.sub(r'([.,;:!?]){2,}', r'\1', text)
    
    return text.strip()


class PerformanceMonitor:
    """Monitor performance metrics during processing."""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        self.start_times[operation] = time.time()
    
    def end_timer(self, operation: str) -> float:
        """End timing an operation and return duration."""
        if operation not in self.start_times:
            logger.warning(f"Timer for {operation} was not started")
            return 0.0
        
        duration = time.time() - self.start_times[operation]
        if operation not in self.metrics:
            self.metrics[operation] = []
        
        self.metrics[operation].append(duration)
        del self.start_times[operation]
        
        return duration
    
    def get_stats(self, operation: str) -> Dict[str, float]:
        """Get statistics for an operation."""
        if operation not in self.metrics or not self.metrics[operation]:
            return {"count": 0, "avg": 0.0, "min": 0.0, "max": 0.0, "total": 0.0}
        
        times = self.metrics[operation]
        return {
            "count": len(times),
            "avg": np.mean(times),
            "min": np.min(times),
            "max": np.max(times),
            "total": np.sum(times)
        }
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics.clear()
        self.start_times.clear()