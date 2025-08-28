"""
Utility Modules Package

This package contains utility modules for the Prosus AI project:
1. Configuration management
2. Mock embedding system
3. Data examination utilities
"""

from .config import *
from .mock_embeddings import MockEmbeddingSystem, mock_embedding_system

__all__ = [
    'MockEmbeddingSystem',
    'mock_embedding_system'
]

__version__ = '1.0.0'
__author__ = 'Prosus AI Team' 