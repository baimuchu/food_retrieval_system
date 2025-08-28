"""
Traditional Embedding-Based Search System Package

This package contains the traditional semantic search approach using:
1. OpenAI Text Embeddings for semantic understanding
2. Metadata-based scoring for relevance boosting
3. Portuguese text preprocessing for optimal language handling
4. Multi-field search across item names, descriptions, categories, and taxonomy
"""

from .semantic_search import SemanticSearchSystem

__all__ = [
    'SemanticSearchSystem'
]

__version__ = '1.0.0'
__author__ = 'Prosus AI Team' 