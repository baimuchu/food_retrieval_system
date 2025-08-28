#!/usr/bin/env python3
"""
Mock Embeddings System for Fallback Use
Provides basic embedding functionality when real API is not available
"""

import numpy as np
import hashlib
import re
from typing import List, Union

class MockEmbeddingSystem:
    """Mock embedding system that generates deterministic embeddings."""
    
    def __init__(self, dimensions: int = 1536):
        self.dimensions = dimensions
        self.seed = 42  # Fixed seed for reproducibility
        
    def text_to_embedding(self, text: str) -> np.ndarray:
        """Convert text to a mock embedding vector."""
        if not text:
            return np.zeros(self.dimensions)
        
        # Create a deterministic hash-based embedding
        text_hash = hashlib.md5(text.lower().encode()).hexdigest()
        
        # Use hash to seed numpy random generator
        rng = np.random.RandomState(int(text_hash[:8], 16))
        
        # Generate embedding vector
        embedding = rng.randn(self.dimensions)
        
        # Normalize to unit vector
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for a list of texts."""
        return [self.text_to_embedding(text) for text in texts]
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        if np.all(vec1 == 0) or np.all(vec2 == 0):
            return 0.0
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def batch_similarity(self, query_embedding: np.ndarray, item_embeddings: List[np.ndarray]) -> List[float]:
        """Calculate similarities between query and multiple items."""
        return [self.cosine_similarity(query_embedding, item_emb) for item_emb in item_embeddings]

# Global instance
mock_embedding_system = MockEmbeddingSystem()

def get_mock_embedding_system() -> MockEmbeddingSystem:
    """Get the global mock embedding system instance."""
    return mock_embedding_system
