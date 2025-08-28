"""
LLM-Based Search Systems Package

This package contains three different LLM-based approaches for semantic search:
1. Simple LLM System - Intelligent mock LLM with food-specific logic
2. Advanced LLM System - Real API integration with fallback
3. Comprehensive LLM System - Full-featured with multiple strategies
"""

from .llm_search_simple import SimpleLLMSearch
from .advanced_llm_search import AdvancedLLMSearch
from .llm_search_system import LLMSearchSystem

__all__ = [
    'SimpleLLMSearch',
    'AdvancedLLMSearch', 
    'LLMSearchSystem'
]

__version__ = '1.0.0'
__author__ = 'Prosus AI Team' 