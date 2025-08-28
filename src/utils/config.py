"""
Configuration file for the Prosus AI Semantic Search System.
"""

import os

def get_api_key_from_file():
    """Read API key from LLM_Key.txt file for security."""
    # Look for LLM_Key.txt in the same directory as config.py (src/utils/)
    current_dir = os.path.dirname(__file__)
    key_file = os.path.join(current_dir, 'LLM_Key.txt')
    
    if os.path.exists(key_file):
        try:
            with open(key_file, 'r', encoding='utf-8') as f:
                api_key = f.read().strip()
                if api_key and len(api_key) > 10:  # Basic validation
                    return api_key
                else:
                    raise ValueError("Invalid API key format in LLM_Key.txt")
        except Exception as e:
            raise Exception(f"Error reading API key from LLM_Key.txt: {e}")
    else:
        raise FileNotFoundError(
            "LLM_Key.txt file not found in src/utils/ directory. "
            "Please create this file with your OpenAI API key for security."
        )

# OpenAI API Configuration - Only from secure file
try:
    OPENAI_API_KEY = get_api_key_from_file()
    print("✅ API key loaded successfully from LLM_Key.txt")
except Exception as e:
    print(f"❌ Failed to load API key: {e}")
    OPENAI_API_KEY = None

OPENAI_API_BASE = os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1')  # OpenAI API endpoint

# OpenAI specific configuration
OPENAI_MODEL = "gpt-4.1-mini"  # model
OPENAI_MAX_TOKENS = 10000
OPENAI_TEMPERATURE = 0.1

# Legacy compatibility (keeping for backward compatibility)
DEEPSEEK_API_KEY = OPENAI_API_KEY
DEEPSEEK_API_BASE = OPENAI_API_BASE
DEEPSEEK_MODEL = OPENAI_MODEL
DEEPSEEK_MAX_TOKENS = OPENAI_MAX_TOKENS
DEEPSEEK_TEMPERATURE = OPENAI_TEMPERATURE

LITELLM_API_KEY = OPENAI_API_KEY
LITELLM_API_BASE = OPENAI_API_BASE
LITELLM_MODEL = OPENAI_MODEL
LITELLM_MAX_TOKENS = OPENAI_MAX_TOKENS
LITELLM_TEMPERATURE = OPENAI_TEMPERATURE

# Search System Configuration
SEMANTIC_WEIGHT = float(os.getenv('SEMANTIC_WEIGHT', '0.7'))
METADATA_WEIGHT = float(os.getenv('METADATA_WEIGHT', '0.3'))
TOP_K_RESULTS = int(os.getenv('TOP_K_RESULTS', '10'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '100'))

# Model Configuration
EMBEDDING_MODEL = "text-embedding-3-small"  # Use OpenAI's latest embedding model
EMBEDDING_DIMENSIONS = 1536

# Data Paths
QUERIES_PATH = "prosusai_assignment_data/queries.csv"
ITEMS_PATH = "prosusai_assignment_data/5k_items_curated.csv"
CACHE_DIR = "embeddings_cache"

# Demo App Configuration
DEMO_MAX_RESULTS = int(os.getenv('DEMO_MAX_RESULTS', '20'))
DEMO_IMAGE_CACHE = os.getenv('DEMO_IMAGE_CACHE', 'True').lower() == 'true'

# Evaluation Configuration
EVALUATION_TOP_K = 5
RESULTS_DIR_PREFIX = "results_"

# Text Processing Configuration
PORTUGUESE_ACCENTS = "àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ"
MIN_TEXT_LENGTH = 10 