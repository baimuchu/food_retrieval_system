import json
import csv
import numpy as np
import openai
from typing import List, Dict, Tuple, Any
import re
from tqdm import tqdm
import os
from dotenv import load_dotenv
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
import config

# Load environment variables
load_dotenv()

class SemanticSearchSystem:
    def __init__(self, api_key: str = None, api_base: str = None):
        """
        Initialize the semantic search system.
        
        Args:
            api_key: OpenAI API key for embeddings (optional, uses config if not provided)
            api_base: API base URL for OpenAI API (optional, uses config if not provided)
        """
        # Use provided values or fall back to config
        self.api_key = api_key or config.OPENAI_API_KEY
        self.api_base = api_base or config.OPENAI_API_BASE
        
        # Initialize OpenAI client with OpenAI configuration
        openai.api_key = self.api_key
        openai.api_base = self.api_base
        
        # Data storage
        self.queries = []
        self.items = []
        self.query_embeddings = None
        self.item_embeddings = None
        self.item_texts = []
        
        # Search parameters
        self.top_k = 10
        self.semantic_weight = 0.7
        self.metadata_weight = 0.3
        
    def load_data(self, queries_path: str, items_path: str):
        """Load and preprocess the queries and items data."""
        print("Loading queries...")
        with open(queries_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            self.queries = list(reader)
        
        print("Loading items...")
        with open(items_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            self.items = list(reader)
        
        print(f"Loaded {len(self.queries)} queries and {len(self.items)} items")
        
    def preprocess_text(self, text: str) -> str:
        """Clean and normalize Portuguese text."""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep Portuguese accents
        text = re.sub(r'[^\w\sàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_item_text(self, item: Dict) -> str:
        """Extract and combine relevant text fields from an item."""
        try:
            metadata = json.loads(item['itemMetadata'])
            
            # Combine relevant fields
            text_parts = []
            
            if 'name' in metadata:
                text_parts.append(metadata['name'])
            
            if 'description' in metadata:
                text_parts.append(metadata['description'])
            
            if 'category_name' in metadata:
                text_parts.append(metadata['category_name'])
            
            if 'taxonomy' in metadata:
                taxonomy = metadata['taxonomy']
                for level in ['l0', 'l1', 'l2']:
                    if level in taxonomy:
                        text_parts.append(taxonomy[level])
            
            if 'tags' in metadata:
                for tag in metadata['tags']:
                    if 'value' in tag:
                        text_parts.extend(tag['value'])
            
            # Combine all text
            combined_text = ' '.join(filter(None, text_parts))
            return self.preprocess_text(combined_text)
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error processing item {item.get('itemId', 'unknown')}: {e}")
            return ""
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """Generate embeddings for a list of texts using OpenAI API or fallback to mock embeddings."""
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch = texts[i:i + batch_size]
            
            try:
                response = openai.embeddings.create(
                    model=config.EMBEDDING_MODEL,
                    input=batch
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)
                
            except Exception as e:
                print(f"API error for batch {i//batch_size}: {e}")
                print("Falling back to mock embeddings for demonstration...")
                
                # Use mock embeddings as fallback
                try:
                    from ..utils.mock_embeddings import mock_embedding_system
                    batch_embeddings = mock_embedding_system.generate_embeddings(batch)
                    embeddings.extend(batch_embeddings)
                except ImportError:
                    # Final fallback: use zero vectors
                    batch_embeddings = [[0.0] * config.EMBEDDING_DIMENSIONS] * len(batch)
                    embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)
    
    def prepare_data(self):
        """Prepare data for semantic search by generating embeddings."""
        print("Preparing item texts...")
        self.item_texts = [self.extract_item_text(item) for item in self.items]
        
        # Filter out empty texts
        valid_indices = [i for i, text in enumerate(self.item_texts) if text.strip()]
        self.items = [self.items[i] for i in valid_indices]
        self.item_texts = [self.item_texts[i] for i in valid_indices]
        
        print(f"Valid items after filtering: {len(self.items)}")
        
        # Generate embeddings for items
        print("Generating item embeddings...")
        self.item_embeddings = self.generate_embeddings(self.item_texts)
        
        # Generate embeddings for queries
        print("Generating query embeddings...")
        query_texts = [self.preprocess_text(query['search_term_pt']) for query in self.queries]
        self.query_embeddings = self.generate_embeddings(query_texts)
        
        print("Data preparation complete!")
    
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def calculate_semantic_similarity(self, query_idx: int, item_idx: int) -> float:
        """Calculate semantic similarity between a query and an item."""
        query_emb = self.query_embeddings[query_idx]
        item_emb = self.item_embeddings[item_idx]
        
        similarity = self.cosine_similarity(query_emb, item_emb)
        return float(similarity)
    
    def calculate_metadata_score(self, query_text: str, item: Dict) -> float:
        """Calculate metadata-based relevance score."""
        try:
            metadata = json.loads(item['itemMetadata'])
            query_words = set(query_text.lower().split())
            
            score = 0.0
            
            # Name matching
            if 'name' in metadata:
                item_name = metadata['name'].lower()
                name_words = set(item_name.split())
                name_overlap = len(query_words.intersection(name_words))
                score += name_overlap * 0.3
            
            # Category matching
            if 'category_name' in metadata:
                category = metadata['category_name'].lower()
                category_words = set(category.split())
                category_overlap = len(query_words.intersection(category_words))
                score += category_overlap * 0.2
            
            # Taxonomy matching
            if 'taxonomy' in metadata:
                taxonomy = metadata['taxonomy']
                for level in ['l0', 'l1', 'l2']:
                    if level in taxonomy:
                        tax_text = taxonomy[level].lower()
                        tax_words = set(tax_text.split())
                        tax_overlap = len(query_words.intersection(tax_words))
                        score += tax_overlap * 0.1
            
            # Normalize score
            return min(score, 1.0)
            
        except (json.JSONDecodeError, KeyError):
            return 0.0
    
    def search(self, query_idx: int, top_k: int = None) -> List[Dict]:
        """Perform semantic search for a given query."""
        if top_k is None:
            top_k = self.top_k
        
        query_text = self.preprocess_text(self.queries[query_idx]['search_term_pt'])
        
        # Calculate scores for all items
        scores = []
        for item_idx in range(len(self.items)):
            semantic_score = self.calculate_semantic_similarity(query_idx, item_idx)
            metadata_score = self.calculate_metadata_score(query_text, self.items[item_idx])
            
            # Combine scores
            combined_score = (self.semantic_weight * semantic_score + 
                            self.metadata_weight * metadata_score)
            
            scores.append({
                'item_idx': item_idx,
                'semantic_score': semantic_score,
                'metadata_score': metadata_score,
                'combined_score': combined_score
            })
        
        # Sort by combined score
        scores.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Return top results
        results = []
        for i, score_info in enumerate(scores[:top_k]):
            item = self.items[score_info['item_idx']]
            metadata = json.loads(item['itemMetadata'])
            
            result = {
                'rank': i + 1,
                'itemId': item['itemId'],
                'name': metadata.get('name', 'N/A'),
                'category': metadata.get('category_name', 'N/A'),
                'description': metadata.get('description', 'N/A'),
                'price': metadata.get('price', 'N/A'),
                'semantic_score': score_info['semantic_score'],
                'metadata_score': score_info['metadata_score'],
                'combined_score': score_info['combined_score'],
                'merchantId': item['merchantId']
            }
            
            # Add image URL if available
            if 'images' in metadata and metadata['images']:
                result['image_url'] = f"https://static.ifood-static.com.br/image/upload/t_low/pratos/{metadata['images'][0]}"
            
            results.append(result)
        
        return results
    
    def evaluate_search(self, query_idx: int, top_k: int = 5) -> Dict:
        """Evaluate search results for a query."""
        results = self.search(query_idx, top_k)
        
        # Calculate evaluation metrics
        avg_semantic_score = np.mean([r['semantic_score'] for r in results])
        avg_metadata_score = np.mean([r['metadata_score'] for r in results])
        avg_combined_score = np.mean([r['combined_score'] for r in results])
        
        # Diversity score (category variety)
        categories = [r['category'] for r in results]
        unique_categories = len(set(categories))
        diversity_score = unique_categories / len(categories) if categories else 0
        
        return {
            'query': self.queries[query_idx]['search_term_pt'],
            'results': results,
            'metrics': {
                'avg_semantic_score': avg_semantic_score,
                'avg_metadata_score': avg_metadata_score,
                'avg_combined_score': avg_combined_score,
                'avg_diversity_score': diversity_score  # Fixed key name
            }
        }
    
    def save_embeddings(self, cache_dir: str):
        """Save embeddings to cache directory."""
        import pickle
        import os
        
        os.makedirs(cache_dir, exist_ok=True)
        
        # Save item embeddings
        with open(f"{cache_dir}/item_embeddings.pkl", 'wb') as f:
            pickle.dump(self.item_embeddings, f)
        
        # Save query embeddings
        with open(f"{cache_dir}/query_embeddings.pkl", 'wb') as f:
            pickle.dump(self.query_embeddings, f)
        
        # Save item texts
        with open(f"{cache_dir}/item_texts.pkl", 'wb') as f:
            pickle.dump(self.item_texts, f)
        
        print(f"Embeddings saved to {cache_dir}")
    
    def load_embeddings(self, cache_dir: str):
        """Load embeddings from cache directory."""
        import pickle
        
        try:
            # Load item embeddings
            with open(f"{cache_dir}/item_embeddings.pkl", 'rb') as f:
                self.item_embeddings = pickle.load(f)
            
            # Load query embeddings
            with open(f"{cache_dir}/query_embeddings.pkl", 'rb') as f:
                self.query_embeddings = pickle.load(f)
            
            # Load item texts
            with open(f"{cache_dir}/item_texts.pkl", 'rb') as f:
                self.item_texts = pickle.load(f)
            
            print(f"Embeddings loaded from {cache_dir}")
            
        except FileNotFoundError:
            print(f"Cache directory {cache_dir} not found. Please run prepare_data() first.")
            raise
    
    def run_evaluation(self, top_k: int = 5) -> Dict:
        """Run evaluation on all queries."""
        print(f"Running evaluation on {len(self.queries)} queries...")
        
        all_results = []
        total_metrics = {
            'avg_semantic_score': 0,
            'avg_metadata_score': 0,
            'avg_combined_score': 0,
            'avg_diversity_score': 0
        }
        
        for query_idx in tqdm(range(len(self.queries)), desc="Evaluating queries"):
            eval_result = self.evaluate_search(query_idx, top_k)
            all_results.append(eval_result)
            
            # Accumulate metrics - handle missing keys gracefully
            for key in total_metrics:
                if key in eval_result['metrics']:
                    total_metrics[key] += eval_result['metrics'][key]
                else:
                    print(f"Warning: Missing metric '{key}' in query {query_idx}")
        
        # Calculate averages
        num_queries = len(self.queries)
        for key in total_metrics:
            total_metrics[key] /= num_queries
        
        return {
            'query_results': all_results,
            'overall_metrics': total_metrics,
            'total_queries': num_queries
        } 