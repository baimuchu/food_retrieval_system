#!/usr/bin/env python3
"""
Advanced LLM-Based Search and Ranking System
Uses real LLM API calls with timeout retry mechanism
"""

import json
import csv
import numpy as np
from typing import List, Dict, Optional
from tqdm import tqdm
import time
import openai
from dataclasses import dataclass
import requests

@dataclass
class LLMSearchResult:
    """Structured search result with LLM reasoning."""
    item_id: str
    name: str
    category: str
    description: str
    price: Optional[str]
    llm_score: float
    llm_reasoning: str
    semantic_score: float
    combined_score: float
    rank: int
    image_url: Optional[str] = None  # Add image_url field

class AdvancedLLMSearch:
    """Advanced LLM-based search with real API integration and timeout retry."""
    
    def __init__(self, api_key: str = None, api_base: str = None):
        """Initialize with API credentials."""
        self.items = []
        self.queries = []
        self.client = None
        self.max_retries = 3
        self.retry_delay = 30  # 30 seconds wait between retries
        
        # Initialize with OpenAI API configuration
        try:
            from src.utils.config import OPENAI_API_KEY, OPENAI_API_BASE, OPENAI_MODEL
            
            self.api_key = api_key or OPENAI_API_KEY
            self.api_base = api_base or OPENAI_API_BASE
            self.llm_model = OPENAI_MODEL
            
            # Initialize OpenAI client
            self.client = openai.OpenAI(
                api_key=self.api_key, 
                base_url=self.api_base
            )
            
            # Test the connection with a simple request
            test_response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            
            print("✅ OpenAI API initialized successfully!")
            print(f"   API Key: {self.api_key[:10]}...{self.api_key[-4:]}")
            print(f"   API Base: {self.api_base}")
            print(f"   Model: {self.llm_model}")
            
        except Exception as e:
            print(f"❌ OpenAI API initialization failed: {e}")
            raise Exception("OpenAI API initialization failed. Please check your API credentials and network connection.")
    
    def load_data(self, queries_path: str, items_path: str):
        """Load data from CSV files."""
        print("Loading data...")
        
        # Load queries
        with open(queries_path, 'r', encoding='utf-8') as f:
            self.queries = list(csv.DictReader(f))
        print(f"Loaded {len(self.queries)} queries")
        
        # Load items
        with open(items_path, 'r', encoding='utf-8') as f:
            self.items = list(csv.DictReader(f))
        print(f"Loaded {len(self.items)} items")
    
    def extract_item_info(self, item: Dict) -> Optional[Dict]:
        """Extract and parse item information."""
        try:
            metadata = json.loads(item['itemMetadata'])
            profile = json.loads(item['itemProfile'])
            
            return {
                'id': item['_id'],
                'name': metadata.get('name', ''),
                'category': metadata.get('category_name', ''),
                'description': metadata.get('description', ''),
                'price': metadata.get('price', ''),
                'taxonomy': metadata.get('taxonomy', ''),
                'tags': metadata.get('tags', []),
                'full_text': f"{metadata.get('name', '')} {metadata.get('description', '')} {metadata.get('category_name', '')} {' '.join(metadata.get('tags', []))}"
            }
        except Exception as e:
            return None
    
    def real_llm_ranking(self, query: str, candidates: List[Dict], top_k: int = 5) -> List[LLMSearchResult]:
        """Use real LLM API for intelligent ranking with timeout retry mechanism."""
        if not self.client:
            raise Exception("OpenAI client not initialized")
        
        # Prepare candidate information for LLM
        candidates_text = []
        for i, candidate in enumerate(candidates[:10]):  # Limit to top 10 for API efficiency
            info = f"{i+1}. {candidate['name']} | Category: {candidate['category']} | Description: {candidate['description'][:150]}"
            candidates_text.append(info)
        
        candidates_text = "\n".join(candidates_text)
        
        # Create LLM prompt
        prompt = f"""
        You are a food search expert. Given this Portuguese food query and food items, rank them by relevance.
        
        Query: "{query}"
        
        Available Items:
        {candidates_text}
        
        For each item, provide:
        1. A relevance score from 0.0 to 1.0 (1.0 = most relevant)
        2. A brief reasoning in English explaining why it's relevant
        
        Return in this exact JSON format (NO markdown formatting, NO code blocks):
        {{
            "rankings": [
                {{
                    "item_index": 1,
                    "score": 0.95,
                    "reasoning": "Perfect match for the search query - exactly what was requested"
                }}
            ]
        }}
        
        IMPORTANT: Return ONLY the raw JSON object, no markdown, no code blocks, no additional text.
        """
        
        # Retry mechanism for API calls
        for attempt in range(self.max_retries):
            try:
                # Call LLM API
                from src.utils.config import OPENAI_MAX_TOKENS, OPENAI_TEMPERATURE
                
                response = self.client.chat.completions.create(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=OPENAI_MAX_TOKENS,
                    temperature=OPENAI_TEMPERATURE
                )
                
                # Parse response
                llm_response = response.choices[0].message.content.strip()
                
                # Clean response: remove markdown code block markers if present
                if llm_response.startswith('```json'):
                    llm_response = llm_response.replace('```json', '').replace('```', '').strip()
                elif llm_response.startswith('```'):
                    llm_response = llm_response.replace('```', '').strip()
                
                rankings = json.loads(llm_response)
                
                # Convert to search results
                results = []
                for ranking in rankings['rankings']:
                    idx = ranking['item_index'] - 1
                    if 0 <= idx < len(candidates):
                        candidate = candidates[idx]
                        # Extract image URL from original item data
                        image_url = None
                        try:
                            original_item = next((item for item in self.items if item['_id'] == candidate['id']), None)
                            if original_item:
                                metadata = json.loads(original_item['itemMetadata'])
                                if 'images' in metadata and metadata['images']:
                                    image_url = f"https://static.ifood-static.com.br/image/upload/t_low/pratos/{metadata['images'][0]}"
                        except Exception:
                            pass
                        
                        results.append(LLMSearchResult(
                            item_id=candidate['id'],
                            name=candidate['name'],
                            category=candidate['category'],
                            description=candidate['description'],
                            price=candidate.get('price', ''),
                            llm_score=ranking['score'],
                            llm_reasoning=ranking['reasoning'],
                            semantic_score=0.0,  # Will be calculated separately
                            combined_score=ranking['score'],
                            rank=len(results) + 1,
                            image_url=image_url
                        ))
                
                return results[:top_k]
                
            except (openai.APITimeoutError, openai.APIConnectionError, requests.exceptions.Timeout) as e:
                print(f"API timeout/connection error (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    print(f"Waiting {self.retry_delay} seconds before retry...")
                    time.sleep(self.retry_delay)
                else:
                    raise Exception(f"API request failed after {self.max_retries} attempts: {e}")
                    
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                print(f"Raw response: {llm_response}")
                raise Exception(f"Invalid JSON response from LLM: {e}")
                
            except Exception as e:
                print(f"Unexpected error: {e}")
                raise Exception(f"LLM ranking failed: {e}")
    
    def search(self, query: str, top_k: int = 5) -> List[LLMSearchResult]:
        """Perform LLM-based search using only real API."""
        print(f"Searching for: '{query}'")
        
        # Get candidates (3 * top_k for better LLM ranking)
        candidates = self.get_candidates(query, top_k * 3)
        
        # Rank with real LLM API
        results = self.real_llm_ranking(query, candidates, top_k)
        
        # Add semantic scores and calculate combined scores
        try:
            from ..utils.mock_embeddings import mock_embedding_system
            query_embedding = mock_embedding_system.text_to_embedding(query)
            
            for result in results:
                candidate = next((c for c in candidates if c['id'] == result.item_id), None)
                if candidate:
                    item_embedding = mock_embedding_system.text_to_embedding(candidate['full_text'])
                    semantic_score = mock_embedding_system.cosine_similarity(query_embedding, item_embedding)
                    result.semantic_score = semantic_score
                    # Calculate combined score: 70% LLM + 30% Semantic
                    result.combined_score = (result.llm_score * 0.7) + (semantic_score * 0.3)
        except ImportError:
            # If embeddings not available, use candidate selection scores as semantic scores
            print("Embeddings not available, using candidate selection scores as semantic scores")
            for result in results:
                candidate = next((c for c in candidates if c['id'] == result.item_id), None)
                if candidate:
                    # Use the candidate's combined score from get_candidates as semantic score
                    # This ensures we still have a meaningful combination
                    semantic_score = candidate.get('combined_score', 0.0)
                    result.semantic_score = semantic_score
                    # Calculate combined score: 70% LLM + 30% Candidate Score
                    result.combined_score = (result.llm_score * 0.7) + (semantic_score * 0.3)
        
        # Final ranking by combined score
        results.sort(key=lambda x: x.combined_score, reverse=True)
        for i, result in enumerate(results):
            result.rank = i + 1
        
        return results[:top_k]
    
    def get_candidates(self, query: str, limit: int = 15) -> List[Dict]:
        """Get candidate items for ranking using the same approach as traditional_system."""
        candidates = []
        query_lower = query.lower()
        
        # Calculate scores for all items (similar to traditional_system)
        scores = []
        for item in self.items:
            parsed_item = self.extract_item_info(item)
            if not parsed_item:
                continue
            
            # Calculate metadata score (similar to traditional_system)
            metadata_score = self.calculate_metadata_score(query_lower, parsed_item)
            
            # Calculate semantic score if embeddings available
            semantic_score = 0.0
            try:
                from ..utils.mock_embeddings import mock_embedding_system
                query_embedding = mock_embedding_system.text_to_embedding(query)
                item_embedding = mock_embedding_system.text_to_embedding(parsed_item['full_text'])
                semantic_score = mock_embedding_system.cosine_similarity(query_embedding, item_embedding)
            except ImportError:
                # If embeddings not available, use metadata score only
                semantic_score = metadata_score
            
            # Combine scores using traditional_system weights
            combined_score = (0.7 * semantic_score + 0.3 * metadata_score)
            
            scores.append({
                'item': parsed_item,
                'semantic_score': semantic_score,
                'metadata_score': metadata_score,
                'combined_score': combined_score
            })
        
        # Sort by combined score (descending)
        scores.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Return top 3 * top_k candidates with score information
        for score_info in scores[:limit]:
            candidate = score_info['item'].copy()
            # Add score information to candidate for later use
            candidate['semantic_score'] = score_info['semantic_score']
            candidate['metadata_score'] = score_info['metadata_score']
            candidate['combined_score'] = score_info['combined_score']
            candidates.append(candidate)
        
        return candidates
    
    def calculate_metadata_score(self, query_text: str, item: Dict) -> float:
        """Calculate metadata-based relevance score (same as traditional_system)."""
        try:
            query_words = set(query_text.lower().split())
            score = 0.0
            
            # Name matching (weight: 0.3)
            if 'name' in item:
                item_name = item['name'].lower()
                name_words = set(item_name.split())
                name_overlap = len(query_words.intersection(name_words))
                score += name_overlap * 0.3
            
            # Category matching (weight: 0.2)
            if 'category' in item:
                category = item['category'].lower()
                category_words = set(category.split())
                category_overlap = len(query_words.intersection(category_words))
                score += category_overlap * 0.2
            
            # Description matching (weight: 0.1)
            if 'description' in item:
                description = item['description'].lower()
                desc_words = set(description.split())
                desc_overlap = len(query_words.intersection(desc_words))
                score += desc_overlap * 0.1
            
            # Normalize score
            return min(score, 1.0)
            
        except Exception:
            return 0.0
    
    def batch_search(self, queries: List[str], top_k: int = 5) -> Dict[str, List[LLMSearchResult]]:
        """Search multiple queries with retry mechanism."""
        results = {}
        
        print(f"Processing {len(queries)} queries with real LLM API...")
        
        for query in tqdm(queries, desc="Searching"):
            try:
                query_results = self.search(query, top_k)
                results[query] = query_results
                
                # Rate limiting between requests
                time.sleep(0.2)
                    
            except Exception as e:
                print(f"Error processing query '{query}': {e}")
                results[query] = []
        
        return results
    
    def evaluate_results(self, query: str, results: List[LLMSearchResult]) -> Dict:
        """Evaluate search result quality."""
        if not results:
            return {'query': query, 'quality_score': 0.0}
        
        # Calculate metrics
        avg_llm_score = np.mean([r.llm_score for r in results])
        avg_semantic_score = np.mean([r.semantic_score for r in results])
        avg_combined_score = np.mean([r.combined_score for r in results])
        category_diversity = len(set(r.category for r in results)) / len(results)
        
        # Reasoning quality (length and content)
        reasoning_quality = np.mean([
            len(r.llm_reasoning.split()) / 8.0  # Normalize by expected length
            for r in results
        ])
        reasoning_quality = min(reasoning_quality, 1.0)
        
        overall_quality = (avg_combined_score + category_diversity + reasoning_quality) / 3
        
        return {
            'query': query,
            'num_results': len(results),
            'avg_llm_score': avg_llm_score,
            'avg_semantic_score': avg_semantic_score,
            'avg_combined_score': avg_combined_score,
            'category_diversity': category_diversity,
            'reasoning_quality': reasoning_quality,
            'overall_quality': overall_quality
        }
    
    def save_results(self, results: Dict[str, List[LLMSearchResult]], filename: str):
        """Save results to JSON."""
        # Convert to serializable format
        serializable_results = {}
        for query, query_results in results.items():
                            serializable_results[query] = [
                    {
                        'item_id': r.item_id,
                        'name': r.name,
                        'category': r.category,
                        'description': r.description,
                        'price': r.price,
                        'llm_score': r.llm_score,
                        'llm_reasoning': r.llm_reasoning,
                        'semantic_score': r.semantic_score,
                        'combined_score': r.combined_score,
                        'rank': r.rank,
                        'image_url': r.image_url
                    }
                    for r in query_results
                ]
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        print(f"Results saved to {filename}")

def main():
    """Main function to demonstrate advanced LLM search."""
    print("Advanced LLM-Based Search System")
    print("=" * 50)
    
    # Initialize system
    search_system = AdvancedLLMSearch()
    
    # Load data
    search_system.load_data(
        "../../prosusai_assignment_data/queries.csv",
        "../../prosusai_assignment_data/5k_items_curated.csv"
    )
    
    # Test queries
    test_queries = [
        "Batatas fritas de rua carregadas",
        "Pizza de massa fina assada em forno a lenha",
        "Sopa de macarrão feita à mão",
        "Sanduíche de frango grelhado",
        "Arroz com feijão tradicional"
    ]
    
    # Perform search
    results = search_system.batch_search(test_queries, top_k=5)
    
    # Evaluate and display results
    print("\n" + "="*50)
    print("SEARCH RESULTS EVALUATION")
    print("="*50)
    
    total_quality = 0.0
    
    for query, query_results in results.items():
        evaluation = search_system.evaluate_results(query, query_results)
        total_quality += evaluation['overall_quality']
        
        print(f"\nQuery: {query}")
        print(f"   Quality Score: {evaluation['overall_quality']:.3f}")
        print(f"   LLM Score: {evaluation['avg_llm_score']:.3f}")
        print(f"   Combined Score: {evaluation['avg_combined_score']:.3f}")
        print(f"   Category Diversity: {evaluation['category_diversity']:.3f}")
        
        print(f"   Top Results:")
        for result in query_results[:3]:
            print(f"     {result.rank}. {result.name}")
            print(f"        Category: {result.category}")
            print(f"        LLM Score: {result.llm_score:.3f}")
            print(f"        Reasoning: {result.llm_reasoning}")
    
    # Overall performance
    avg_quality = total_quality / len(test_queries)
    print(f"\nOverall System Performance: {avg_quality:.3f}")
    print(f"LLM Mode: Real API with retry mechanism")
    
    # Save results
    search_system.save_results(results, "advanced_llm_results.json")

if __name__ == "__main__":
    main() 