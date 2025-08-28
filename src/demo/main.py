#!/usr/bin/env python3
"""
Main execution script for the Prosus AI Semantic Search System.
"""

import os
import json
import numpy as np
import sys
from datetime import datetime
from pathlib import Path

# Add parent directories to path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent  # Go up to workspace root
src_dir = current_dir.parent

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(src_dir / "traditional_system"))

from traditional_system.semantic_search import SemanticSearchSystem

def create_results_directory():
    """Create results directory with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def save_results(results_dir: str, evaluation_results: dict):
    """Save evaluation results to files."""
    # Save detailed results
    with open(f"{results_dir}/detailed_results.json", 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
    
    # Save summary
    summary = {
        'total_queries': evaluation_results['total_queries'],
        'overall_metrics': evaluation_results['overall_metrics'],
        'timestamp': datetime.now().isoformat()
    }
    
    with open(f"{results_dir}/summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # Save top results for each query
    top_results = []
    for query_result in evaluation_results['query_results']:
        query_info = {
            'query': query_result['query'],
            'top_5_results': query_result['results'][:5]
        }
        top_results.append(query_info)
    
    with open(f"{results_dir}/top_5_results.json", 'w', encoding='utf-8') as f:
        json.dump(top_results, f, ensure_ascii=False, indent=2)

def create_visualizations(results_dir: str, evaluation_results: dict):
    """Create simple text-based visualizations for the results."""
    # Create a simple metrics summary file
    metrics = evaluation_results['overall_metrics']
    
    with open(f"{results_dir}/metrics_summary.txt", 'w', encoding='utf-8') as f:
        f.write("PROSUS AI SEMANTIC SEARCH - PERFORMANCE METRICS\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("OVERALL METRICS:\n")
        f.write("-" * 20 + "\n")
        for metric_name, metric_value in metrics.items():
            f.write(f"{metric_name}: {metric_value:.4f}\n")
        
        f.write(f"\nTOTAL QUERIES: {evaluation_results['total_queries']}\n")
        
        # Score distribution summary
        f.write("\nSCORE DISTRIBUTION SUMMARY:\n")
        f.write("-" * 30 + "\n")
        
        all_semantic_scores = []
        all_combined_scores = []
        
        for query_result in evaluation_results['query_results']:
            all_semantic_scores.extend([r['semantic_score'] for r in query_result['results']])
            all_combined_scores.extend([r['combined_score'] for r in query_result['results']])
        
        f.write(f"Semantic Scores - Min: {min(all_semantic_scores):.4f}, Max: {max(all_semantic_scores):.4f}, Mean: {np.mean(all_semantic_scores):.4f}\n")
        f.write(f"Combined Scores - Min: {min(all_combined_scores):.4f}, Max: {max(all_combined_scores):.4f}, Mean: {np.mean(all_combined_scores):.4f}\n")
    
    print(f"Metrics summary saved to {results_dir}/metrics_summary.txt")

def print_summary(evaluation_results: dict):
    """Print a summary of the evaluation results."""
    print("\n" + "="*60)
    print("SEMANTIC SEARCH EVALUATION SUMMARY")
    print("="*60)
    
    metrics = evaluation_results['overall_metrics']
    print(f"Total Queries Evaluated: {evaluation_results['total_queries']}")
    print(f"Average Semantic Score: {metrics['avg_semantic_score']:.4f}")
    print(f"Average Metadata Score: {metrics['avg_metadata_score']:.4f}")
    print(f"Average Combined Score: {metrics['avg_combined_score']:.4f}")
    print(f"Average Diversity Score: {metrics['avg_diversity_score']:.4f}")
    
    print("\n" + "="*60)
    print("SAMPLE QUERY RESULTS")
    print("="*60)
    
    # Show first 3 queries as examples
    for i, query_result in enumerate(evaluation_results['query_results'][:3]):
        print(f"\nQuery {i+1}: {query_result['query']}")
        print("-" * 50)
        for j, result in enumerate(query_result['results'][:3]):
            print(f"  {j+1}. {result['name']}")
            print(f"     Category: {result['category']}")
            print(f"     Combined Score: {result['combined_score']:.4f}")
            print(f"     Semantic Score: {result['semantic_score']:.4f}")

def main():
    """Main execution function."""
    print("Prosus AI Semantic Search System")
    print("="*50)
    
    # Initialize the search system with LiteLLM configuration
    print("Initializing semantic search system...")
    search_system = SemanticSearchSystem()
    
    # Load data
    print("Loading data...")
    search_system.load_data(
        queries_path=str(project_root / 'prosusai_assignment_data' / 'queries.csv'),
        items_path=str(project_root / 'prosusai_assignment_data' / '5k_items_curated.csv')
    )
    
    # Prepare data (generate embeddings)
    print("Preparing data and generating embeddings...")
    search_system.prepare_data()
    
    # Save embeddings for demo app
    print("Saving embeddings to cache...")
    search_system.save_embeddings(str(project_root / 'embeddings_cache'))
    
    # Run evaluation
    print("Running evaluation...")
    evaluation_results = search_system.run_evaluation(top_k=5)
    
    # Create results directory
    results_dir = create_results_directory()
    
    # Save results
    print(f"Saving results to {results_dir}...")
    save_results(results_dir, evaluation_results)
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(results_dir, evaluation_results)
    
    # Print summary
    print_summary(evaluation_results)
    
    print(f"\nResults saved to: {results_dir}")
    print("Files created:")
    print(f"  - {results_dir}/detailed_results.json")
    print(f"  - {results_dir}/summary.json")
    print(f"  - {results_dir}/top_5_results.json")
    print(f"  - {results_dir}/metrics_summary.txt")

if __name__ == "__main__":
    main() 