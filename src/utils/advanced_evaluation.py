"""
Advanced Evaluation Module for Prosus AI Semantic Search
Addresses Todo items: evaluation metrics, method comparison, and improvements
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

class AdvancedEvaluator:
    """
    Advanced evaluation system that addresses the three Todo items:
    1. Design evaluation metrics without ground truth
    2. Evaluate differences between methods and show trade-offs
    3. Provide improvement suggestions
    """
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        self.traditional_results = None
        self.llm_results = None
        self.comparison_metrics = {}
        
    def load_results(self, traditional_file: str, llm_file: str):
        """Load results from both traditional and LLM systems"""
        try:
            with open(traditional_file, 'r', encoding='utf-8') as f:
                self.traditional_results = json.load(f)
            print(f"✅ Loaded traditional results: {len(self.traditional_results)} queries")
        except Exception as e:
            print(f"❌ Error loading traditional results: {e}")
            
        try:
            with open(llm_file, 'r', encoding='utf-8') as f:
                self.llm_results = json.load(f)
            print(f"✅ Loaded LLM results: {len(self.llm_results)} queries")
        except Exception as e:
            print(f"❌ Error loading LLM results: {e}")
    
    def calculate_ground_truth_free_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """
        Todo Item 1: Design evaluation metrics without ground truth
        Implements sophisticated evaluation using multiple proxy metrics
        """
        metrics = {}
        
        # 1. Semantic Coherence Score
        semantic_scores = []
        for query_data in results:
            if 'results' in query_data and len(query_data['results']) > 0:
                # Calculate semantic similarity between query and top results
                query_text = query_data['query'].lower()
                result_texts = [r.get('name', '') + ' ' + r.get('description', '') 
                              for r in query_data['results'][:3]]
                
                # TF-IDF based similarity as proxy for semantic coherence
                try:
                    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
                    vectors = vectorizer.fit_transform([query_text] + result_texts)
                    similarities = cosine_similarity(vectors[0:1], vectors[1:])
                    semantic_scores.append(np.mean(similarities))
                except:
                    semantic_scores.append(0.0)
        
        metrics['semantic_coherence'] = np.mean(semantic_scores) if semantic_scores else 0.0
        
        # 2. Result Diversity Score
        diversity_scores = []
        for query_data in results:
            if 'results' in query_data and len(query_data['results']) > 0:
                categories = [r.get('category', 'Unknown') for r in query_data['results'][:5]]
                unique_categories = len(set(categories))
                diversity_scores.append(unique_categories / min(len(categories), 5))
        
        metrics['result_diversity'] = np.mean(diversity_scores) if diversity_scores else 0.0
        
        # 3. Query-Result Consistency Score
        consistency_scores = []
        for query_data in results:
            if 'results' in query_data and len(query_data['results']) > 0:
                query_words = set(query_data['query'].lower().split())
                consistency_score = 0
                for result in query_data['results'][:3]:
                    result_text = (result.get('name', '') + ' ' + 
                                 result.get('description', '')).lower()
                    result_words = set(result_text.split())
                    word_overlap = len(query_words.intersection(result_words))
                    consistency_score += word_overlap / max(len(query_words), 1)
                consistency_scores.append(consistency_score / 3)
        
        metrics['query_result_consistency'] = np.mean(consistency_scores) if consistency_scores else 0.0
        
        # 4. Ranking Stability Score
        stability_scores = []
        for query_data in results:
            if 'results' in query_data and len(query_data['results']) >= 2:
                scores = [r.get('score', 0) for r in query_data['results'][:5]]
                if len(scores) > 1:
                    # Calculate how well scores differentiate between ranks
                    score_diffs = [abs(scores[i] - scores[i+1]) for i in range(len(scores)-1)]
                    stability_scores.append(np.mean(score_diffs))
        
        metrics['ranking_stability'] = np.mean(stability_scores) if stability_scores else 0.0
        
        # 5. Cultural Relevance Score (Portuguese food context)
        cultural_scores = []
        portuguese_food_terms = {
            'batatas', 'pizza', 'sopa', 'feijoada', 'churrasco', 'moqueca',
            'acarajé', 'vatapá', 'caruru', 'bobó', 'tacacá', 'tucupi'
        }
        
        for query_data in results:
            if 'results' in query_data and len(query_data['results']) > 0:
                query_lower = query_data['query'].lower()
                cultural_terms = sum(1 for term in portuguese_food_terms if term in query_lower)
                cultural_scores.append(min(cultural_terms / 3, 1.0))  # Normalize to 0-1
        
        metrics['cultural_relevance'] = np.mean(cultural_scores) if cultural_scores else 0.0
        
        # 6. Overall Quality Score (weighted combination)
        weights = {
            'semantic_coherence': 0.25,
            'result_diversity': 0.20,
            'query_result_consistency': 0.25,
            'ranking_stability': 0.15,
            'cultural_relevance': 0.15
        }
        
        overall_score = sum(metrics[key] * weights[key] for key in weights.keys())
        metrics['overall_quality'] = overall_score
        
        return metrics
    
    def compare_methods(self) -> Dict[str, Any]:
        """
        Todo Item 2: Evaluate differences between methods and show trade-offs
        Comprehensive comparison between traditional and LLM approaches
        """
        if not self.traditional_results or not self.llm_results:
            print("❌ Both traditional and LLM results must be loaded for comparison")
            return {}
        
        comparison = {
            'method_comparison': {},
            'trade_offs': {},
            'performance_analysis': {},
            'recommendations': {}
        }
        
        # Calculate metrics for both methods
        traditional_metrics = self.calculate_ground_truth_free_metrics(self.traditional_results)
        llm_metrics = self.calculate_ground_truth_free_metrics(self.llm_results)
        
        # Method comparison
        comparison['method_comparison'] = {
            'traditional': traditional_metrics,
            'llm': llm_metrics
        }
        
        # Performance analysis
        comparison['performance_analysis'] = {
            'semantic_coherence_improvement': 
                (llm_metrics['semantic_coherence'] - traditional_metrics['semantic_coherence']) / 
                max(traditional_metrics['semantic_coherence'], 0.001) * 100,
            'diversity_improvement': 
                (llm_metrics['result_diversity'] - traditional_metrics['result_diversity']) / 
                max(traditional_metrics['result_diversity'], 0.001) * 100,
            'overall_quality_improvement': 
                (llm_metrics['overall_quality'] - traditional_metrics['overall_quality']) / 
                max(traditional_metrics['overall_quality'], 0.001) * 100
        }
        
        # Trade-offs analysis
        comparison['trade_offs'] = {
            'traditional_advantages': [
                "Faster execution time",
                "No API dependencies",
                "Deterministic results",
                "Lower computational cost",
                "Easier to debug and maintain"
            ],
            'traditional_disadvantages': [
                "Limited semantic understanding",
                "Rigid scoring algorithms",
                "No reasoning capabilities",
                "Poor handling of complex queries",
                "Limited cultural context awareness"
            ],
            'llm_advantages': [
                "Superior semantic understanding",
                "Explainable results with reasoning",
                "Cultural and contextual awareness",
                "Adaptive scoring algorithms",
                "Better handling of complex queries"
            ],
            'llm_disadvantages': [
                "API dependency and costs",
                "Slower execution time",
                "Non-deterministic results",
                "Higher computational complexity",
                "Requires prompt engineering expertise"
            ]
        }
        
        # Recommendations
        comparison['recommendations'] = {
            'use_traditional_when': [
                "Real-time performance is critical",
                "Budget constraints limit API usage",
                "Deterministic results are required",
                "Simple keyword-based queries",
                "System resources are limited"
            ],
            'use_llm_when': [
                "Search quality is paramount",
                "Complex, nuanced queries",
                "Explainable AI is required",
                "Cultural context matters",
                "Budget allows for API usage"
            ],
            'hybrid_approach': [
                "Combine both methods for optimal results",
                "Use traditional for initial filtering, LLM for ranking",
                "Implement fallback mechanisms",
                "Cache LLM responses for common queries"
            ]
        }
        
        self.comparison_metrics = comparison
        return comparison
    
    def generate_improvement_suggestions(self) -> Dict[str, List[str]]:
        """
        Todo Item 3: Give improvements for current methods
        Provides specific, actionable improvement recommendations
        """
        improvements = {
            'traditional_system_improvements': [
                "Implement query expansion using Portuguese food synonyms",
                "Add fuzzy matching for Portuguese accent variations",
                "Create food category hierarchies for better grouping",
                "Implement user feedback learning mechanisms",
                "Add seasonal and regional food preferences",
                "Optimize embedding dimensions for food domain",
                "Implement caching for frequently searched items",
                "Add multi-language support beyond Portuguese"
            ],
            'llm_system_improvements': [
                "Implement query expansion using LLM reasoning",
                "Add few-shot learning with Portuguese food examples",
                "Create specialized food domain prompts",
                "Implement result explanation generation",
                "Add user preference learning from interactions",
                "Optimize prompt engineering for food queries",
                "Implement batch processing for efficiency",
                "Add confidence scoring for results"
            ],
            'hybrid_system_improvements': [
                "Create adaptive switching between methods",
                "Implement ensemble scoring from both approaches",
                "Add real-time method selection based on query complexity",
                "Create unified evaluation framework",
                "Implement A/B testing capabilities",
                "Add performance monitoring and alerting",
                "Create automated prompt optimization",
                "Implement cost-benefit analysis for method selection"
            ],
            'evaluation_improvements': [
                "Implement user satisfaction surveys",
                "Add click-through rate tracking",
                "Create relevance feedback mechanisms",
                "Implement A/B testing framework",
                "Add real-time performance monitoring",
                "Create automated quality assessment",
                "Implement continuous learning from user behavior",
                "Add multi-dimensional quality metrics"
            ]
        }
        
        return improvements
    
    def generate_comprehensive_report(self) -> str:
        """Generate a comprehensive evaluation report"""
        if not self.comparison_metrics:
            self.compare_methods()
        
        improvements = self.generate_improvement_suggestions()
        
        report = f"""
# 🎯 ADVANCED EVALUATION REPORT - PROSUS AI SEMANTIC SEARCH
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 METHOD COMPARISON RESULTS

### Traditional System Performance
{json.dumps(self.comparison_metrics['method_comparison']['traditional'], indent=2)}

### LLM System Performance  
{json.dumps(self.comparison_metrics['method_comparison']['llm'], indent=2)}

### Performance Improvements
- Semantic Coherence: {self.comparison_metrics['performance_analysis']['semantic_coherence_improvement']:.1f}%
- Result Diversity: {self.comparison_metrics['performance_analysis']['diversity_improvement']:.1f}%
- Overall Quality: {self.comparison_metrics['performance_analysis']['overall_quality_improvement']:.1f}%

## ⚖️ TRADE-OFFS ANALYSIS

### Traditional System
**Advantages:**
{chr(10).join(f"- {adv}" for adv in self.comparison_metrics['trade_offs']['traditional_advantages'])}

**Disadvantages:**
{chr(10).join(f"- {dis}" for dis in self.comparison_metrics['trade_offs']['traditional_disadvantages'])}

### LLM System
**Advantages:**
{chr(10).join(f"- {adv}" for adv in self.comparison_metrics['trade_offs']['llm_advantages'])}

**Disadvantages:**
{chr(10).join(f"- {dis}" for dis in self.comparison_metrics['trade_offs']['llm_disadvantages'])}

## 🚀 IMPROVEMENT RECOMMENDATIONS

### Traditional System Improvements
{chr(10).join(f"{i+1}. {imp}" for i, imp in enumerate(improvements['traditional_system_improvements']))}

### LLM System Improvements
{chr(10).join(f"{i+1}. {imp}" for i, imp in enumerate(improvements['llm_system_improvements']))}

### Hybrid System Improvements
{chr(10).join(f"{i+1}. {imp}" for i, imp in enumerate(improvements['hybrid_system_improvements']))}

### Evaluation Improvements
{chr(10).join(f"{i+1}. {imp}" for i, imp in enumerate(improvements['evaluation_improvements']))}

## 🎯 RECOMMENDED APPROACH

Based on the analysis, the recommended approach is:

**Hybrid Implementation:**
- Use traditional system for initial filtering and real-time queries
- Apply LLM system for complex queries and result ranking
- Implement intelligent switching based on query complexity
- Add comprehensive evaluation and monitoring

## 📈 NEXT STEPS

1. Implement hybrid system architecture
2. Add real-time performance monitoring
3. Create user feedback collection system
4. Implement A/B testing framework
5. Optimize prompts for Portuguese food domain
6. Add cost-benefit analysis dashboard
"""
        
        return report
    
    def save_report(self, filename: str = None):
        """Save the comprehensive report to file"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"advanced_evaluation_report_{timestamp}.md"
        
        report = self.generate_comprehensive_report()
        
        # Ensure results directory exists
        os.makedirs(self.results_dir, exist_ok=True)
        
        filepath = os.path.join(self.results_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✅ Advanced evaluation report saved: {filepath}")
        return filepath
    
    def create_visualizations(self, save_dir: str = None):
        """Create visualizations for the comparison analysis"""
        if not save_dir:
            save_dir = self.results_dir
        
        if not self.comparison_metrics:
            self.compare_methods()
        
        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Performance Comparison Chart
        plt.figure(figsize=(12, 8))
        
        metrics = ['semantic_coherence', 'result_diversity', 'query_result_consistency', 
                  'ranking_stability', 'cultural_relevance', 'overall_quality']
        
        traditional_scores = [self.comparison_metrics['method_comparison']['traditional'][m] for m in metrics]
        llm_scores = [self.comparison_metrics['method_comparison']['llm'][m] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.bar(x - width/2, traditional_scores, width, label='Traditional System', alpha=0.8)
        plt.bar(x + width/2, llm_scores, width, label='LLM System', alpha=0.8)
        
        plt.xlabel('Evaluation Metrics')
        plt.ylabel('Scores')
        plt.title('Performance Comparison: Traditional vs LLM Systems')
        plt.xticks(x, [m.replace('_', ' ').title() for m in metrics], rotation=45)
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(os.path.join(save_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Improvement Analysis Chart
        plt.figure(figsize=(10, 6))
        
        improvements = ['Semantic\nCoherence', 'Result\nDiversity', 'Overall\nQuality']
        improvement_values = [
            self.comparison_metrics['performance_analysis']['semantic_coherence_improvement'],
            self.comparison_metrics['performance_analysis']['diversity_improvement'],
            self.comparison_metrics['performance_analysis']['overall_quality_improvement']
        ]
        
        colors = ['green' if x > 0 else 'red' for x in improvement_values]
        bars = plt.bar(improvements, improvement_values, color=colors, alpha=0.7)
        
        plt.xlabel('Metrics')
        plt.ylabel('Improvement (%)')
        plt.title('LLM System Improvements Over Traditional System')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, improvement_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (1 if value > 0 else -1),
                    f'{value:.1f}%', ha='center', va='bottom' if value > 0 else 'top')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'improvement_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Visualizations saved to: {save_dir}")
        return save_dir

