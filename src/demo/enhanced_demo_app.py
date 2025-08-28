#!/usr/bin/env python3
"""
Simplified Enhanced Demo Application for Prosus AI Search Systems
Focuses on Traditional vs LLM method comparison with batch results display
"""

import os
import json
import sys
from pathlib import Path

# More reliable path calculation for Streamlit environment
def get_project_root():
    """Get the project root directory reliably."""
    # Method 1: Try to find from current file location
    current_file = Path(__file__).resolve()
    demo_dir = current_file.parent
    src_dir = demo_dir.parent
    project_root = src_dir.parent
    
    # Method 2: If that doesn't work, try to find from current working directory
    if not project_root.exists():
        cwd = Path.cwd()
        # Look for prosusai_assignment_data directory
        for parent in [cwd] + list(cwd.parents):
            if (parent / "prosusai_assignment_data").exists():
                project_root = parent
                break
    
    # Method 3: Fallback to current working directory
    if not project_root.exists():
        project_root = Path.cwd()
    
    return project_root

# Get paths
project_root = get_project_root()
src_dir = project_root / "src"
current_dir = Path(__file__).parent

# Debug: Print actual paths
print(f"Current directory: {Path.cwd()}")
print(f"Project root: {project_root}")
print(f"Source directory: {src_dir}")
print(f"Current file: {Path(__file__).resolve()}")

# Add paths in the correct order
sys.path.insert(0, str(src_dir / "traditional_system"))
sys.path.insert(0, str(src_dir / "llm_systems"))
sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(project_root))

# Also try relative paths as fallback
try:
    sys.path.insert(0, "..")
    sys.path.insert(0, "../..")
except:
    pass

import streamlit as st
from PIL import Image
import requests
from io import BytesIO

# Import search systems with multiple fallback attempts
try:
    from traditional_system.semantic_search import SemanticSearchSystem
    print("✅ Traditional system imported successfully")
except ImportError as e:
    print(f"❌ First import attempt failed: {e}")
    try:
        from src.traditional_system.semantic_search import SemanticSearchSystem
        print("✅ Traditional system imported via src path")
    except ImportError as e:
        print(f"❌ Second import attempt failed: {e}")
        try:
            from semantic_search import SemanticSearchSystem
            print("✅ Traditional system imported directly")
        except ImportError as e:
            print(f"❌ All import attempts failed: {e}")
            SemanticSearchSystem = None

try:
    from llm_systems.advanced_llm_search import AdvancedLLMSearch
    print("✅ Advanced LLM system imported successfully")
except ImportError as e:
    print(f"❌ Advanced LLM import failed: {e}")
    try:
        from src.llm_systems.advanced_llm_search import AdvancedLLMSearch
        print("✅ Advanced LLM system imported via src path")
    except ImportError as e:
        print(f"❌ Advanced LLM import via src path failed: {e}")
        try:
            from advanced_llm_search import AdvancedLLMSearch
            print("✅ Advanced LLM system imported directly")
        except ImportError as e:
            print(f"❌ All Advanced LLM import attempts failed: {e}")
            AdvancedLLMSearch = None

def load_traditional_search_system():
    """Load the pre-trained traditional semantic search system."""
    try:
        if SemanticSearchSystem is None:
            st.error("❌ Traditional search system module not available!")
            st.error(f"Python path: {sys.path[:3]}")
            return None
            
        # Check if embeddings exist - use absolute paths
        embeddings_path = project_root / "embeddings_cache"
        queries_path = project_root / "prosusai_assignment_data" / "queries.csv"
        items_path = project_root / "prosusai_assignment_data" / "5k_items_curated.csv"
        
        # Debug: Print actual file paths
        print(f"Looking for embeddings at: {embeddings_path}")
        print(f"Looking for queries at: {queries_path}")
        print(f"Looking for items at: {items_path}")
        
        # Check if files exist and show detailed info
        # st.info(f"🔍 **Debug Info:**")
        # st.info(f"Project Root: `{project_root}`")
        # st.info(f"Current Working Directory: `{Path.cwd()}`")
        # st.info(f"Embeddings Path: `{embeddings_path}`")
        # st.info(f"Queries Path: `{queries_path}`")
        # st.info(f"Items Path: `{items_path}`")
        
        if not embeddings_path.exists():
            st.error(f"⚠️ Traditional search system requires embeddings. Path not found: {embeddings_path}")
            st.error("Please run the main.py script first!")
            return None
            
        if not queries_path.exists():
            st.error(f"⚠️ Queries file not found: {queries_path}")
            st.error(f"Please check if the file exists at: {queries_path}")
            return None
            
        if not items_path.exists():
            st.error(f"⚠️ Items file not found: {items_path}")
            st.error(f"Please check if the file exists at: {items_path}")
            return None
        
        search_system = SemanticSearchSystem()
        search_system.load_data(
            queries_path=str(queries_path),
            items_path=str(items_path)
        )
        
        # Load pre-computed embeddings
        search_system.load_embeddings(str(embeddings_path))
        return search_system
    except Exception as e:
        st.error(f"❌ Error loading traditional search system: {e}")
        st.error(f"Current working directory: {os.getcwd()}")
        st.error(f"Project root: {project_root}")
        st.error(f"Embeddings path: {embeddings_path}")
        return None

def load_advanced_llm_search_system():
    """Load the Advanced LLM-based search system."""
    try:
        if AdvancedLLMSearch is None:
            st.error("❌ Advanced LLM search system module not available!")
            return None
            
        # Use absolute paths
        queries_path = project_root / "prosusai_assignment_data" / "queries.csv"
        items_path = project_root / "prosusai_assignment_data" / "5k_items_curated.csv"
        
        # Debug: Print actual file paths
        print(f"Advanced LLM system - queries path: {queries_path}")
        print(f"Advanced LLM system - items path: {items_path}")
        
        if not queries_path.exists():
            st.error(f"⚠️ Queries file not found: {queries_path}")
            return None
            
        if not items_path.exists():
            st.error(f"⚠️ Items file not found: {items_path}")
            return None
            
        search_system = AdvancedLLMSearch()
        search_system.load_data(
            queries_path=str(queries_path),
            items_path=str(items_path)
        )
        return search_system
    except Exception as e:
        st.error(f"❌ Error loading Advanced LLM search system: {e}")
        return None

def display_item_card_traditional(item_data, rank):
    """Display an item in a card format for traditional search results."""
    with st.container():
        col1, col2 = st.columns([1, 3])
        
        with col1:
            if 'image_url' in item_data and item_data['image_url']:
                try:
                    response = requests.get(item_data['image_url'])
                    if response.status_code == 200:
                        img = Image.open(BytesIO(response.content))
                        st.image(img, width=100, caption=f"Rank {rank}")
                    else:
                        st.image("https://via.placeholder.com/100x100?text=No+Image", width=100)
                except:
                    st.image("https://via.placeholder.com/100x100?text=Error", width=100)
            else:
                st.image("https://via.placeholder.com/100x100?text=No+Image", width=100)
        
        with col2:
            st.markdown(f"**{item_data['name']}**")
            st.markdown(f"*{item_data['category']}*")
            if item_data['description'] != 'N/A':
                st.markdown(f"{item_data['description']}")
            
            col2a, col2b, col2c = st.columns(3)
            with col2a:
                st.metric("Price", f"R$ {item_data['price']}" if item_data['price'] != 'N/A' else "N/A")
            with col2b:
                st.metric("Semantic Score", f"{item_data['semantic_score']:.3f}")
            with col2c:
                st.metric("Combined Score", f"{item_data['combined_score']:.3f}")
            
            st.markdown("---")

def display_item_card_llm(item_data, rank):
    """Display an item in a card format for LLM search results."""
    with st.container():
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.image("https://via.placeholder.com/100x100?text=🍕", width=100, caption=f"Rank {rank}")
        
        with col2:
            st.markdown(f"**{item_data['name']}**")
            st.markdown(f"*{item_data['category']}*")
            if item_data['description']:
                st.markdown(f"{item_data['description']}")
            
            col2a, col2b, col2c = st.columns(3)
            with col2a:
                st.metric("Price", f"R$ {item_data['price']}" if item_data['price'] else "N/A")
            with col2b:
                st.metric("LLM Score", f"{item_data['llm_score']:.3f}")
            with col2c:
                st.metric("Rank", f"{item_data['rank']}")
            
            # Display LLM reasoning
            with st.expander("🤖 LLM Reasoning"):
                st.info(item_data['llm_reasoning'])
            
            st.markdown("---")

def display_advanced_llm_result(result, rank):
    """Display an item in a card format for Advanced LLM search results."""
    with st.container():
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Check if result has image_url attribute or is a dict with image_url
            if hasattr(result, 'image_url') and result.image_url:
                try:
                    response = requests.get(result.image_url)
                    if response.status_code == 200:
                        img = Image.open(BytesIO(response.content))
                        st.image(img, width=100, caption=f"Rank {rank}")
                    else:
                        st.image("https://via.placeholder.com/100x100?text=No+Image", width=100)
                except:
                    st.image("https://via.placeholder.com/100x100?text=No+Image", width=100)
            elif isinstance(result, dict) and 'image_url' in result and result['image_url']:
                try:
                    response = requests.get(result['image_url'])
                    if response.status_code == 200:
                        img = Image.open(BytesIO(response.content))
                        st.image(img, width=100, caption=f"Rank {rank}")
                    else:
                        st.image("https://via.placeholder.com/100x100?text=No+Image", width=100)
                except:
                    st.image("https://via.placeholder.com/100x100?text=No+Image", width=100)
            else:
                st.image("https://via.placeholder.com/100x100?text=🚀", width=100, caption=f"Rank {rank}")
        
        with col2:
            # Handle both object and dict formats
            if hasattr(result, 'name'):
                name = result.name
                category = result.category
                description = result.description
                price = result.price
                llm_score = result.llm_score
                combined_score = result.combined_score
            else:
                name = result.get('name', 'N/A')
                category = result.get('category', 'N/A')
                description = result.get('description', 'N/A')
                price = result.get('price', 'N/A')
                llm_score = result.get('llm_score', 0.0)
                combined_score = result.get('combined_score', 0.0)
            
            st.markdown(f"**{name}**")
            st.markdown(f"*{category}*")
            if description:
                st.markdown(f"{description}")
            
            col2a, col2b, col2c = st.columns(3)
            with col2a:
                st.metric("Price", f"R$ {price}" if price else "N/A")
            with col2b:
                st.metric("LLM Score", f"{llm_score:.3f}")
            with col2c:
                st.metric("Combined Score", f"{combined_score:.3f}")
            
            # Display LLM reasoning if available
            if hasattr(result, 'llm_reasoning') and result.llm_reasoning:
                with st.expander("🤖 Advanced LLM Reasoning"):
                    st.info(result.llm_reasoning)
            elif isinstance(result, dict) and 'llm_reasoning' in result and result['llm_reasoning']:
                with st.expander("🤖 Advanced LLM Reasoning"):
                    st.info(result['llm_reasoning'])
            
            st.markdown("---")

# def display_hybrid_result(result, rank):
#     """Display an item in a card format for Hybrid search results."""
#     with st.container():
#         col1, col2 = st.columns([1, 3])
        
#         with col1:
#             # Check if result has image_url attribute or is a dict with image_url
#             if hasattr(result, 'image_url') and result.image_url:
#                 try:
#                     response = requests.get(result.image_url)
#                     if response.status_code == 200:
#                         img = Image.open(BytesIO(response.content))
#                         st.image(img, width=100, caption=f"Rank {rank}")
#                     else:
#                         st.image("https://via.placeholder.com/100x100?text=No+Image", width=100)
#                 except:
#                     st.image("https://via.placeholder.com/100x100?text=No+Image", width=100)
#             elif isinstance(result, dict) and 'image_url' in result and result['image_url']:
#                 try:
#                     response = requests.get(result['image_url'])
#                     if response.status_code == 200:
#                         img = Image.open(BytesIO(response.content))
#                         st.image(img, width=100, caption=f"Rank {rank}")
#                     else:
#                         st.image("https://via.placeholder.com/100x100?text=No+Image", width=100)
#                 except:
#                     st.image("https://via.placeholder.com/100x100?text=No+Image", width=100)
#             else:
#                 st.image("https://via.placeholder.com/100x100?text=🔀", width=100, caption=f"Rank {rank}")
        
#         with col2:
#             # Handle both object and dict formats
#             if hasattr(result, 'name'):
#                 name = result.name
#                 category = result.category
#                 description = result.description
#                 price = result.price
#                 hybrid_score = result.hybrid_score
#                 traditional_score = result.traditional_score
#                 llm_score = result.llm_score
#                 method = result.method
#             else:
#                 name = result.get('name', 'N/A')
#                 category = result.get('category', 'N/A')
#                 description = result.get('description', 'N/A')
#                 price = result.get('price', 'N/A')
#                 hybrid_score = result.get('hybrid_score', 0.0)
#                 traditional_score = result.get('traditional_score', 0.0)
#                 llm_score = result.get('llm_score', 0.0)
#                 method = result.get('method', 'hybrid')
            
#             st.markdown(f"**{name}**")
#             st.markdown(f"*{category}*")
#             if description:
#                 st.markdown(f"{description}")
            
#             # Display method indicator
#             method_colors = {
#                 'traditional': '🔍',
#                 'llm': '🚀',
#                 'hybrid': '🔀'
#             }
#             st.markdown(f"**Method**: {method_colors.get(method, '🔀')} {method.upper()}")
            
#             col2a, col2b, col2c, col2d = st.columns(4)
#             with col2a:
#                 st.metric("Price", f"R$ {price}" if price else "N/A")
#             with col2b:
#                 st.metric("Hybrid Score", f"{hybrid_score:.3f}")
#             with col2c:
#                 st.metric("Traditional", f"{traditional_score:.3f}")
#             with col2d:
#                 st.metric("LLM Score", f"{llm_score:.3f}")
            
#             # Display execution time if available
#             if 'execution_time' in result:
#                 st.info(f"⏱️ Execution time: {result['execution_time']:.3f}s")
            
#             st.markdown("---")

def traditional_search_tab():
    """Traditional semantic search tab."""
    st.header("🔍 Traditional Semantic Search")
    st.markdown("**Vector-based semantic search using OpenAI embeddings**")
    
    # Load traditional search system
    traditional_system = load_traditional_search_system()
    if not traditional_system:
        st.error("❌ Traditional search system failed to load!")
        st.warning("Please ensure embeddings are generated by running the main.py script first.")
        return
    
    st.success("✅ Traditional search system loaded successfully!")
    
    # Search options
    col1, col2 = st.columns([2, 1])
    with col1:
        search_type = st.selectbox(
            "Search Type",
            ["Sample Queries", "Batch Search"],
            key="traditional_search_type"
        )
    
    with col2:
        top_k = st.slider("Number of Results", 1, 20, 5, key="traditional_top_k")
    
    if search_type == "Sample Queries":
        st.subheader("📋 Sample Queries")
        st.success("✅ These queries work with the pre-computed embeddings!")
        
        sample_queries = [
            "Batatas fritas de rua carregadas",
            "Pizza de massa fina assada em forno a lenha",
            "Sopa de macarrão feita à mão",
            "Almoço estilo havaiano",
            "Sanduíche de café da manhã com abacate"
        ]
        
        selected_query = st.selectbox("Choose a sample query:", sample_queries, key="traditional_sample_query")
        
        if st.button("Search Sample Query", type="primary", key="traditional_sample_search_btn"):
            with st.spinner("Searching with traditional system..."):
                try:
                    # Find the query index
                    query_idx = None
                    for i, query in enumerate(traditional_system.queries):
                        if query['search_term_pt'] == selected_query:
                            query_idx = i
                            break
                    
                    if query_idx is not None:
                        results = traditional_system.search(query_idx, top_k)
                        
                        st.subheader(f"Traditional Search Results for: '{selected_query}'")
                        st.markdown(f"Found {len(results)} results")
                        
                        for i, result in enumerate(results):
                            display_item_card_traditional(result, i + 1)
                    else:
                        st.error("Query not found in dataset.")
                        
                except Exception as e:
                    st.error(f"Error during traditional search: {e}")
    
    elif search_type == "Batch Search":
        st.subheader("📊 Batch Search Results")
        st.success("✅ Batch evaluation works with all pre-computed queries!")
        
        if st.button("Run Traditional Batch Evaluation", type="primary", key="traditional_batch_btn"):
            with st.spinner("Running traditional batch evaluation..."):
                try:
                    evaluation_results = traditional_system.run_evaluation(top_k=top_k)
                    
                    # Display overall metrics
                    st.subheader("Traditional System Performance Metrics")
                    metrics = evaluation_results['overall_metrics']
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Queries", evaluation_results['total_queries'])
                    with col2:
                        st.metric("Avg Semantic Score", f"{metrics['avg_semantic_score']:.4f}")
                    with col3:
                        st.metric("Avg Combined Score", f"{metrics['avg_combined_score']:.4f}")
                    with col4:
                        st.metric("Avg Diversity Score", f"{metrics['avg_diversity_score']:.4f}")
                    
                    # Show all query results
                    st.subheader("All Query Results")
                    
                    for i, query_result in enumerate(evaluation_results['query_results']):
                        with st.expander(f"Query {i+1}: {query_result['query']}"):
                            st.markdown(f"**Query**: {query_result['query']}")
                            st.markdown(f"**Results**: {len(query_result['results'])}")
                            
                            # Show results based on user's top_k selection
                            for j, result in enumerate(query_result['results'][:top_k]):
                                display_item_card_traditional(result, j + 1)
                    
                    # Download results
                    st.subheader("Download Results")
                    results_json = json.dumps(evaluation_results, ensure_ascii=False, indent=2)
                    st.download_button(
                        label="Download Traditional Results (JSON)",
                        data=results_json,
                        file_name="traditional_batch_results.json",
                        mime="application/json"
                    )
                    
                except Exception as e:
                    st.error(f"Error during traditional batch evaluation: {e}")

def advanced_llm_search_tab():
    """Advanced LLM-based search tab."""
    st.header("🚀 Advanced LLM-Based Search")
    st.markdown("**Real API integration with intelligent candidate selection system**")
    
    # Load Advanced LLM search system
    llm_system = load_advanced_llm_search_system()
    if not llm_system:
        st.warning("Advanced LLM search system not available.")
        return
    
    # Search options
    col1, col2 = st.columns([2, 1])
    with col1:
        search_type = st.selectbox(
            "Search Type",
            ["Sample Queries", "Batch Search"],
            key="advanced_llm_search_type"
        )
    
    with col2:
        top_k = st.slider("Number of Results", 1, 20, 5, key="advanced_llm_top_k")
    
    if search_type == "Sample Queries":
        st.subheader("Sample Queries")
        
        sample_queries = [
            "Batatas fritas de rua carregadas",
            "Pizza de massa fina assada em forno a lenha",
            "Sopa de macarrão feita à mão",
            "Almoço estilo havaiano",
            "Sanduíche de café da manhã com abacate"
        ]
        
        selected_query = st.selectbox("Choose a sample query:", sample_queries, key="advanced_llm_sample_query")
        
        if st.button("Search Sample Query", type="primary", key="advanced_llm_sample_search_btn"):
            with st.spinner("Searching with Advanced LLM system..."):
                try:
                    results = llm_system.search(selected_query, top_k)
                    
                    st.subheader(f"Advanced LLM Search Results for: '{selected_query}'")
                    st.markdown(f"Found {len(results)} results")
                    
                    for i, result in enumerate(results):
                        display_advanced_llm_result(result, i + 1)
                        
                except Exception as e:
                    st.error(f"Error during Advanced LLM search: {e}")
    
    elif search_type == "Batch Search":
        st.subheader("Batch Search Results")
        
        if st.button("Run Advanced LLM Batch Evaluation", type="primary", key="advanced_llm_batch_btn"):
            with st.spinner("Running Advanced LLM batch evaluation..."):
                try:
                    # Get all queries from the system
                    all_queries = [query['search_term_pt'] for query in llm_system.queries]
                    
                    batch_results = llm_system.batch_search(all_queries, top_k)
                    
                    # Calculate overall metrics
                    total_queries = len(batch_results)
                    all_scores = []
                    all_categories = set()
                    
                    for query, results in batch_results.items():
                        for result in results:
                            # Handle both object and dict formats
                            if hasattr(result, 'llm_score'):
                                all_scores.append(result.llm_score)
                                all_categories.add(result.category)
                            else:
                                all_scores.append(result.get('llm_score', 0.0))
                                all_categories.add(result.get('category', 'Unknown'))
                    
                    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
                    category_diversity = len(all_categories) / total_queries if total_queries > 0 else 0
                    
                    # Display overall metrics
                    st.subheader("Advanced LLM System Performance Metrics")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Queries", total_queries)
                    with col2:
                        st.metric("Avg LLM Score", f"{avg_score:.4f}")
                    with col3:
                        st.metric("Category Diversity", f"{category_diversity:.4f}")
                    with col4:
                        st.metric("Total Results", sum(len(results) for results in batch_results.values()))
                    
                    # Show all query results
                    st.subheader("All Query Results")
                    
                    for i, (query, results) in enumerate(batch_results.items()):
                        with st.expander(f"Query {i+1}: {query}"):
                            st.markdown(f"**Query**: {query}")
                            st.markdown(f"**Results**: {len(results)}")
                            
                            # Show results based on user's top_k selection
                            for j, result in enumerate(results[:top_k]):
                                display_advanced_llm_result(result, j + 1)
                    
                    # Download results
                    st.subheader("Download Results")
                    results_json = json.dumps(batch_results, ensure_ascii=False, indent=2)
                    st.download_button(
                        label="Download Advanced LLM Results (JSON)",
                        data=results_json,
                        file_name="advanced_llm_batch_results.json",
                        mime="application/json"
                    )
                    
                except Exception as e:
                    st.error(f"Error during Advanced LLM batch evaluation: {e}")

# def comparison_tab():
#     """Comparison tab between traditional and Advanced LLM systems."""
#     st.header("⚖️ System Comparison")
#     st.markdown("**Compare results between Traditional Semantic Search and Advanced LLM-Based Search**")
    
#     # Load both systems
#     traditional_system = load_traditional_search_system()
#     llm_system = load_advanced_llm_search_system()
    
#     if not traditional_system or not llm_system:
#         st.warning("Both search systems must be available for comparison.")
#         return
    
#     st.info("💡 **Tip**: Use the individual tabs to see detailed results for each system.")
    
#     # Run comparison on sample queries
#     if st.button("Run System Comparison", type="primary", key="comparison_btn"):
#         with st.spinner("Comparing both systems..."):
#             try:
#                 # Sample queries for comparison
#                 sample_queries = [
#                     "Batatas fritas de rua carregadas",
#                     "Pizza de massa fina assada em forno a lenha",
#                     "Sopa de macarrão feita à mão"
#                 ]
                
#                 comparison_results = {}
                
#                 for query in sample_queries:
#                     comparison_results[query] = {
#                         'traditional': None,
#                         'llm': None
#                     }
                    
#                     # Get traditional results
#                     try:
#                         query_idx = None
#                         for i, q in enumerate(traditional_system.queries):
#                             if q['search_term_pt'] == query:
#                                 query_idx = i
#                                 break
                        
#                         if query_idx is not None:
#                             traditional_results = traditional_system.search(query_idx, 5)
#                             comparison_results[query]['traditional'] = traditional_results
#                     except Exception as e:
#                         st.warning(f"Traditional search failed for '{query}': {e}")
                    
#                     # Get Advanced LLM results
#                     try:
#                         llm_results = llm_system.search(query, 5)
#                         comparison_results[query]['llm'] = llm_results
#                     except Exception as e:
#                         st.warning(f"Advanced LLM search failed for '{query}': {e}")
                
#                 # Display comparison results
#                 st.subheader("📊 Comparison Results")
                
#                 for query, results in comparison_results.items():
#                     with st.expander(f"Query: {query}"):
#                         col1, col2 = st.columns(2)
                        
#                         with col1:
#                             st.subheader("🔍 Traditional Results")
#                             if results['traditional']:
#                                 st.success(f"Found {len(results['traditional'])} results")
#                                 for i, result in enumerate(results['traditional'][:3]):
#                                     st.markdown(f"**{i+1}.** {result['name']} (Score: {result['combined_score']:.3f})")
#                             else:
#                                 st.warning("No traditional results available")
                        
#                         with col2:
#                             st.subheader("🚀 Advanced LLM Results")
#                             if results['llm']:
#                                 st.success(f"Found {len(results['llm'])} results")
#                                 for i, result in enumerate(results['llm'][:3]):
#                                     # Handle both object and dict formats
#                                     if hasattr(result, 'name'):
#                                         name = result.name
#                                         score = result.llm_score
#                                     else:
#                                         name = result.get('name', 'N/A')
#                                         score = result.get('llm_score', 0.0)
#                                     st.markdown(f"**{i+1}.** {name} (Score: {score:.3f})")
#                             else:
#                                 st.warning("No Advanced LLM results available")
                
#                 # Performance summary
#                 st.subheader("🎯 Performance Summary")
                
#                 traditional_scores = []
#                 llm_scores = []
                
#                 for query, results in comparison_results.items():
#                     if results['traditional']:
#                         traditional_scores.extend([r['combined_score'] for r in results['traditional']])
#                     if results['llm']:
#                         for result in results['llm']:
#                             if hasattr(result, 'llm_score'):
#                                 llm_scores.append(result.llm_score)
#                             else:
#                                 llm_scores.append(result.get('llm_score', 0.0))
                
#                 col1, col2 = st.columns(2)
#                 with col1:
#                     if traditional_scores:
#                         avg_traditional = sum(traditional_scores) / len(traditional_scores)
#                         st.metric("Traditional Avg Score", f"{avg_traditional:.3f}")
#                     else:
#                         st.metric("Traditional Avg Score", "N/A")
                
#                 with col2:
#                     if llm_scores:
#                         avg_llm = sum(llm_scores) / len(llm_scores)
#                         st.metric("Advanced LLM Avg Score", f"{avg_llm:.3f}")
#                     else:
#                         st.metric("Advanced LLM Avg Score", "N/A")
                
#             except Exception as e:
#                 st.error(f"Error during comparison: {e}")

def main():
    st.set_page_config(
        page_title="Prosus AI Food Search Demo",
        page_icon="🍕",
        layout="wide"
    )
    
    st.title("🍕 Prosus AI Food Search Demo")
    st.markdown("### Interactive Demo: Multiple Search Systems Comparison")
    
    # Sidebar
    st.sidebar.header("System Status")
    
    # Check system availability
    traditional_available = (project_root / "embeddings_cache").exists()
    advanced_llm_available = AdvancedLLMSearch is not None
    # llm_search_system_available = LLMSearchSystem is not None
    # hybrid_search_system_available = HybridSearchSystem is not None
    
    # System status display
    st.sidebar.subheader("🔧 System Status")
    
    col1, col2 = st.columns(2)
    with col1:
        if traditional_available:
            st.sidebar.success("✅ Traditional")
        else:
            st.sidebar.error("❌ Traditional")
    
    with col2:
        if advanced_llm_available:
            st.sidebar.success("✅ Advanced LLM")
        else:
            st.sidebar.error("❌ Advanced LLM")
    
    # col3, col4 = st.columns(2)
    # with col3:
    #     if llm_search_system_available:
    #         st.sidebar.success("✅ LLM System")
    #     else:
    #         st.sidebar.error("❌ LLM System")
    
    # with col4:
    #     if hybrid_search_system_available:
    #         st.sidebar.success("✅ Hybrid System")
    #     else:
    #         st.sidebar.error("❌ Hybrid System")
    
    if not traditional_available:
        st.sidebar.warning("⚠️ Run main.py first to generate embeddings for traditional search")
    
    # System info
    st.sidebar.subheader("🧠 System Info")
    st.sidebar.info("""
    **Traditional**: Vector-based semantic search using OpenAI embeddings
    **Advanced LLM**: Real API integration with intelligent candidate selection system
    """)
    
    # Main content with tabs
    tab1, tab2= st.tabs([
        "🔍 Traditional Search", 
        "🚀 Advanced LLM",
        # "🧠 LLM System",
        # "🔀 Hybrid System"
    ])
    
    with tab1:
        traditional_search_tab()
    
    with tab2:
        advanced_llm_search_tab()
    
    # with tab3:
    #     llm_search_system_tab()
    
    # with tab4:
    #     hybrid_search_system_tab()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Prosus AI Technical Assignment** | "
        "Enhanced Search System Demo | "
        "Multiple Search Systems Comparison"
    )

if __name__ == "__main__":
    main()
