# Prosus AI Semantic Search System

A comprehensive semantic search and ranking system that combines traditional search methods with advanced LLM-based approaches for food item discovery and recommendation.

## 🚀 Features

- **Multiple Search Systems**: Traditional semantic search, Advanced LLM search, LLM Search System, and Hybrid Search System
- **Real-time LLM Integration**: OpenAI API integration with intelligent fallback and retry mechanisms
- **Hybrid Scoring**: Combines semantic similarity, metadata matching, and LLM reasoning
- **Interactive Demo**: Streamlit-based web interface for system comparison and testing
- **Batch Processing**: Efficient handling of multiple queries with concurrent processing
- **Comprehensive Evaluation**: Multi-metric assessment of search result quality

## 🏗️ Architecture

```
src/
├── demo/                          # Streamlit demo application
│   └── enhanced_demo_app.py      # Main demo interface
├── traditional_system/            # Traditional semantic search implementation
│   └── semantic_search.py        # Semantic + metadata search
├── llm_systems/                  # LLM-based search systems
│   └── advanced_llm_search.py    # Advanced LLM with API integration
├── utils/                        # Utility modules
│   ├── config.py                 # Configuration management
│   ├── mock_embeddings.py        # Mock embedding system
│   └── advanced_evaluation.py    # Evaluation metrics
└── prosusai_assignment_data/     # Data files
    ├── queries.csv               # Search queries
    └── 5k_items_curated.csv     # Food items dataset
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- pip package manager
- OpenAI API key (or compatible API)

### 1. Clone the Repository

```bash
git clone
cd Prosus_clean
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Setup

#### **API Key Configuration (Required)**

For security reasons, the OpenAI API key must be stored in a separate file:

1. **Create API Key File**: Create `LLM_Key.txt` in the `src/utils/` directory (same as `config.py`)
2. **Add Your Key**: Place your OpenAI API key in the file (one line, no quotes)
3. **File Format**:
   ```
   sk-your-actual-openai-api-key-here
   ```

**Example**:
```bash
# Create the key file in the correct directory
cd src/utils
echo "sk-your-openai-api-key" > LLM_Key.txt

# Or manually create src/utils/LLM_Key.txt with your key
```

#### **Environment Variables (Optional)**

Create a `.env` file for additional configuration:

```bash
# OpenAI Configuration (API key is read from LLM_Key.txt)
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_MODEL=gpt-4.1

# Search System Configuration
SEMANTIC_WEIGHT=0.7
METADATA_WEIGHT=0.3
TOP_K_RESULTS=10
```

**⚠️ Security Note**: Never commit `LLM_Key.txt` to version control. It's already added to `.gitignore` and will be located in `src/utils/` directory.

## 🚀 Usage

### 1. Launch the Demo Application

```bash
cd src/demo
python main.py
python -m streamlit run enhanced_demo_app.py
```

The application will open in your browser at `http://localhost:8501`.

### 2. Demo Interface Features

#### **Traditional Semantic Search Tab**
- **Sample Queries**: Test predefined food-related queries
- **Batch Search**: Process multiple queries simultaneously
- **Hybrid Search**: Combine semantic and metadata approaches
- **Method Analysis**: Compare different search strategies

#### **Advanced LLM Search Tab**
- **Real API Integration**: Uses OpenAI API for intelligent ranking
- **Timeout Retry**: Automatic retry mechanism for API failures
- **Hybrid Scoring**: Combines LLM reasoning with semantic similarity
- **Image Display**: Shows food item images with search results

#### **LLM System Tab**
- **Multiple Search Modes**: Direct LLM, Mock, and Hybrid approaches
- **Concurrent Processing**: Efficient batch query handling
- **Advanced Ranking**: Sophisticated candidate selection and re-ranking

#### **Hybrid System Tab**
- **Intelligent Routing**: Automatically selects optimal search method
- **Performance Optimization**: Query complexity caching and batch processing
- **Unified Interface**: Single entry point for all search methods

### 3. Configuration Options

#### **Search Parameters**
- **Top-K**: Number of results to return (1-20)
- **Search Type**: Choose between different search strategies
- **Batch Size**: Number of queries to process simultaneously

#### **System Settings**
- **API Configuration**: OpenAI model, temperature, max tokens
- **Scoring Weights**: Semantic vs. metadata importance
- **Retry Settings**: API timeout and retry intervals

## 📊 Data Format

### Queries CSV Structure
```csv
query,query_id
"Batatas fritas de rua carregadas",1
"Pizza de massa fina assada em forno a lenha",2
...
```

### Items CSV Structure
```csv
_id,itemMetadata,itemProfile
"item_1",{"name":"Food Name","category":"Category","description":"Description","price":"$10.99","images":["image1.jpg"]},{...}
...
```

## 🔧 API Configuration

### OpenAI API Setup

1. **Get API Key**: Sign up at [OpenAI Platform](https://platform.openai.com/)
2. **Configure**: Add your API key to `LLM_Key.txt` file (see Environment Setup)
3. **Model Selection**: Choose from available models (default: gpt-4.1-mini)

### Security Features

- **Secure Key Storage**: API keys are stored in separate `LLM_Key.txt` file in `src/utils/` directory
- **No Hardcoded Keys**: Configuration file never contains actual API keys
- **Git Protection**: `LLM_Key.txt` is automatically excluded from version control
- **Validation**: Basic API key format validation before use
- **Local Access**: Key file is co-located with configuration for easy management

### Alternative API Providers

The system supports other API providers through configuration:
- DeepSeek
- LiteLLM
- Custom endpoints

## 📈 Evaluation Metrics

### Search Quality Assessment
- **Overall Quality**: Combined score of all metrics
- **LLM Score**: Relevance assessment from language model
- **Semantic Score**: Vector similarity between query and items
- **Category Diversity**: Variety of food categories in results
- **Reasoning Quality**: Length and content of LLM explanations

### Performance Metrics
- **Response Time**: API call latency and processing speed
- **Throughput**: Queries processed per second
- **Success Rate**: Percentage of successful API calls
- **Cache Hit Rate**: Embedding and result caching efficiency

## 🚨 Troubleshooting

### Common Issues

#### **API Connection Errors**
```bash
# Check API key and base URL
echo $OPENAI_API_KEY
echo $OPENAI_API_BASE

# Verify network connectivity
curl -H "Authorization: Bearer $OPENAI_API_KEY" $OPENAI_API_BASE/models
```

#### **Import Errors**
```bash
# Ensure correct Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Check module structure
python -c "from src.llm_systems.advanced_llm_search import AdvancedLLMSearch"
```

#### **Data Loading Issues**
```bash
# Verify data file paths
ls -la prosusai_assignment_data/

# Check file permissions
chmod 644 prosusai_assignment_data/*.csv
```

### Performance Optimization

1. **Reduce API Calls**: Use batch processing for multiple queries
2. **Enable Caching**: Store embeddings and results locally
3. **Adjust Batch Size**: Optimize based on your API rate limits
4. **Use Mock System**: For development and testing without API costs

## 🔄 Development

### Adding New Search Systems

1. **Create System Class**: Inherit from base search system
2. **Implement Methods**: `search()`, `batch_search()`, `evaluate_results()`
3. **Add to Demo**: Integrate with Streamlit interface
4. **Update Tests**: Add evaluation and comparison metrics

### Extending Evaluation Metrics

1. **Define New Metrics**: Add calculation functions
2. **Update Evaluation**: Integrate with existing assessment framework
3. **Visualization**: Add charts and graphs to demo interface

## 🔒 Security

### API Key Protection

- **Never commit API keys** to version control
- **Use `LLM_Key.txt`** for storing sensitive credentials
- **Check `.gitignore`** to ensure key files are excluded
- **Rotate keys regularly** for production environments

### Best Practices

1. **Local Development**: Use separate API keys for development
2. **Production**: Use environment-specific keys with proper access controls
3. **Monitoring**: Monitor API usage and set rate limits
4. **Backup**: Keep secure backups of your API keys

