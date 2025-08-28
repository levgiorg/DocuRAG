# DocuRAG - Production RAG System

A Retrieval-Augmented Generation (RAG) system built with Ollama, FAISS, and LangChain. DocuRAG processes documents, performs semantic search, and provides context-aware question answering through a Streamlit web interface.


## ✨ Features

### Core Capabilities
- **Multi-format Document Processing**: PDF, TXT, DOCX, and Markdown support
- **Intelligent Chunking**: Configurable chunk sizes with smart overlap handling
- **Hybrid Retrieval**: Combines semantic search with keyword matching
- **Source Attribution**: Shows which documents and pages were used for answers
- **Conversation History**: Maintains context across multiple questions
- **Quality Scoring**: Automatically scores chunk quality for better retrieval

### Advanced Features
- **Query Rewriting**: Expands and rephrases queries for better retrieval
- **Metadata Filtering**: Filter results by document properties
- **Response Confidence Scoring**: Provides confidence metrics for answers
- **Performance Monitoring**: Tracks processing times and system metrics
- **Persistent Storage**: FAISS-based vector storage with automatic persistence

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   RAG Engine     │    │   Ollama LLM    │
│   Interface     │◄──►│                  │◄──►│  (gemma3n:e2b) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                       ┌────────┼────────┐
                       ▼                 ▼
               ┌──────────────┐  ┌──────────────┐
               │   Document   │  │    FAISS     │
               │  Processor   │  │ Vector Store │
               └──────────────┘  └──────────────┘
                       │                 │
                       ▼                 ▼
               ┌──────────────┐  ┌──────────────┐
               │ PDF/DOCX/TXT │  │  Embeddings  │
               │   Sources    │  │ (MiniLM-L6)  │
               └──────────────┘  └──────────────┘
```

## 🚀 Quick Start

### Prerequisites

1. **Python 3.9+** installed on your system
2. **Ollama** installed and running locally
3. **gemma3n:e2b** model pulled in Ollama

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd DocuRAG
```

2. **Install Python dependencies:**
```bash
pip install -r requirements.txt

# If you encounter import issues, also install in conda environment:
/opt/homebrew/anaconda3/bin/pip install -r requirements.txt
```

3. **Set up Ollama:**
```bash
# Install Ollama (if not already installed)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the required model
ollama pull gemma3n:e2b

# Start Ollama (if not already running)
ollama serve
```

### Running the Application

1. **Test the system (recommended first step):**
```bash
python test_system.py
```

2. **Start the web interface:**
```bash
python start_app.py
```

3. **Open your browser** to `http://localhost:8501`

4. **Upload documents** in the "Document Management" tab or place PDFs in `data/documents/`

5. **Start asking questions** in the "Chat" tab!

## 📖 Usage Guide

### Document Management

1. **Upload Files**: Use the file uploader to add PDF, TXT, DOCX, or MD files
2. **Process Directory**: Point to a local directory containing documents
3. **View Statistics**: Monitor processed documents and chunks in the sidebar

### Asking Questions

1. Navigate to the **Chat** tab
2. Type your question in the chat input
3. View the response with **source attributions**
4. Check **confidence scores** and **response times**
5. Explore **source documents** in the expandable sections

### Configuration

Edit files in the `configs/` directory to customize:

- **Model Settings** (`model_config.yaml`): LLM model, embedding model, API settings
- **Retrieval Settings** (`retrieval_config.yaml`): Chunk size, similarity thresholds
- **App Settings** (`app_config.yaml`): File size limits, conversation history

## 🔧 Configuration Options

### Model Configuration (`configs/model_config.yaml`)
```yaml
llm_model: "gemma3n:e2b"           # Ollama model name
ollama_base_url: "http://localhost:11434"  # Ollama API URL
embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
temperature: 0.7                   # Generation temperature
max_tokens: 2048                   # Maximum response tokens
```

### Retrieval Configuration (`configs/retrieval_config.yaml`)
```yaml
chunk_size: 1000                   # Document chunk size
chunk_overlap: 200                 # Overlap between chunks
top_k: 5                          # Number of chunks to retrieve
similarity_threshold: 0.7          # Minimum similarity for results
```

### Application Configuration (`configs/app_config.yaml`)
```yaml
max_file_size_mb: 50              # Maximum upload file size
allowed_extensions: [".pdf", ".txt", ".docx", ".md"]
enable_conversation_history: true  # Enable chat history
max_conversation_turns: 10        # Maximum history length
```

## 📊 System Components

### Document Processor (`src/document_processor.py`)
- Multi-format document parsing (PDF, DOCX, TXT, MD)
- Intelligent text chunking with configurable overlap
- Metadata extraction (title, author, creation date, page count)
- Quality scoring for chunks
- Text cleaning and normalization

### Vector Store (`src/vector_store.py`)
- FAISS-based vector storage with persistence
- Multiple index types (Flat, IVF, HNSW)
- Hybrid search (semantic + keyword)
- Metadata filtering capabilities
- Efficient batch processing

### RAG Engine (`src/rag_engine.py`)
- Query processing and rewriting
- Retrieval orchestration
- LLM integration with Ollama
- Response generation with source attribution
- Conversation history management
- Performance monitoring

### Web Interface (`app.py`)
- Streamlit-based user interface
- Document upload and management
- Real-time chat interface
- System monitoring and statistics
- Configuration management

## 📁 Project Structure

```
DocuRAG/
├── src/
│   ├── document_processor.py     # Document parsing and chunking
│   ├── vector_store.py          # FAISS vector operations
│   ├── rag_engine.py            # Main RAG orchestration
│   └── utils.py                 # Shared utilities
├── configs/
│   ├── app_config.yaml          # Application settings
│   ├── model_config.yaml        # Model configurations
│   └── retrieval_config.yaml    # Retrieval parameters
├── data/
│   ├── documents/               # Uploaded documents
│   ├── vector_store/           # FAISS indices
│   └── logs/                   # Application logs
├── tests/                      # Test files
├── app.py                      # Streamlit interface
├── config.py                   # Configuration management
├── requirements.txt            # Python dependencies
├── CLAUDE.md                   # Development standards
└── README.md                   # This file
```

## 🧪 Testing

Run the test suite:
```bash
# Install test dependencies
pip install pytest pytest-cov pytest-mock

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 🔍 Troubleshooting

### Ollama Connection Issues
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If you get "address already in use" error, Ollama is already running
# Verify model is available
ollama list
```

### Import Errors
```bash
# Reinstall dependencies in both environments
pip install -r requirements.txt --force-reinstall
/opt/homebrew/anaconda3/bin/pip install -r requirements.txt --force-reinstall

# Check Python version
python --version  # Should be 3.9+
```

### Streamlit Import Issues
```bash
# Use the provided launcher script instead of direct streamlit command
python start_app.py
```

### Memory Issues
- Reduce `chunk_size` and `top_k` in retrieval config
- Use smaller embedding model if needed
- Process documents in smaller batches

### Performance Optimization
- Use IVF or HNSW index for large document collections
- Adjust `similarity_threshold` for faster retrieval
- Enable GPU acceleration for embeddings if available

## 📈 Performance Tips

1. **Document Processing**:
   - Optimal chunk size: 500-1500 characters
   - Use overlap of 10-20% of chunk size
   - Pre-clean documents to remove unnecessary formatting

2. **Retrieval Optimization**:
   - Start with top_k=5, adjust based on needs
   - Use similarity_threshold > 0.6 to filter low-quality matches
   - Enable metadata filtering for large document sets

3. **Resource Management**:
   - Monitor memory usage with large document collections
   - Use batch processing for multiple file uploads
   - Regularly save vector store to persist changes

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes following the coding standards in `CLAUDE.md`
4. Add tests for new functionality
5. Run the test suite: `pytest tests/`
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## Acknowledgments

This project builds upon several open-source technologies:

- Ollama for local LLM inference
- FAISS for efficient vector similarity search  
- LangChain for RAG framework components
- Streamlit for the web interface
- Sentence Transformers for embedding generation

---

**DocuRAG** - Document processing and question-answering system
