# DocuRAG - Production RAG System

## Project Overview
DocuRAG is a Retrieval-Augmented Generation system built with Ollama, FAISS, and LangChain. It provides document processing, semantic search, and context-aware question answering with a Streamlit interface.


## Architecture
- **LLM**: Ollama with gemma3n:e2b model for generation
- **Vector Store**: FAISS for efficient local vector storage
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 for semantic encoding
- **Framework**: LangChain for RAG orchestration
- **Interface**: Streamlit for web-based interaction
- **Processing**: PyPDF2, python-docx for multi-format document support

## Project Structure
```
DocuRAG/
├── src/
│   ├── rag_engine.py          # Core RAG logic and orchestration
│   ├── document_processor.py   # PDF/text processing and chunking
│   ├── vector_store.py        # FAISS operations and persistence
│   └── utils.py               # Shared utilities and helpers
├── configs/
│   ├── app_config.yaml        # Application configuration
│   ├── model_config.yaml      # Model and embedding settings
│   └── retrieval_config.yaml  # Retrieval parameters
├── data/
│   ├── documents/             # Input documents
│   ├── vector_store/          # FAISS index storage
│   └── logs/                  # Application logs
├── tests/
│   ├── test_document_processor.py
│   ├── test_vector_store.py
│   └── test_rag_engine.py
├── app.py                     # Streamlit web interface
├── config.py                 # Configuration management
├── requirements.txt          # Python dependencies
├── docker-compose.yml        # Docker deployment
└── README.md                 # Setup and usage instructions
```

## Code Standards

### Python Style
- Follow PEP 8 and PEP 257 conventions strictly
- Use type hints for all function parameters and return values
- Prefer f-strings for string formatting
- Use snake_case for functions/variables, PascalCase for classes
- Keep code minimal - no superfluous imports or blank lines

### ML Engineering Practices
- All configuration in YAML files under configs/
- Structured logging with informative messages (no print statements)
- Proper error handling with custom exceptions
- GPU-aware code with device detection
- Reproducible results with seed management
- Performance metrics and evaluation tracking

### Documentation Requirements
- Docstrings for all public functions using Google style
- Type hints for all parameters and return values
- Inline comments only for complex logic
- README with clear setup instructions and examples

### Testing Standards
- Pytest for all testing with fixtures
- Test both happy path and error conditions
- Property-based testing for data transformations
- Mock external dependencies (Ollama, file system)
- Minimum 80% test coverage

### Configuration Management
- Use Hydra for experiment management
- Environment variables for sensitive data
- Hierarchical config structure (data, model, training, evaluation)
- Never hard-code parameters in source code

### Performance Considerations
- Lazy loading for large models and data
- Efficient batching for embeddings
- Proper memory management for large documents
- Caching for frequently accessed data
- Async operations where beneficial

### Security Guidelines
- Input validation for all user data
- Sanitize file paths and names
- Rate limiting for API endpoints
- No sensitive data in logs
- Secure handling of uploaded files

## Development Workflow
1. Create feature branch from main
2. Implement with comprehensive tests
3. Run full test suite and linting
4. Update documentation as needed
5. Create pull request with clear description

## Deployment
- Docker containerization for production
- Health checks and monitoring
- Graceful shutdown handling
- Resource limits and scaling considerations
- Environment-specific configurations