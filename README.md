# DocuRAG

A Python-based Retrieval-Augmented Generation (RAG) system for document question-answering. Built with FastAPI, HuggingFace transformers, and FAISS vector search.

## Features

- Document ingestion (PDF, TXT, DOCX, MD)
- FAISS-based vector indexing with persistent storage
- HuggingFace transformer models for embeddings and generation
- FastAPI REST API with exact specification compliance
- Context highlighting in responses
- Dynamic document uploading
- Comprehensive logging and monitoring
- Optional Streamlit web interface

## Requirements

- Python 3.10+
- HuggingFace account token (for gated models)

## Quick Start

### Installation

```bash
git clone <repository-url>
cd DocuRAG
pip install -r requirements.txt
```

### Environment Setup

```bash
cp .env.example .env
# Edit .env and add your HuggingFace token:
# HF_TOKEN=your_huggingface_token_here
```

### Run API Server

```bash
python start_api.py
```

The API will be available at `http://localhost:8000` with documentation at `http://localhost:8000/docs`

### Run Web Interface (Optional)

```bash
python start_app.py
```

Access the web interface at `http://localhost:8502`

## API Usage

### Ask Questions

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of the document?"}'
```

**Response:**
```json
{
  "answer": "The main topic is...",
  "sources": ["document1.pdf", "document2.txt"]
}
```

### Upload Documents

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@document.pdf"
```

### List Documents

```bash
curl -X GET "http://localhost:8000/documents"
```

## Configuration

The system uses YAML configuration files in the `configs/` directory:

- `model_config.yaml`: Model settings (LLM, embeddings, device)
- `retrieval_config.yaml`: Retrieval parameters (chunk size, top-k)
- `app_config.yaml`: Application settings (file limits, history)

## Architecture

- **Document Processor**: Multi-format parsing and intelligent chunking
- **Vector Store**: FAISS-based similarity search with metadata filtering  
- **RAG Engine**: Query processing, retrieval orchestration, response generation
- **HF Generator**: HuggingFace transformers integration with context highlighting
- **FastAPI**: REST API endpoints with automatic validation
- **Streamlit UI**: Optional web interface for document management

## Project Structure

```
DocuRAG/
├── src/
│   ├── document_processor.py  # Document parsing and chunking
│   ├── vector_store.py        # FAISS vector operations  
│   ├── rag_engine.py          # Main RAG orchestration
│   ├── hf_response_generator.py # HuggingFace integration
│   └── utils.py               # Shared utilities
├── configs/                   # Configuration files
├── api.py                     # FastAPI application
├── app.py                     # Streamlit interface
├── start_api.py              # API launcher
├── start_app.py              # UI launcher
├── requirements.txt          # Dependencies
└── .env.example             # Environment template
```

## Testing

```bash
# Install test dependencies
pip install pytest

# Run tests
pytest tests/ -v

# System integration test
python test_system.py
```

## Example Usage

1. **Start the API server:**
   ```bash
   python start_api.py
   ```

2. **Upload a document:**
   ```bash
   curl -F "file=@example.pdf" http://localhost:8000/upload
   ```

3. **Ask a question:**
   ```bash
   curl -X POST http://localhost:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "What are the key findings?"}'
   ```

## Troubleshooting

**Model Loading Issues:**
- Ensure HF_TOKEN is set in `.env`
- Check internet connection for model download
- Verify sufficient disk space (~2GB for model)

**Memory Issues:**
- Reduce `hf_max_length` in model config
- Use CPU instead of GPU if memory limited
- Process smaller document batches

**API Connection:**
- Check if port 8000 is available
- Verify FastAPI server started successfully
- Check logs for detailed error messages