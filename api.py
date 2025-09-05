"""FastAPI application for DocuRAG system.

Provides REST API endpoints for document question-answering with exact specification
matching job assignment requirements.
"""

import os
from typing import List, Optional

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.rag_engine import RAGEngine
from src.hf_response_generator import HuggingFaceResponseGenerator
from config import config


# Request/Response Models
class QuestionRequest(BaseModel):
    """Request model for /ask endpoint."""
    question: str = Field(..., description="The question to ask")


class QuestionResponse(BaseModel):
    """Response model for /ask endpoint."""
    answer: str = Field(..., description="The generated answer")
    sources: List[str] = Field(..., description="List of source documents")


class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    message: str
    document_id: str
    chunks_created: int


# Initialize FastAPI app
app = FastAPI(
    title="DocuRAG API",
    description="Document-based Question Answering API using RAG (Retrieval-Augmented Generation)",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG engine instance
rag_engine: Optional[RAGEngine] = None


@app.on_event("startup")
async def startup_event():
    """Initialize RAG engine on startup."""
    global rag_engine
    
    try:
        logger.info("Initializing RAG engine...")
        
        # Create HuggingFace response generator
        hf_generator = HuggingFaceResponseGenerator(
            model_name=config.model_config.hf_model,
            max_length=config.model_config.hf_max_length,
            device=config.model_config.hf_device
        )
        
        # Initialize RAG engine with HuggingFace generator
        rag_engine = RAGEngine(
            config=config,
            response_generator=hf_generator
        )
        
        logger.info("RAG engine initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG engine: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "DocuRAG API is running", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    return {"status": "healthy", "rag_engine": "initialized"}


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question using the RAG system.
    
    Exact specification as required by job assignment:
    - Input: {"question": "string"}
    - Output: {"answer": "string", "sources": ["doc1.txt", "doc2.txt"]}
    """
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    try:
        # Query the RAG engine
        result = rag_engine.query(
            query=request.question,
            top_k=config.retrieval_config.top_k,
            score_threshold=config.retrieval_config.score_threshold
        )
        
        if not result:
            raise HTTPException(status_code=500, detail="Failed to generate response")
        
        # Extract source document names
        sources = []
        if "sources" in result:
            for source_info in result["sources"]:
                if isinstance(source_info, dict) and "source_document" in source_info:
                    doc_name = source_info["source_document"]
                    if doc_name not in sources:
                        sources.append(doc_name)
                elif isinstance(source_info, str):
                    if source_info not in sources:
                        sources.append(source_info)
        
        # Return response in exact format specified
        return QuestionResponse(
            answer=result.get("answer", "No answer generated"),
            sources=sources
        )
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload a document to the RAG system."""
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    # Check file type
    allowed_extensions = {'.pdf', '.txt', '.docx', '.md'}
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Save uploaded file temporarily
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, file.filename)
        
        # Write file content
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Process document with RAG engine
        result = rag_engine.ingest_document(file_path)
        
        if not result or not result.get("success", False):
            raise HTTPException(status_code=500, detail="Failed to process document")
        
        return DocumentUploadResponse(
            message="Document uploaded and processed successfully",
            document_id=file.filename,
            chunks_created=result.get("chunks_created", 0)
        )
        
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")


@app.get("/documents")
async def list_documents():
    """List all documents in the RAG system."""
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    try:
        # Get document list from vector store
        documents = rag_engine.vector_store.get_document_list()
        return {"documents": documents}
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    # Run the application
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )