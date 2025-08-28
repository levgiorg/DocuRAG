#!/usr/bin/env python3
"""Simple test script to verify DocuRAG system works."""

import sys
import time
import logging
from pathlib import Path

# Setup simple logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test if all required modules can be imported."""
    logger.info("Testing imports...")
    
    try:
        from src.rag_engine import RAGEngine
        logger.info("✅ RAG Engine imported successfully")
    except Exception as e:
        logger.error(f"❌ Failed to import RAG Engine: {e}")
        return False
    
    try:
        from src.document_processor import DocumentProcessor
        logger.info("✅ Document Processor imported successfully")
    except Exception as e:
        logger.error(f"❌ Failed to import Document Processor: {e}")
        return False
    
    try:
        from src.vector_store import VectorStore
        logger.info("✅ Vector Store imported successfully")
    except Exception as e:
        logger.error(f"❌ Failed to import Vector Store: {e}")
        return False
    
    return True

def test_ollama_connection():
    """Test Ollama connection."""
    logger.info("Testing Ollama connection...")
    
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status()
        models = response.json().get("models", [])
        model_names = [model["name"] for model in models]
        
        logger.info(f"✅ Ollama connected! Available models: {model_names}")
        
        if "gemma3n:e2b" in model_names:
            logger.info("✅ gemma3n:e2b model is available")
            return True
        else:
            logger.warning("⚠️ gemma3n:e2b model not found")
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed to connect to Ollama: {e}")
        return False

def test_document_processing():
    """Test document processing with the existing PDF."""
    logger.info("Testing document processing...")
    
    try:
        from src.document_processor import DocumentProcessor
        
        # Check if PDF exists
        pdf_path = Path("data/documents/1706.03762v7.pdf")
        if not pdf_path.exists():
            logger.error(f"❌ PDF not found at {pdf_path}")
            return False
        
        # Initialize processor
        processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
        
        # Process the PDF
        logger.info("Processing PDF document...")
        chunks, metadata = processor.process_file(pdf_path)
        
        logger.info(f"✅ Successfully processed PDF:")
        logger.info(f"  - Filename: {metadata.filename}")
        logger.info(f"  - File size: {metadata.file_size} bytes")
        logger.info(f"  - Word count: {metadata.word_count}")
        logger.info(f"  - Chunks created: {len(chunks)}")
        
        if len(chunks) > 0:
            logger.info(f"  - First chunk preview: {chunks[0].content[:100]}...")
            return True
        else:
            logger.error("❌ No chunks were created")
            return False
            
    except Exception as e:
        logger.error(f"❌ Document processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rag_system():
    """Test the complete RAG system."""
    logger.info("Testing complete RAG system...")
    
    try:
        from src.rag_engine import RAGEngine
        
        # Initialize RAG engine
        logger.info("Initializing RAG engine...")
        rag = RAGEngine(store_path="data/vector_store")
        
        # Test Ollama connection
        connection_result = rag.test_ollama_connection()
        if not connection_result["success"]:
            logger.error(f"❌ Ollama connection failed: {connection_result.get('error')}")
            return False
        
        logger.info("✅ RAG engine initialized successfully")
        
        # Ingest the PDF document
        pdf_path = "data/documents/1706.03762v7.pdf"
        logger.info(f"Ingesting document: {pdf_path}")
        
        ingest_result = rag.ingest_document(pdf_path)
        if not ingest_result["success"]:
            logger.error(f"❌ Document ingestion failed: {ingest_result.get('error')}")
            return False
        
        logger.info(f"✅ Document ingested successfully:")
        logger.info(f"  - Chunks created: {ingest_result['chunks_created']}")
        
        # Test a simple query
        logger.info("Testing query processing...")
        query_result = rag.query("What is this document about?")
        
        if query_result["success"]:
            logger.info("✅ Query processed successfully:")
            logger.info(f"  - Answer: {query_result['answer'][:200]}...")
            logger.info(f"  - Sources found: {len(query_result['sources'])}")
            logger.info(f"  - Confidence: {query_result['confidence']:.2f}")
            logger.info(f"  - Response time: {query_result['response_time']:.2f}s")
            return True
        else:
            logger.error(f"❌ Query failed: {query_result.get('error')}")
            return False
            
    except Exception as e:
        logger.error(f"❌ RAG system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    logger.info("🚀 Starting DocuRAG System Tests")
    logger.info("=" * 50)
    
    tests = [
        ("Import Tests", test_imports),
        ("Ollama Connection", test_ollama_connection),
        ("Document Processing", test_document_processing),
        ("RAG System", test_rag_system)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running {test_name}...")
        try:
            if test_func():
                passed += 1
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED with exception: {e}")
    
    logger.info("\n" + "=" * 50)
    logger.info(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! DocuRAG system is working correctly.")
        logger.info("\n🌐 You can now start the web interface with:")
        logger.info("   streamlit run app.py")
    else:
        logger.error("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())