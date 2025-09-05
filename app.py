"""Streamlit web interface for DocuRAG system.

A user-friendly web interface for document ingestion and question-answering.
"""

import os
import tempfile
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from src.rag_engine import RAGEngine
from src.hf_response_generator import HuggingFaceResponseGenerator
from config import config


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "rag_engine" not in st.session_state:
        st.session_state.rag_engine = None
    
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    
    if "processed_documents" not in st.session_state:
        st.session_state.processed_documents = []


def setup_rag_engine():
    """Initialize RAG engine with HuggingFace model."""
    if st.session_state.rag_engine is None:
        with st.spinner("Initializing RAG system with HuggingFace model..."):
            try:
                # Create HuggingFace response generator
                hf_generator = HuggingFaceResponseGenerator(
                    model_name=config.model_config.hf_model,
                    max_length=config.model_config.hf_max_length,
                    device=config.model_config.hf_device
                )
                
                # Initialize RAG engine with HF generator
                st.session_state.rag_engine = RAGEngine(
                    config=config,
                    response_generator=hf_generator
                )
                st.success(f"RAG system initialized with {config.model_config.hf_model}!")
            except Exception as e:
                st.error(f"Failed to initialize RAG system: {e}")
                st.stop()


def test_model_status():
    """Display current model information."""
    if st.session_state.rag_engine:
        st.info(f"🤖 Using HuggingFace model: **{config.model_config.hf_model}**")
        st.info(f"💾 Device: **{config.model_config.hf_device}**")
        st.info(f"📏 Max length: **{config.model_config.hf_max_length}** tokens")


def display_sidebar():
    """Display sidebar with system information and controls."""
    with st.sidebar:
        st.title("🔧 System Status")
        
        # Model information
        if st.button("Show Model Info"):
            test_model_status()
        
        # System statistics
        if st.session_state.rag_engine:
            stats = st.session_state.rag_engine.get_system_stats()
            
            st.subheader("📊 Statistics")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Documents", stats["vector_store"]["total_documents"])
                st.metric("Chunks", stats["vector_store"]["total_chunks"])
            
            with col2:
                st.metric("Conversations", stats["conversation"]["total_turns"])
                st.metric("Model", config.model_config.llm_model)
        
        # Configuration
        st.subheader("⚙️ Settings")
        
        # Retrieval settings
        top_k = st.slider("Top-K Results", 1, 20, config.retrieval_config.top_k)
        similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 
                                       config.retrieval_config.similarity_threshold, 0.05)
        
        # Update configuration
        config.retrieval_config.top_k = top_k
        config.retrieval_config.similarity_threshold = similarity_threshold
        
        # Conversation history controls
        st.subheader("💬 Conversation")
        if st.button("Clear History"):
            if st.session_state.rag_engine:
                st.session_state.rag_engine.clear_conversation_history()
                st.session_state.conversation_history = []
                st.success("Conversation history cleared!")
        
        # Document management
        st.subheader("📚 Documents")
        st.info(f"Processed: {len(st.session_state.processed_documents)} files")


def document_ingestion_tab():
    """Document ingestion interface."""
    st.header("📥 Document Ingestion")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        accept_multiple_files=True,
        type=['pdf', 'txt', 'docx', 'md'],
        help="Supported formats: PDF, TXT, DOCX, MD"
    )
    
    if uploaded_files:
        st.write(f"Selected {len(uploaded_files)} files:")
        for file in uploaded_files:
            st.write(f"- {file.name} ({file.size:,} bytes)")
        
        if st.button("Process Files", type="primary"):
            process_uploaded_files(uploaded_files)
    
    # Directory processing
    st.subheader("📁 Process Local Directory")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        directory_path = st.text_input(
            "Directory Path",
            placeholder="/path/to/your/documents",
            help="Enter the full path to a directory containing documents"
        )
    
    with col2:
        recursive = st.checkbox("Recursive", value=True, 
                               help="Process subdirectories")
    
    if directory_path and st.button("Process Directory"):
        if Path(directory_path).exists():
            process_directory(directory_path, recursive)
        else:
            st.error("Directory does not exist!")


def process_uploaded_files(uploaded_files: List[Any]):
    """Process uploaded files."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    successful_files = []
    failed_files = []
    total_chunks = 0
    
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing {uploaded_file.name}...")
        
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_file_path = tmp_file.name
        
        try:
            # Process the file
            result = st.session_state.rag_engine.ingest_document(tmp_file_path)
            
            if result["success"]:
                successful_files.append({
                    "name": uploaded_file.name,
                    "chunks": result["chunks_created"],
                    "stats": result.get("processing_stats", {})
                })
                total_chunks += result["chunks_created"]
            else:
                failed_files.append({
                    "name": uploaded_file.name,
                    "error": result["error"]
                })
        
        except Exception as e:
            failed_files.append({
                "name": uploaded_file.name,
                "error": str(e)
            })
        
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
        
        # Update progress
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    # Display results
    status_text.text("Processing complete!")
    
    if successful_files:
        st.success(f"Successfully processed {len(successful_files)} files ({total_chunks} chunks created)")
        
        # Update session state
        st.session_state.processed_documents.extend(successful_files)
        
        # Show details
        with st.expander("Processing Details"):
            for file_info in successful_files:
                st.write(f"✅ **{file_info['name']}** - {file_info['chunks']} chunks")
                stats = file_info.get('stats', {})
                if stats:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Words", stats.get('word_count', 0))
                    with col2:
                        st.metric("Characters", stats.get('char_count', 0))
                    with col3:
                        st.metric("Pages", stats.get('page_count', 0) or 'N/A')
    
    if failed_files:
        st.error(f"Failed to process {len(failed_files)} files")
        with st.expander("Error Details"):
            for file_info in failed_files:
                st.write(f"❌ **{file_info['name']}**: {file_info['error']}")


def process_directory(directory_path: str, recursive: bool):
    """Process a local directory."""
    with st.spinner(f"Processing directory: {directory_path}"):
        try:
            result = st.session_state.rag_engine.ingest_directory(directory_path, recursive)
            
            if result["success"]:
                st.success(
                    f"Successfully processed {result['documents_processed']} documents "
                    f"({result['chunks_created']} chunks created)"
                )
                
                # Show processing statistics
                stats = result.get("processing_stats", {})
                if stats:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Words", stats.get('total_words', 0))
                    with col2:
                        st.metric("Total Characters", stats.get('total_chars', 0))
                    with col3:
                        st.metric("Total Pages", stats.get('total_pages', 0))
                
                # Update session state
                st.session_state.processed_documents.append({
                    "name": f"Directory: {directory_path}",
                    "chunks": result['chunks_created'],
                    "documents": result['documents_processed']
                })
            
            else:
                st.error(f"Failed to process directory: {result['error']}")
        
        except Exception as e:
            st.error(f"Error processing directory: {e}")


def chat_interface():
    """Main chat interface for question-answering."""
    st.header("💬 Ask Questions")
    
    # Display conversation history
    for i, turn in enumerate(st.session_state.conversation_history):
        with st.chat_message("user"):
            st.write(turn["question"])
        
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            
            # Show sources in expander
            if turn.get("sources"):
                with st.expander(f"📚 Sources ({len(turn['sources'])})"):
                    for source in turn["sources"]:
                        st.write(f"**{source['document']}**")
                        if source.get('page'):
                            st.write(f"Page {source['page']}")
                        st.write(f"Quality Score: {source.get('quality_score', 'N/A'):.2f}")
                        st.text_area(
                            "Content Preview", 
                            source["content_preview"], 
                            height=100, 
                            key=f"source_{i}_{source['source_id']}"
                        )
                        st.write("---")
            
            # Show confidence and timing
            col1, col2 = st.columns(2)
            with col1:
                confidence = turn.get("confidence", 0) * 100
                st.metric("Confidence", f"{confidence:.1f}%")
            with col2:
                response_time = turn.get("response_time", 0)
                st.metric("Response Time", f"{response_time:.2f}s")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.rag_engine.query(prompt)
            
            if result["success"]:
                st.write(result["answer"])
                
                # Add to conversation history
                st.session_state.conversation_history.append({
                    "question": prompt,
                    "answer": result["answer"],
                    "sources": result.get("sources", []),
                    "confidence": result.get("confidence", 0),
                    "response_time": result.get("response_time", 0)
                })
                
                # Show sources
                if result.get("sources"):
                    with st.expander(f"📚 Sources ({len(result['sources'])})"):
                        for source in result["sources"]:
                            st.write(f"**{source['document']}**")
                            if source.get('page'):
                                st.write(f"Page {source['page']}")
                            st.write(f"Quality Score: {source.get('quality_score', 'N/A'):.2f}")
                            st.text_area(
                                "Content Preview", 
                                source["content_preview"], 
                                height=100, 
                                key=f"new_source_{source['source_id']}"
                            )
                            st.write("---")
                
                # Show metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    confidence = result.get("confidence", 0) * 100
                    st.metric("Confidence", f"{confidence:.1f}%")
                with col2:
                    response_time = result.get("response_time", 0)
                    st.metric("Response Time", f"{response_time:.2f}s")
                with col3:
                    chunks_retrieved = result.get("chunks_retrieved", 0)
                    st.metric("Chunks Used", chunks_retrieved)
            
            else:
                st.error(f"Error: {result.get('error', 'Unknown error')}")


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="DocuRAG - Document Q&A System",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🔍 DocuRAG - Production RAG System")
    st.markdown("*Intelligent document processing and question-answering with HuggingFace & FAISS*")
    
    # Initialize session state
    initialize_session_state()
    
    # Setup RAG engine
    setup_rag_engine()
    
    # Display sidebar
    display_sidebar()
    
    # Main content tabs
    tab1, tab2 = st.tabs(["💬 Chat", "📥 Document Management"])
    
    with tab1:
        if st.session_state.rag_engine:
            # Check if any documents are loaded
            stats = st.session_state.rag_engine.get_system_stats()
            if stats["vector_store"]["total_documents"] == 0:
                st.info("👋 Welcome! Please upload some documents in the 'Document Management' tab to start asking questions.")
            else:
                chat_interface()
        else:
            st.error("RAG engine not initialized")
    
    with tab2:
        if st.session_state.rag_engine:
            document_ingestion_tab()
        else:
            st.error("RAG engine not initialized")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**DocuRAG** - Built with Streamlit, LangChain, FAISS, and HuggingFace | "
        f"Current Model: {config.model_config.llm_model}"
    )


if __name__ == "__main__":
    main()