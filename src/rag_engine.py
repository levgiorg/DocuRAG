"""RAG Engine implementation for DocuRAG system.

Main orchestration component that combines document processing, vector storage,
and LLM generation for question-answering.
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union

import requests
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
from pydantic import BaseModel

from .document_processor import DocumentProcessor, DocumentChunk, DocumentMetadata
from .vector_store import VectorStore
from .utils import timing_decorator, PerformanceMonitor
from config import config


class ConversationTurn(BaseModel):
    """Represents a single conversation turn."""
    
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    timestamp: datetime
    response_time: float
    confidence_score: Optional[float] = None


class QueryRewriter:
    """Query rewriting for better retrieval."""
    
    @staticmethod
    def expand_query(query: str) -> List[str]:
        """Expand query with synonyms and related terms.
        
        Args:
            query: Original query
            
        Returns:
            List[str]: List of query variants
        """
        # Simple query expansion (can be enhanced with NLP models)
        queries = [query]
        
        # Add query without stopwords
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        filtered_words = [word for word in query.lower().split() if word not in stopwords]
        if len(filtered_words) > 1:
            queries.append(' '.join(filtered_words))
        
        # Add individual important terms for broader search
        important_words = [word for word in filtered_words if len(word) > 3]
        if important_words:
            queries.extend(important_words)
        
        return list(set(queries))  # Remove duplicates
    
    @staticmethod
    def rephrase_query(query: str) -> str:
        """Rephrase query for better matching.
        
        Args:
            query: Original query
            
        Returns:
            str: Rephrased query
        """
        # Simple rephrasing rules
        query = query.strip()
        
        # Convert questions to statements
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which']
        for word in question_words:
            if query.lower().startswith(word):
                # Remove question word and question mark
                rephrased = query[len(word):].strip()
                if rephrased.endswith('?'):
                    rephrased = rephrased[:-1]
                return rephrased.strip()
        
        return query


class ResponseGenerator:
    """Generate responses using Ollama LLM."""
    
    def __init__(self, model_name: str, base_url: str):
        """Initialize response generator.
        
        Args:
            model_name: Name of the Ollama model
            base_url: Base URL for Ollama API
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.timeout = 120  # 2 minute timeout
    
    def generate_response(self, 
                         query: str, 
                         context_chunks: List[DocumentChunk],
                         conversation_history: Optional[List[ConversationTurn]] = None) -> Tuple[str, float]:
        """Generate response using retrieved context.
        
        Args:
            query: User question
            context_chunks: Retrieved document chunks
            conversation_history: Previous conversation turns
            
        Returns:
            Tuple[str, float]: Generated response and confidence score
        """
        # Build context from chunks
        context_text = self._build_context(context_chunks)
        
        # Build conversation context
        conversation_context = ""
        if conversation_history:
            conversation_context = self._build_conversation_context(conversation_history)
        
        # Create prompt
        prompt = self._create_prompt(query, context_text, conversation_context)
        
        # Generate response
        try:
            response_text = self._call_ollama(prompt)
            confidence_score = self._calculate_confidence(response_text, context_chunks)
            
            return response_text, confidence_score
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return "I apologize, but I encountered an error while generating the response. Please try again.", 0.0
    
    def _build_context(self, chunks: List[DocumentChunk]) -> str:
        """Build context string from document chunks.
        
        Args:
            chunks: Retrieved chunks
            
        Returns:
            str: Formatted context string
        """
        if not chunks:
            return "No relevant context found."
        
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.metadata.get("source_document", "Unknown")
            page = chunk.metadata.get("page_number")
            
            source_info = f"Source {i}: {source}"
            if page:
                source_info += f" (Page {page})"
            
            context_parts.append(f"{source_info}\n{chunk.content}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def _build_conversation_context(self, history: List[ConversationTurn]) -> str:
        """Build conversation context from history.
        
        Args:
            history: Conversation history
            
        Returns:
            str: Formatted conversation context
        """
        if not history:
            return ""
        
        # Use last few turns for context
        recent_turns = history[-3:]  # Last 3 turns
        
        context_parts = []
        for turn in recent_turns:
            context_parts.append(f"Previous Question: {turn.question}")
            context_parts.append(f"Previous Answer: {turn.answer}")
        
        return "\n".join(context_parts)
    
    def _create_prompt(self, query: str, context: str, conversation_context: str = "") -> str:
        """Create prompt for the LLM.
        
        Args:
            query: User question
            context: Document context
            conversation_context: Previous conversation context
            
        Returns:
            str: Formatted prompt
        """
        base_prompt = """You are a helpful AI assistant that answers questions based on the provided context. 
Use the context information to provide accurate and detailed answers. If the context doesn't contain 
enough information to fully answer the question, say so and provide what information you can.

Always cite your sources by mentioning which source number you're referencing (e.g., "According to Source 1...").

"""
        
        if conversation_context:
            base_prompt += f"\nPrevious Conversation:\n{conversation_context}\n"
        
        base_prompt += f"""
Context Information:
{context}

Question: {query}

Please provide a comprehensive answer based on the context above:"""
        
        return base_prompt
    
    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API for text generation.
        
        Args:
            prompt: Input prompt
            
        Returns:
            str: Generated response
        """
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "options": {
                "temperature": config.model_config.temperature,
                "num_predict": config.model_config.max_tokens,
                "stop": ["Question:", "Context Information:"]
            },
            "stream": False
        }
        
        response = self.session.post(url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        return result.get("response", "").strip()
    
    def _calculate_confidence(self, response: str, context_chunks: List[DocumentChunk]) -> float:
        """Calculate confidence score for the response.
        
        Args:
            response: Generated response
            context_chunks: Source chunks
            
        Returns:
            float: Confidence score between 0 and 1
        """
        if not response or not context_chunks:
            return 0.0
        
        # Base confidence
        confidence = 0.5
        
        # Boost confidence if response mentions sources
        source_mentions = response.lower().count("source")
        if source_mentions > 0:
            confidence += 0.2
        
        # Check for hedge words that indicate uncertainty
        hedge_words = ["maybe", "possibly", "might", "could", "uncertain", "unclear", "don't know"]
        hedge_count = sum(1 for word in hedge_words if word in response.lower())
        confidence -= hedge_count * 0.1
        
        # Boost confidence based on context quality
        avg_quality = sum(chunk.quality_score or 0.5 for chunk in context_chunks) / len(context_chunks)
        confidence += (avg_quality - 0.5) * 0.2
        
        # Length factor (very short responses might be less confident)
        if len(response.split()) < 10:
            confidence -= 0.1
        
        return max(0.0, min(1.0, confidence))


class RAGEngine:
    """Main RAG Engine coordinating all components."""
    
    def __init__(self, config=None, response_generator=None, store_path: str = "data/vector_store"):
        """Initialize RAG Engine.
        
        Args:
            config: Configuration object (defaults to global config)
            response_generator: Custom response generator (optional)
            store_path: Path for vector store
        """
        # Use provided config or default
        if config is None:
            from config import config as default_config
            config = default_config
        
        # Initialize components
        self.document_processor = DocumentProcessor(
            chunk_size=config.retrieval_config.chunk_size,
            chunk_overlap=config.retrieval_config.chunk_overlap
        )
        
        self.vector_store = VectorStore(
            embedding_model=config.model_config.embedding_model,
            store_path=store_path
        )
        
        # Use provided response generator or create default Ollama one
        if response_generator is not None:
            self.response_generator = response_generator
        else:
            self.response_generator = ResponseGenerator(
                model_name=config.model_config.llm_model,
                base_url=config.model_config.ollama_base_url
            )
        
        self.query_rewriter = QueryRewriter()
        
        # Conversation management
        self.conversation_history: List[ConversationTurn] = []
        self.max_history_length = config.app_config.max_conversation_turns
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        
        logger.info("RAG Engine initialized successfully")
    
    @timing_decorator
    def ingest_document(self, file_path: str) -> Dict[str, Any]:
        """Ingest a single document into the system.
        
        Args:
            file_path: Path to document file
            
        Returns:
            Dict[str, Any]: Ingestion results
        """
        logger.info(f"Ingesting document: {file_path}")
        
        try:
            # Process document
            self.performance_monitor.start_timer("document_processing")
            chunks, metadata = self.document_processor.process_file(file_path)
            self.performance_monitor.end_timer("document_processing")
            
            if not chunks:
                return {
                    "success": False,
                    "error": "No content extracted from document",
                    "chunks_created": 0
                }
            
            # Add to vector store
            self.performance_monitor.start_timer("vector_indexing")
            chunk_ids = self.vector_store.add_chunks(chunks)
            self.vector_store.add_document_metadata(metadata.dict())
            self.performance_monitor.end_timer("vector_indexing")
            
            # Save vector store
            self.vector_store.save()
            
            result = {
                "success": True,
                "chunks_created": len(chunks),
                "chunk_ids": chunk_ids,
                "document_metadata": metadata.dict(),
                "processing_stats": {
                    "word_count": metadata.word_count,
                    "char_count": metadata.char_count,
                    "page_count": metadata.page_count
                }
            }
            
            logger.info(f"Successfully ingested {file_path}: {len(chunks)} chunks created")
            return result
            
        except Exception as e:
            logger.error(f"Failed to ingest document {file_path}: {e}")
            return {
                "success": False,
                "error": str(e),
                "chunks_created": 0
            }
    
    @timing_decorator
    def ingest_directory(self, directory_path: str, recursive: bool = True) -> Dict[str, Any]:
        """Ingest all documents in a directory.
        
        Args:
            directory_path: Path to directory
            recursive: Whether to process subdirectories
            
        Returns:
            Dict[str, Any]: Ingestion results
        """
        logger.info(f"Ingesting directory: {directory_path}")
        
        try:
            # Process all documents
            self.performance_monitor.start_timer("document_processing")
            all_chunks, all_metadata = self.document_processor.process_directory(
                directory_path, recursive
            )
            self.performance_monitor.end_timer("document_processing")
            
            if not all_chunks:
                return {
                    "success": False,
                    "error": "No content extracted from any documents",
                    "documents_processed": 0,
                    "chunks_created": 0
                }
            
            # Add to vector store
            self.performance_monitor.start_timer("vector_indexing")
            chunk_ids = self.vector_store.add_chunks(all_chunks)
            
            for metadata in all_metadata:
                self.vector_store.add_document_metadata(metadata.dict())
            self.performance_monitor.end_timer("vector_indexing")
            
            # Save vector store
            self.vector_store.save()
            
            result = {
                "success": True,
                "documents_processed": len(all_metadata),
                "chunks_created": len(all_chunks),
                "chunk_ids": chunk_ids,
                "processing_stats": {
                    "total_words": sum(m.word_count for m in all_metadata),
                    "total_chars": sum(m.char_count for m in all_metadata),
                    "total_pages": sum(m.page_count or 0 for m in all_metadata)
                }
            }
            
            logger.info(
                f"Successfully ingested directory {directory_path}: "
                f"{len(all_metadata)} documents, {len(all_chunks)} chunks"
            )
            return result
            
        except Exception as e:
            logger.error(f"Failed to ingest directory {directory_path}: {e}")
            return {
                "success": False,
                "error": str(e),
                "documents_processed": 0,
                "chunks_created": 0
            }
    
    @timing_decorator
    def query(self, 
             query: str = None,  # For backward compatibility
             question: str = None,  # Alternative parameter name
             top_k: int = None,
             score_threshold: float = None,
             use_conversation_history: bool = True,
             metadata_filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Query the RAG system with a question.
        
        Args:
            query: User question (alternative to question)
            question: User question (alternative to query)
            top_k: Number of chunks to retrieve (overrides config)
            score_threshold: Minimum similarity score (overrides config)
            use_conversation_history: Whether to use conversation history
            metadata_filter: Optional metadata filters for retrieval
            
        Returns:
            Dict[str, Any]: Query results with answer and sources
        """
        # Handle parameter compatibility
        user_question = question or query
        if not user_question:
            raise ValueError("Either 'query' or 'question' parameter must be provided")
        
        # Use provided parameters or defaults from config
        retrieval_top_k = top_k or config.retrieval_config.top_k
        retrieval_threshold = score_threshold or getattr(config.retrieval_config, 'score_threshold', 0.0)
        
        start_time = time.time()
        logger.info(f"Processing query: {user_question}")
        
        try:
            # Query rewriting for better retrieval
            expanded_queries = self.query_rewriter.expand_query(user_question)
            
            # Retrieve relevant chunks using hybrid search
            self.performance_monitor.start_timer("retrieval")
            all_chunks = []
            
            for query_variant in expanded_queries[:3]:  # Use top 3 variants
                chunks = self.vector_store.hybrid_search(
                    query=query_variant,
                    top_k=retrieval_top_k,
                    metadata_filter=metadata_filter
                )
                all_chunks.extend(chunks)
            
            # Deduplicate and rank chunks
            chunk_dict = {}
            for chunk, score in all_chunks:
                chunk_key = (chunk.source_file, chunk.chunk_index)
                if chunk_key not in chunk_dict or chunk_dict[chunk_key][1] < score:
                    chunk_dict[chunk_key] = (chunk, score)
            
            # Get top chunks
            ranked_chunks = sorted(chunk_dict.values(), key=lambda x: x[1], reverse=True)
            top_chunks = [chunk for chunk, score in ranked_chunks[:retrieval_top_k]]
            
            self.performance_monitor.end_timer("retrieval")
            
            if not top_chunks:
                return {
                    "success": True,
                    "answer": "I couldn't find any relevant information to answer your question. Please try rephrasing your question or check if the relevant documents have been ingested.",
                    "sources": [],
                    "confidence": 0.0,
                    "response_time": time.time() - start_time
                }
            
            # Generate response
            self.performance_monitor.start_timer("generation")
            conversation_context = self.conversation_history if use_conversation_history else None
            
            # Check if using HuggingFace generator (has context highlighting)
            if hasattr(self.response_generator, '_generate_highlighted_contexts'):
                answer, confidence, highlighted_contexts = self.response_generator.generate_response(
                    query=user_question,
                    context_chunks=top_chunks,
                    conversation_history=conversation_context
                )
            else:
                # Fallback for Ollama generator
                answer, confidence = self.response_generator.generate_response(
                    query=user_question,
                    context_chunks=top_chunks,
                    conversation_history=conversation_context
                )
                highlighted_contexts = []
            self.performance_monitor.end_timer("generation")
            
            # Prepare sources information
            sources = []
            for i, chunk in enumerate(top_chunks):
                source_info = {
                    "source_id": i + 1,
                    "source_document": chunk.metadata.get("source_document", "Unknown"),
                    "document": chunk.metadata.get("source_document", "Unknown"),  # Keep both for compatibility
                    "page": chunk.metadata.get("page_number"),
                    "content_preview": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                    "quality_score": chunk.quality_score,
                    "chunk_index": chunk.chunk_index
                }
                sources.append(source_info)
            
            # Store conversation turn
            response_time = time.time() - start_time
            if config.app_config.enable_conversation_history:
                turn = ConversationTurn(
                    question=user_question,
                    answer=answer,
                    sources=sources,
                    timestamp=datetime.now(),
                    response_time=response_time,
                    confidence_score=confidence
                )
                
                self.conversation_history.append(turn)
                
                # Trim history if too long
                if len(self.conversation_history) > self.max_history_length:
                    self.conversation_history = self.conversation_history[-self.max_history_length:]
            
            result = {
                "success": True,
                "answer": answer,
                "sources": sources,
                "confidence": confidence,
                "response_time": response_time,
                "chunks_retrieved": len(top_chunks)
            }
            
            # Add highlighted contexts if available
            if highlighted_contexts:
                result["highlighted_contexts"] = highlighted_contexts
            
            logger.info(f"Query completed successfully in {response_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Failed to process query: {e}")
            return {
                "success": False,
                "error": str(e),
                "answer": "I encountered an error while processing your question. Please try again.",
                "sources": [],
                "confidence": 0.0,
                "response_time": time.time() - start_time
            }
    
    def clear_conversation_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history.clear()
        logger.info("Conversation history cleared")
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history as list of dictionaries.
        
        Returns:
            List[Dict[str, Any]]: Conversation history
        """
        return [turn.dict() for turn in self.conversation_history]
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics.
        
        Returns:
            Dict[str, Any]: System statistics
        """
        vector_stats = self.vector_store.get_stats()
        
        performance_stats = {}
        for operation in ["document_processing", "vector_indexing", "retrieval", "generation"]:
            stats = self.performance_monitor.get_stats(operation)
            if stats["count"] > 0:
                performance_stats[operation] = stats
        
        return {
            "vector_store": vector_stats,
            "performance": performance_stats,
            "conversation": {
                "total_turns": len(self.conversation_history),
                "max_history_length": self.max_history_length
            },
            "configuration": {
                "chunk_size": config.retrieval_config.chunk_size,
                "chunk_overlap": config.retrieval_config.chunk_overlap,
                "top_k": config.retrieval_config.top_k,
                "similarity_threshold": config.retrieval_config.similarity_threshold,
                "llm_model": config.model_config.llm_model,
                "embedding_model": config.model_config.embedding_model
            }
        }
    
