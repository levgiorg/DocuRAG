"""Tests for RAG engine module."""

import tempfile
from unittest.mock import Mock, patch, MagicMock

import pytest
import requests

from src.rag_engine import RAGEngine, QueryRewriter, ResponseGenerator, ConversationTurn
from src.document_processor import DocumentChunk


class TestQueryRewriter:
    """Test cases for QueryRewriter."""
    
    def test_expand_query(self):
        """Test query expansion functionality."""
        query = "What is machine learning in artificial intelligence?"
        expanded = QueryRewriter.expand_query(query)
        
        assert len(expanded) >= 1
        assert query in expanded
        
        # Should include query without stopwords
        filtered_query = "machine learning artificial intelligence"
        assert any(filtered_query in exp_query for exp_query in expanded)
    
    def test_expand_short_query(self):
        """Test expansion of short query."""
        query = "AI"
        expanded = QueryRewriter.expand_query(query)
        
        assert query in expanded
        assert len(expanded) >= 1
    
    def test_rephrase_query(self):
        """Test query rephrasing."""
        # Question starting with "what"
        question = "What is machine learning?"
        rephrased = QueryRewriter.rephrase_query(question)
        assert rephrased == "is machine learning"
        
        # Question starting with "how"
        question = "How does AI work?"
        rephrased = QueryRewriter.rephrase_query(question)
        assert rephrased == "does AI work"
        
        # Statement (no rephrasing)
        statement = "Machine learning is powerful"
        rephrased = QueryRewriter.rephrase_query(statement)
        assert rephrased == statement


class TestResponseGenerator:
    """Test cases for ResponseGenerator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = ResponseGenerator("test-model", "http://localhost:11434")
    
    def test_init(self):
        """Test ResponseGenerator initialization."""
        assert self.generator.model_name == "test-model"
        assert self.generator.base_url == "http://localhost:11434"
        assert self.generator.session.timeout == 120
    
    def test_build_context_empty(self):
        """Test building context from empty chunks."""
        context = self.generator._build_context([])
        assert context == "No relevant context found."
    
    def test_build_context_with_chunks(self):
        """Test building context from chunks."""
        chunks = [
            DocumentChunk(
                content="Test content 1",
                metadata={"source_document": "test1.pdf", "page_number": 1},
                chunk_index=0,
                source_file="test1.pdf"
            ),
            DocumentChunk(
                content="Test content 2",
                metadata={"source_document": "test2.pdf"},
                chunk_index=0,
                source_file="test2.pdf"
            )
        ]
        
        context = self.generator._build_context(chunks)
        
        assert "Source 1: test1.pdf (Page 1)" in context
        assert "Source 2: test2.pdf" in context
        assert "Test content 1" in context
        assert "Test content 2" in context
    
    def test_build_conversation_context_empty(self):
        """Test building conversation context from empty history."""
        context = self.generator._build_conversation_context([])
        assert context == ""
    
    def test_build_conversation_context(self):
        """Test building conversation context."""
        history = [
            ConversationTurn(
                question="What is AI?",
                answer="AI is artificial intelligence.",
                sources=[],
                timestamp="2023-01-01T00:00:00",
                response_time=1.0
            )
        ]
        
        context = self.generator._build_conversation_context(history)
        
        assert "Previous Question: What is AI?" in context
        assert "Previous Answer: AI is artificial intelligence." in context
    
    def test_create_prompt(self):
        """Test prompt creation."""
        query = "What is machine learning?"
        context = "Machine learning is a subset of AI."
        conversation_context = "Previous: What is AI?"
        
        prompt = self.generator._create_prompt(query, context, conversation_context)
        
        assert query in prompt
        assert context in prompt
        assert conversation_context in prompt
        assert "Context Information:" in prompt
    
    @patch('requests.Session.post')
    def test_call_ollama_success(self, mock_post):
        """Test successful Ollama API call."""
        mock_response = Mock()
        mock_response.json.return_value = {"response": "Test response"}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        response = self.generator._call_ollama("Test prompt")
        
        assert response == "Test response"
        mock_post.assert_called_once()
    
    @patch('requests.Session.post')
    def test_call_ollama_failure(self, mock_post):
        """Test Ollama API call failure."""
        mock_post.side_effect = requests.RequestException("Connection error")
        
        with pytest.raises(requests.RequestException):
            self.generator._call_ollama("Test prompt")
    
    def test_calculate_confidence_empty(self):
        """Test confidence calculation with empty inputs."""
        confidence = self.generator._calculate_confidence("", [])
        assert confidence == 0.0
    
    def test_calculate_confidence_with_sources(self):
        """Test confidence calculation with source mentions."""
        response = "According to Source 1, this is the answer."
        chunks = [
            DocumentChunk(
                content="Test content",
                metadata={},
                chunk_index=0,
                source_file="test.pdf",
                quality_score=0.8
            )
        ]
        
        confidence = self.generator._calculate_confidence(response, chunks)
        
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be higher due to source mention
    
    def test_calculate_confidence_with_hedges(self):
        """Test confidence calculation with hedge words."""
        response = "Maybe this is the answer, but I'm not certain."
        chunks = [
            DocumentChunk(
                content="Test content",
                metadata={},
                chunk_index=0,
                source_file="test.pdf",
                quality_score=0.5
            )
        ]
        
        confidence = self.generator._calculate_confidence(response, chunks)
        
        assert 0.0 <= confidence <= 1.0
        # Should be lower due to hedge words
    
    @patch.object(ResponseGenerator, '_call_ollama')
    def test_generate_response(self, mock_ollama):
        """Test complete response generation."""
        mock_ollama.return_value = "This is a test response."
        
        chunks = [
            DocumentChunk(
                content="Test content",
                metadata={"source_document": "test.pdf"},
                chunk_index=0,
                source_file="test.pdf",
                quality_score=0.8
            )
        ]
        
        response, confidence = self.generator.generate_response(
            "Test question", chunks
        )
        
        assert response == "This is a test response."
        assert 0.0 <= confidence <= 1.0
        mock_ollama.assert_called_once()


class TestRAGEngine:
    """Test cases for RAGEngine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.temp_dir = temp_dir
            
            # Mock all the components
            with patch('src.rag_engine.DocumentProcessor'), \
                 patch('src.rag_engine.VectorStore'), \
                 patch('src.rag_engine.ResponseGenerator'), \
                 patch('src.rag_engine.config') as mock_config:
                
                # Setup mock config
                mock_config.retrieval_config.chunk_size = 1000
                mock_config.retrieval_config.chunk_overlap = 200
                mock_config.retrieval_config.top_k = 5
                mock_config.model_config.embedding_model = "test-model"
                mock_config.model_config.llm_model = "test-llm"
                mock_config.model_config.ollama_base_url = "http://localhost:11434"
                mock_config.app_config.max_conversation_turns = 10
                
                self.rag_engine = RAGEngine(store_path=temp_dir)
    
    def test_init(self):
        """Test RAGEngine initialization."""
        assert self.rag_engine.document_processor is not None
        assert self.rag_engine.vector_store is not None
        assert self.rag_engine.response_generator is not None
        assert len(self.rag_engine.conversation_history) == 0
    
    def test_ingest_document_success(self):
        """Test successful document ingestion."""
        # Mock successful processing
        mock_chunks = [
            DocumentChunk(
                content="Test content",
                metadata={},
                chunk_index=0,
                source_file="test.pdf"
            )
        ]
        mock_metadata = Mock()
        mock_metadata.dict.return_value = {"filename": "test.pdf"}
        mock_metadata.word_count = 100
        mock_metadata.char_count = 500
        mock_metadata.page_count = 1
        
        self.rag_engine.document_processor.process_file.return_value = (mock_chunks, mock_metadata)
        self.rag_engine.vector_store.add_chunks.return_value = [0]
        
        result = self.rag_engine.ingest_document("test.pdf")
        
        assert result["success"] is True
        assert result["chunks_created"] == 1
        assert result["chunk_ids"] == [0]
    
    def test_ingest_document_failure(self):
        """Test failed document ingestion."""
        # Mock processing failure
        self.rag_engine.document_processor.process_file.side_effect = Exception("Processing failed")
        
        result = self.rag_engine.ingest_document("test.pdf")
        
        assert result["success"] is False
        assert "Processing failed" in result["error"]
        assert result["chunks_created"] == 0
    
    def test_ingest_document_no_content(self):
        """Test document ingestion with no content."""
        # Mock empty processing result
        self.rag_engine.document_processor.process_file.return_value = ([], Mock())
        
        result = self.rag_engine.ingest_document("test.pdf")
        
        assert result["success"] is False
        assert "No content extracted" in result["error"]
        assert result["chunks_created"] == 0
    
    def test_query_success(self):
        """Test successful query processing."""
        # Mock vector store search
        mock_chunk = DocumentChunk(
            content="Test content about AI",
            metadata={"source_document": "ai.pdf", "page_number": 1},
            chunk_index=0,
            source_file="ai.pdf",
            quality_score=0.8
        )
        
        self.rag_engine.vector_store.hybrid_search.return_value = [(mock_chunk, 0.9)]
        
        # Mock response generation
        self.rag_engine.response_generator.generate_response.return_value = (
            "AI is artificial intelligence.", 0.85
        )
        
        result = self.rag_engine.query("What is AI?")
        
        assert result["success"] is True
        assert result["answer"] == "AI is artificial intelligence."
        assert result["confidence"] == 0.85
        assert len(result["sources"]) == 1
        assert result["sources"][0]["document"] == "ai.pdf"
    
    def test_query_no_results(self):
        """Test query with no search results."""
        # Mock empty search results
        self.rag_engine.vector_store.hybrid_search.return_value = []
        
        result = self.rag_engine.query("What is AI?")
        
        assert result["success"] is True
        assert "couldn't find any relevant information" in result["answer"]
        assert result["sources"] == []
        assert result["confidence"] == 0.0
    
    def test_query_failure(self):
        """Test query processing failure."""
        # Mock search failure
        self.rag_engine.vector_store.hybrid_search.side_effect = Exception("Search failed")
        
        result = self.rag_engine.query("What is AI?")
        
        assert result["success"] is False
        assert "Search failed" in result["error"]
    
    def test_clear_conversation_history(self):
        """Test clearing conversation history."""
        # Add some history
        self.rag_engine.conversation_history = [Mock(), Mock()]
        
        self.rag_engine.clear_conversation_history()
        
        assert len(self.rag_engine.conversation_history) == 0
    
    def test_get_conversation_history(self):
        """Test getting conversation history."""
        # Mock conversation turn
        mock_turn = Mock()
        mock_turn.dict.return_value = {"question": "Test?", "answer": "Test answer"}
        self.rag_engine.conversation_history = [mock_turn]
        
        history = self.rag_engine.get_conversation_history()
        
        assert len(history) == 1
        assert history[0]["question"] == "Test?"
    
    def test_get_system_stats(self):
        """Test getting system statistics."""
        # Mock vector store stats
        self.rag_engine.vector_store.get_statistics.return_value = {
            "total_chunks": 100,
            "total_documents": 10
        }
        
        stats = self.rag_engine.get_system_stats()
        
        assert "vector_store" in stats
        assert "performance" in stats
        assert "conversation" in stats
        assert "configuration" in stats
    
    @patch('requests.get')
    def test_test_ollama_connection_success(self, mock_get):
        """Test successful Ollama connection test."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "models": [{"name": "test-llm"}, {"name": "other-model"}]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = self.rag_engine.test_ollama_connection()
        
        assert result["success"] is True
        assert result["connected"] is True
        assert "test-llm" in result["available_models"]
    
    @patch('requests.get')
    def test_test_ollama_connection_failure(self, mock_get):
        """Test failed Ollama connection test."""
        mock_get.side_effect = requests.RequestException("Connection failed")
        
        result = self.rag_engine.test_ollama_connection()
        
        assert result["success"] is False
        assert result["connected"] is False
        assert "Connection failed" in result["error"]
    
    def test_ingest_directory_success(self):
        """Test successful directory ingestion."""
        # Mock successful processing
        mock_chunks = [Mock(), Mock()]
        mock_metadata = [Mock(), Mock()]
        for meta in mock_metadata:
            meta.dict.return_value = {"filename": "test.pdf"}
            meta.word_count = 100
            meta.char_count = 500
            meta.page_count = 1
        
        self.rag_engine.document_processor.process_directory.return_value = (mock_chunks, mock_metadata)
        self.rag_engine.vector_store.add_chunks.return_value = [0, 1]
        
        result = self.rag_engine.ingest_directory("/test/dir")
        
        assert result["success"] is True
        assert result["documents_processed"] == 2
        assert result["chunks_created"] == 2
    
    def test_ingest_directory_no_content(self):
        """Test directory ingestion with no content."""
        self.rag_engine.document_processor.process_directory.return_value = ([], [])
        
        result = self.rag_engine.ingest_directory("/test/dir")
        
        assert result["success"] is False
        assert "No content extracted" in result["error"]


class TestConversationTurn:
    """Test cases for ConversationTurn model."""
    
    def test_valid_conversation_turn(self):
        """Test creating valid conversation turn."""
        turn = ConversationTurn(
            question="What is AI?",
            answer="AI is artificial intelligence.",
            sources=[{"document": "ai.pdf"}],
            timestamp="2023-01-01T00:00:00",
            response_time=1.5,
            confidence_score=0.8
        )
        
        assert turn.question == "What is AI?"
        assert turn.answer == "AI is artificial intelligence."
        assert turn.response_time == 1.5
        assert turn.confidence_score == 0.8