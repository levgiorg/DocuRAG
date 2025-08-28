"""Tests for vector store module."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.document_processor import DocumentChunk
from src.vector_store import VectorStore


class TestVectorStore:
    """Test cases for VectorStore."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.temp_dir = temp_dir
            # Use a mock embedding model for faster testing
            with patch('src.vector_store.SentenceTransformer') as mock_st:
                mock_model = Mock()
                mock_model.get_sentence_embedding_dimension.return_value = 384
                mock_model.encode.return_value = np.random.rand(1, 384).astype(np.float32)
                mock_st.return_value = mock_model
                
                self.vector_store = VectorStore(
                    embedding_model="test-model",
                    store_path=temp_dir
                )
    
    def test_init(self):
        """Test VectorStore initialization."""
        assert self.vector_store.dimension == 384
        assert self.vector_store.index_type == "flat"
        assert self.vector_store.index.ntotal == 0
    
    def test_create_flat_index(self):
        """Test flat index creation."""
        index = self.vector_store._create_index()
        assert index.d == 384  # Dimension should match
    
    def test_create_ivf_index(self):
        """Test IVF index creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('src.vector_store.SentenceTransformer') as mock_st:
                mock_model = Mock()
                mock_model.get_sentence_embedding_dimension.return_value = 384
                mock_st.return_value = mock_model
                
                vector_store = VectorStore(
                    embedding_model="test-model",
                    index_type="ivf",
                    store_path=temp_dir
                )
                assert vector_store.index.ntotal == 0
    
    def test_create_hnsw_index(self):
        """Test HNSW index creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('src.vector_store.SentenceTransformer') as mock_st:
                mock_model = Mock()
                mock_model.get_sentence_embedding_dimension.return_value = 384
                mock_st.return_value = mock_model
                
                vector_store = VectorStore(
                    embedding_model="test-model",
                    index_type="hnsw",
                    store_path=temp_dir
                )
                assert vector_store.index.ntotal == 0
    
    def test_unsupported_index_type(self):
        """Test unsupported index type raises error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ValueError, match="Unsupported index type"):
                VectorStore(
                    embedding_model="test-model",
                    index_type="unsupported",
                    store_path=temp_dir
                )
    
    def test_add_empty_chunks(self):
        """Test adding empty chunks list."""
        result = self.vector_store.add_chunks([])
        assert result == []
        assert self.vector_store.index.ntotal == 0
    
    def test_add_chunks(self):
        """Test adding chunks to vector store."""
        # Create test chunks
        chunks = [
            DocumentChunk(
                content="Test content 1",
                metadata={"source": "test1.pdf"},
                chunk_index=0,
                source_file="test1.pdf"
            ),
            DocumentChunk(
                content="Test content 2",
                metadata={"source": "test2.pdf"},
                chunk_index=1,
                source_file="test2.pdf"
            )
        ]
        
        # Mock embedding generation
        mock_embeddings = np.random.rand(2, 384).astype(np.float32)
        self.vector_store.embedding_model.encode.return_value = mock_embeddings
        
        chunk_ids = self.vector_store.add_chunks(chunks)
        
        assert len(chunk_ids) == 2
        assert self.vector_store.index.ntotal == 2
        assert len(self.vector_store.chunk_metadata) == 2
        assert len(self.vector_store.chunk_content) == 2
    
    def test_generate_embeddings(self):
        """Test embedding generation."""
        texts = ["Test text 1", "Test text 2"]
        mock_embeddings = np.random.rand(2, 384).astype(np.float32)
        self.vector_store.embedding_model.encode.return_value = mock_embeddings
        
        embeddings = self.vector_store._generate_embeddings(texts)
        
        assert embeddings.shape == (2, 384)
        assert embeddings.dtype == np.float32
    
    def test_distance_to_similarity(self):
        """Test distance to similarity conversion."""
        # Small distance should give high similarity
        high_sim = self.vector_store._distance_to_similarity(0.1)
        
        # Large distance should give low similarity
        low_sim = self.vector_store._distance_to_similarity(10.0)
        
        assert high_sim > low_sim
        assert 0.0 <= low_sim <= 1.0
        assert 0.0 <= high_sim <= 1.0
    
    def test_search_empty_store(self):
        """Test search on empty vector store."""
        results = self.vector_store.search("test query")
        assert results == []
    
    def test_search_with_results(self):
        """Test search with actual results."""
        # Add chunks first
        chunks = [
            DocumentChunk(
                content="Test content about AI",
                metadata={"source": "ai.pdf"},
                chunk_index=0,
                source_file="ai.pdf"
            )
        ]
        
        # Mock embeddings
        add_embeddings = np.random.rand(1, 384).astype(np.float32)
        query_embeddings = np.random.rand(1, 384).astype(np.float32)
        
        self.vector_store.embedding_model.encode.side_effect = [add_embeddings, query_embeddings]
        
        # Add chunks
        self.vector_store.add_chunks(chunks)
        
        # Search
        results = self.vector_store.search("AI query", top_k=1)
        
        # Should return at least some results (depending on similarity threshold)
        assert isinstance(results, list)
    
    def test_matches_metadata_filter(self):
        """Test metadata filtering."""
        # Add chunk with metadata
        chunk_id = 0
        self.vector_store.chunk_metadata[chunk_id] = {
            "source": "test.pdf",
            "page": 1,
            "author": "Test Author"
        }
        
        # Test matching filter
        assert self.vector_store._matches_metadata_filter(
            chunk_id, {"source": "test.pdf"}
        )
        
        # Test non-matching filter
        assert not self.vector_store._matches_metadata_filter(
            chunk_id, {"source": "other.pdf"}
        )
        
        # Test list filter
        assert self.vector_store._matches_metadata_filter(
            chunk_id, {"source": ["test.pdf", "other.pdf"]}
        )
    
    def test_reconstruct_chunk(self):
        """Test chunk reconstruction."""
        chunk_id = 0
        original_content = "Test content"
        original_metadata = {"source": "test.pdf"}
        
        # Store chunk data
        self.vector_store.chunk_content[chunk_id] = original_content
        self.vector_store.chunk_metadata[chunk_id] = {
            **original_metadata,
            "chunk_index": 0,
            "source_file": "test.pdf"
        }
        
        # Reconstruct chunk
        reconstructed = self.vector_store._reconstruct_chunk(chunk_id)
        
        assert reconstructed is not None
        assert reconstructed.content == original_content
        assert reconstructed.chunk_index == 0
    
    def test_reconstruct_chunk_missing(self):
        """Test reconstruction of missing chunk."""
        result = self.vector_store._reconstruct_chunk(999)
        assert result is None
    
    def test_keyword_search(self):
        """Test keyword search functionality."""
        # Add test content
        self.vector_store.chunk_content = {
            0: "This document discusses machine learning algorithms",
            1: "Python programming is very useful for data science",
            2: "The weather today is quite nice"
        }
        
        self.vector_store.chunk_metadata = {
            0: {"source": "ml.pdf", "chunk_index": 0, "source_file": "ml.pdf"},
            1: {"source": "python.pdf", "chunk_index": 0, "source_file": "python.pdf"},
            2: {"source": "weather.pdf", "chunk_index": 0, "source_file": "weather.pdf"}
        }
        
        results = self.vector_store._keyword_search("machine learning", top_k=5)
        
        # Should find the ML document
        assert len(results) > 0
        found_ml = any("machine learning" in chunk.content.lower() for chunk, score in results)
        assert found_ml
    
    def test_hybrid_search(self):
        """Test hybrid search combining semantic and keyword."""
        # Add test content
        chunks = [
            DocumentChunk(
                content="Machine learning algorithms are powerful",
                metadata={"source": "ml.pdf"},
                chunk_index=0,
                source_file="ml.pdf"
            )
        ]
        
        # Mock embeddings
        embeddings = np.random.rand(1, 384).astype(np.float32)
        self.vector_store.embedding_model.encode.return_value = embeddings
        
        # Add chunks
        self.vector_store.add_chunks(chunks)
        
        # Perform hybrid search
        results = self.vector_store.hybrid_search(
            "machine learning", 
            top_k=1,
            semantic_weight=0.7,
            keyword_weight=0.3
        )
        
        assert isinstance(results, list)
    
    def test_save_and_load(self):
        """Test saving and loading vector store."""
        # Add some test data
        chunks = [
            DocumentChunk(
                content="Test content",
                metadata={"source": "test.pdf"},
                chunk_index=0,
                source_file="test.pdf"
            )
        ]
        
        embeddings = np.random.rand(1, 384).astype(np.float32)
        self.vector_store.embedding_model.encode.return_value = embeddings
        self.vector_store.add_chunks(chunks)
        
        # Save
        self.vector_store.save("test_index")
        
        # Verify files exist
        index_file = Path(self.temp_dir) / "test_index.faiss"
        metadata_file = Path(self.temp_dir) / "test_index_metadata.pkl"
        
        assert index_file.exists()
        assert metadata_file.exists()
        
        # Create new vector store and load
        with patch('src.vector_store.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_st.return_value = mock_model
            
            new_store = VectorStore(
                embedding_model="test-model",
                store_path=self.temp_dir
            )
            
            success = new_store.load("test_index")
            assert success
            assert new_store.index.ntotal == 1
            assert len(new_store.chunk_content) == 1
    
    def test_load_nonexistent(self):
        """Test loading non-existent index."""
        success = self.vector_store.load("nonexistent")
        assert not success
    
    def test_clear(self):
        """Test clearing vector store."""
        # Add some data first
        chunks = [
            DocumentChunk(
                content="Test content",
                metadata={"source": "test.pdf"},
                chunk_index=0,
                source_file="test.pdf"
            )
        ]
        
        embeddings = np.random.rand(1, 384).astype(np.float32)
        self.vector_store.embedding_model.encode.return_value = embeddings
        self.vector_store.add_chunks(chunks)
        
        assert self.vector_store.index.ntotal == 1
        
        # Clear
        self.vector_store.clear()
        
        assert self.vector_store.index.ntotal == 0
        assert len(self.vector_store.chunk_metadata) == 0
        assert len(self.vector_store.chunk_content) == 0
    
    def test_get_statistics(self):
        """Test getting vector store statistics."""
        stats = self.vector_store.get_statistics()
        
        assert "total_chunks" in stats
        assert "embedding_dimension" in stats
        assert "index_type" in stats
        assert "embedding_model" in stats
        assert stats["total_chunks"] == 0
        assert stats["embedding_dimension"] == 384
    
    def test_add_document_metadata(self):
        """Test adding document metadata."""
        doc_metadata = {
            "file_hash": "test_hash",
            "filename": "test.pdf",
            "page_count": 5
        }
        
        self.vector_store.add_document_metadata(doc_metadata)
        
        assert "test_hash" in self.vector_store.document_metadata
        assert self.vector_store.document_metadata["test_hash"]["page_count"] == 5
    
    def test_remove_document(self):
        """Test removing document chunks."""
        # Add chunks with same document hash
        chunks = [
            DocumentChunk(
                content="Test content 1",
                metadata={"source": "test.pdf", "document_hash": "test_hash"},
                chunk_index=0,
                source_file="test.pdf"
            ),
            DocumentChunk(
                content="Test content 2",
                metadata={"source": "other.pdf", "document_hash": "other_hash"},
                chunk_index=0,
                source_file="other.pdf"
            )
        ]
        
        embeddings = np.random.rand(2, 384).astype(np.float32)
        self.vector_store.embedding_model.encode.return_value = embeddings
        self.vector_store.add_chunks(chunks)
        
        # Add document metadata
        self.vector_store.document_metadata["test_hash"] = {"filename": "test.pdf"}
        
        # Remove one document
        removed_count = self.vector_store.remove_document("test_hash")
        
        assert removed_count == 1
        assert self.vector_store.index.ntotal == 1  # Should have 1 chunk remaining
        assert "test_hash" not in self.vector_store.document_metadata