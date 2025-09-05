"""FAISS-based vector store for DocuRAG system.

Handles document embeddings, similarity search, and persistent storage.
"""

import os
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

try:
    import faiss
    from sentence_transformers import SentenceTransformer
except ImportError:
    logger.error("Required packages not installed. Run: pip install faiss-cpu sentence-transformers")
    raise

from .document_processor import DocumentChunk
from .utils import get_device


class VectorStore:
    """FAISS-based vector store for document chunks."""
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2", store_path: str = "data/vector_store"):
        """Initialize vector store.
        
        Args:
            embedding_model: HuggingFace embedding model name
            store_path: Path to store vector index and metadata
        """
        self.embedding_model_name = embedding_model
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        
        # Device setup
        self.device = get_device()
        
        # Load embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model, device=str(self.device))
        
        # Get embedding dimension
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")
        
        # Initialize FAISS index
        self.index = self._create_index()
        
        # Metadata storage
        self.chunk_metadata: List[Dict[str, Any]] = []
        self.document_metadata: List[Dict[str, Any]] = []
        
        # Load existing data
        self.load()
    
    def _create_index(self) -> faiss.Index:
        """Create FAISS index."""
        # Use flat index for simplicity and accuracy
        index = faiss.IndexFlatL2(self.embedding_dim)
        logger.info(f"Created flat FAISS index with dimension {self.embedding_dim}")
        return index
    
    def add_chunks(self, chunks: List[DocumentChunk]) -> List[int]:
        """Add document chunks to vector store.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            List[int]: Chunk IDs
        """
        if not chunks:
            return []
        
        # Extract text for embedding
        texts = [chunk.content for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Add to FAISS index
        start_id = self.index.ntotal
        self.index.add(embeddings.astype(np.float32))
        
        # Store metadata
        chunk_ids = []
        for i, chunk in enumerate(chunks):
            chunk_id = start_id + i
            chunk_ids.append(chunk_id)
            
            metadata = {
                "chunk_id": chunk_id,
                "content": chunk.content,
                "source_file": chunk.source_file,
                "chunk_index": chunk.chunk_index,
                "quality_score": chunk.quality_score,
                "metadata": chunk.metadata
            }
            self.chunk_metadata.append(metadata)
        
        logger.info(f"Added {len(chunks)} chunks to vector store")
        return chunk_ids
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """Search for similar chunks.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of (chunk, similarity_score) tuples
        """
        if self.index.ntotal == 0:
            logger.warning("Vector store is empty")
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Search FAISS index
        distances, indices = self.index.search(query_embedding.astype(np.float32), top_k)
        
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunk_metadata):
                metadata = self.chunk_metadata[idx]
                
                # Convert back to DocumentChunk
                chunk = DocumentChunk(
                    content=metadata["content"],
                    source_file=metadata["source_file"],
                    chunk_index=metadata["chunk_index"],
                    quality_score=metadata["quality_score"],
                    metadata=metadata["metadata"]
                )
                
                # Convert L2 distance to similarity score
                similarity_score = 1.0 / (1.0 + distance)
                results.append((chunk, similarity_score))
        
        logger.info(f"Found {len(results)} chunks matching query")
        return results
    
    def hybrid_search(self, query: str, top_k: int = 5, metadata_filter: Optional[Dict] = None) -> List[Tuple[DocumentChunk, float]]:
        """Perform hybrid search with optional metadata filtering.
        
        Args:
            query: Search query
            top_k: Number of results
            metadata_filter: Optional metadata filters
            
        Returns:
            List of (chunk, score) tuples
        """
        # For now, just do semantic search
        # Can be enhanced with keyword matching later
        results = self.search(query, top_k * 2)  # Get more for filtering
        
        # Apply metadata filters if provided
        if metadata_filter:
            filtered_results = []
            for chunk, score in results:
                match = True
                for key, value in metadata_filter.items():
                    if key in chunk.metadata and chunk.metadata[key] != value:
                        match = False
                        break
                if match:
                    filtered_results.append((chunk, score))
            results = filtered_results
        
        return results[:top_k]
    
    def add_document_metadata(self, metadata: Dict[str, Any]):
        """Add document metadata.
        
        Args:
            metadata: Document metadata
        """
        self.document_metadata.append(metadata)
    
    def get_document_list(self) -> List[str]:
        """Get list of document names.
        
        Returns:
            List[str]: Document names
        """
        return [meta.get("source_document", "Unknown") for meta in self.document_metadata]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics.
        
        Returns:
            Dict[str, Any]: Statistics
        """
        return {
            "total_chunks": self.index.ntotal,
            "total_documents": len(self.document_metadata),
            "embedding_model": self.embedding_model_name,
            "embedding_dimension": self.embedding_dim,
            "device": str(self.device)
        }
    
    def save(self):
        """Save vector store to disk."""
        try:
            # Save FAISS index
            index_path = self.store_path / "faiss_index.bin"
            faiss.write_index(self.index, str(index_path))
            
            # Save metadata
            metadata_path = self.store_path / "metadata.pkl"
            with open(metadata_path, "wb") as f:
                pickle.dump({
                    "chunk_metadata": self.chunk_metadata,
                    "document_metadata": self.document_metadata,
                    "embedding_model": self.embedding_model_name,
                    "embedding_dim": self.embedding_dim
                }, f)
            
            logger.info(f"Saved vector store to {self.store_path}")
            
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")
    
    def load(self):
        """Load vector store from disk."""
        try:
            index_path = self.store_path / "faiss_index.bin"
            metadata_path = self.store_path / "metadata.pkl"
            
            if index_path.exists() and metadata_path.exists():
                # Load FAISS index
                self.index = faiss.read_index(str(index_path))
                
                # Load metadata
                with open(metadata_path, "rb") as f:
                    data = pickle.load(f)
                    self.chunk_metadata = data["chunk_metadata"]
                    self.document_metadata = data["document_metadata"]
                
                logger.info(f"Loaded vector store with {self.index.ntotal} chunks")
            else:
                logger.info("No existing vector store found, starting fresh")
                
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            # Reset on error
            self.index = self._create_index()
            self.chunk_metadata = []
            self.document_metadata = []
    
    def clear(self):
        """Clear all data from vector store."""
        self.index = self._create_index()
        self.chunk_metadata = []
        self.document_metadata = []
        logger.info("Cleared vector store")