"""Vector store implementation using FAISS for DocuRAG system.

Handles vector storage, similarity search, and persistence.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import faiss
import numpy as np
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
from sentence_transformers import SentenceTransformer

from .document_processor import DocumentChunk
from .utils import get_device, timing_decorator, PerformanceMonitor


class VectorStore:
    """FAISS-based vector store for document chunks."""
    
    def __init__(self, 
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 index_type: str = "flat",
                 dimension: Optional[int] = None,
                 store_path: str = "data/vector_store"):
        """Initialize vector store.
        
        Args:
            embedding_model: Name of the sentence transformer model
            index_type: FAISS index type ('flat', 'ivf', 'hnsw')
            dimension: Embedding dimension (auto-detected if None)
            store_path: Path to store index and metadata
        """
        self.embedding_model_name = embedding_model
        self.index_type = index_type
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model
        device = get_device()
        device_str = "cuda" if device.type == "cuda" else "cpu"
        
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model, device=device_str)
        
        # Get embedding dimension
        self.dimension = dimension or self.embedding_model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.dimension}")
        
        # Initialize FAISS index
        self.index = self._create_index()
        
        # Storage for chunk metadata
        self.chunk_metadata: Dict[int, Dict[str, Any]] = {}
        self.chunk_content: Dict[int, str] = {}
        self.document_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        
        # Load existing data if available
        self._load_if_exists()
    
    def _create_index(self) -> faiss.Index:
        """Create FAISS index based on configuration.
        
        Returns:
            faiss.Index: Initialized FAISS index
        """
        if self.index_type == "flat":
            # L2 distance (Euclidean)
            index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "ivf":
            # IVF with flat quantizer
            quantizer = faiss.IndexFlatL2(self.dimension)
            nlist = 100  # Number of clusters
            index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
        elif self.index_type == "hnsw":
            # Hierarchical NSW
            m = 16  # Number of connections
            index = faiss.IndexHNSWFlat(self.dimension, m)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        logger.info(f"Created {self.index_type} FAISS index with dimension {self.dimension}")
        return index
    
    @timing_decorator
    def add_chunks(self, chunks: List[DocumentChunk], 
                   batch_size: int = 32) -> List[int]:
        """Add document chunks to the vector store.
        
        Args:
            chunks: List of document chunks to add
            batch_size: Batch size for embedding generation
            
        Returns:
            List[int]: List of assigned chunk IDs
        """
        if not chunks:
            logger.warning("No chunks provided to add")
            return []
        
        logger.info(f"Adding {len(chunks)} chunks to vector store")
        
        # Extract text content for embedding
        texts = [chunk.content for chunk in chunks]
        
        # Generate embeddings in batches
        self.performance_monitor.start_timer("embedding_generation")
        embeddings = self._generate_embeddings(texts, batch_size)
        self.performance_monitor.end_timer("embedding_generation")
        
        # Add to FAISS index
        self.performance_monitor.start_timer("index_addition")
        start_id = self.index.ntotal
        self.index.add(embeddings)
        self.performance_monitor.end_timer("index_addition")
        
        # Store metadata and content
        chunk_ids = []
        for i, chunk in enumerate(chunks):
            chunk_id = start_id + i
            chunk_ids.append(chunk_id)
            
            # Store chunk metadata
            self.chunk_metadata[chunk_id] = {
                **chunk.metadata,
                "chunk_index": chunk.chunk_index,
                "source_file": chunk.source_file,
                "page_number": chunk.page_number,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "quality_score": chunk.quality_score
            }
            
            # Store chunk content
            self.chunk_content[chunk_id] = chunk.content
        
        logger.info(f"Added {len(chunks)} chunks with IDs {chunk_ids[0]}-{chunk_ids[-1]}")
        return chunk_ids
    
    @timing_decorator
    def add_document_metadata(self, document_metadata: Dict[str, Any]) -> None:
        """Add document-level metadata.
        
        Args:
            document_metadata: Document metadata to store
        """
        doc_key = document_metadata.get("file_hash", document_metadata.get("filename"))
        if doc_key:
            self.document_metadata[doc_key] = document_metadata
            logger.debug(f"Added document metadata for {doc_key}")
    
    @timing_decorator
    def search(self, 
               query: str,
               top_k: int = 5,
               similarity_threshold: float = 0.7,
               metadata_filter: Optional[Dict[str, Any]] = None) -> List[Tuple[DocumentChunk, float]]:
        """Search for similar chunks.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity threshold
            metadata_filter: Optional metadata filters
            
        Returns:
            List[Tuple[DocumentChunk, float]]: List of (chunk, similarity_score) tuples
        """
        if self.index.ntotal == 0:
            logger.warning("Vector store is empty")
            return []
        
        self.performance_monitor.start_timer("query_search")
        
        # Generate query embedding
        query_embedding = self._generate_embeddings([query], batch_size=1)[0:1]
        
        # Search in FAISS index
        # Get more results for filtering
        search_k = min(top_k * 3, self.index.ntotal)
        distances, indices = self.index.search(query_embedding, search_k)
        
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1:  # Invalid index
                continue
            
            # Convert distance to similarity score
            similarity_score = self._distance_to_similarity(distance)
            
            if similarity_score < similarity_threshold:
                continue
            
            # Apply metadata filtering
            if metadata_filter and not self._matches_metadata_filter(idx, metadata_filter):
                continue
            
            # Reconstruct chunk
            chunk = self._reconstruct_chunk(idx)
            if chunk:
                results.append((chunk, similarity_score))
            
            # Stop if we have enough results
            if len(results) >= top_k:
                break
        
        self.performance_monitor.end_timer("query_search")
        
        logger.info(f"Found {len(results)} chunks matching query")
        return results
    
    def hybrid_search(self,
                     query: str,
                     top_k: int = 5,
                     semantic_weight: float = 0.7,
                     keyword_weight: float = 0.3,
                     metadata_filter: Optional[Dict[str, Any]] = None) -> List[Tuple[DocumentChunk, float]]:
        """Perform hybrid search combining semantic and keyword matching.
        
        Args:
            query: Search query
            top_k: Number of results to return
            semantic_weight: Weight for semantic similarity
            keyword_weight: Weight for keyword matching
            metadata_filter: Optional metadata filters
            
        Returns:
            List[Tuple[DocumentChunk, float]]: Ranked results
        """
        # Get semantic search results
        semantic_results = self.search(
            query, top_k=top_k * 2, similarity_threshold=0.0, 
            metadata_filter=metadata_filter
        )
        
        # Perform keyword matching
        keyword_results = self._keyword_search(query, top_k * 2, metadata_filter)
        
        # Combine and rank results
        combined_scores = {}
        
        # Add semantic scores
        for chunk, score in semantic_results:
            chunk_id = self._get_chunk_id_from_chunk(chunk)
            combined_scores[chunk_id] = {"chunk": chunk, "semantic": score, "keyword": 0.0}
        
        # Add keyword scores
        for chunk, score in keyword_results:
            chunk_id = self._get_chunk_id_from_chunk(chunk)
            if chunk_id in combined_scores:
                combined_scores[chunk_id]["keyword"] = score
            else:
                combined_scores[chunk_id] = {"chunk": chunk, "semantic": 0.0, "keyword": score}
        
        # Calculate combined scores
        final_results = []
        for chunk_id, scores in combined_scores.items():
            combined_score = (
                semantic_weight * scores["semantic"] + 
                keyword_weight * scores["keyword"]
            )
            final_results.append((scores["chunk"], combined_score))
        
        # Sort by combined score and return top_k
        final_results.sort(key=lambda x: x[1], reverse=True)
        return final_results[:top_k]
    
    def _keyword_search(self, 
                       query: str, 
                       top_k: int,
                       metadata_filter: Optional[Dict[str, Any]] = None) -> List[Tuple[DocumentChunk, float]]:
        """Perform keyword-based search.
        
        Args:
            query: Search query
            top_k: Number of results to return
            metadata_filter: Optional metadata filters
            
        Returns:
            List[Tuple[DocumentChunk, float]]: Keyword matching results
        """
        query_terms = set(query.lower().split())
        results = []
        
        for chunk_id, content in self.chunk_content.items():
            # Apply metadata filtering
            if metadata_filter and not self._matches_metadata_filter(chunk_id, metadata_filter):
                continue
            
            # Calculate keyword match score
            content_terms = set(content.lower().split())
            intersection = query_terms.intersection(content_terms)
            
            if intersection:
                # Score based on term overlap and frequency
                score = len(intersection) / len(query_terms)
                
                # Boost score for exact phrase matches
                if query.lower() in content.lower():
                    score *= 1.5
                
                chunk = self._reconstruct_chunk(chunk_id)
                if chunk:
                    results.append((chunk, min(score, 1.0)))
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def _generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            np.ndarray: Embeddings matrix
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch_texts, 
                convert_to_numpy=True,
                show_progress_bar=False
            )
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings).astype(np.float32)
    
    def _distance_to_similarity(self, distance: float) -> float:
        """Convert L2 distance to similarity score.
        
        Args:
            distance: L2 distance
            
        Returns:
            float: Similarity score between 0 and 1
        """
        # Convert L2 distance to similarity (higher is better)
        # Using exponential decay function
        return np.exp(-distance / 2)
    
    def _matches_metadata_filter(self, chunk_id: int, metadata_filter: Dict[str, Any]) -> bool:
        """Check if chunk matches metadata filter.
        
        Args:
            chunk_id: Chunk ID to check
            metadata_filter: Filter criteria
            
        Returns:
            bool: True if chunk matches filter
        """
        if chunk_id not in self.chunk_metadata:
            return False
        
        chunk_meta = self.chunk_metadata[chunk_id]
        
        for key, value in metadata_filter.items():
            if key not in chunk_meta:
                return False
            
            if isinstance(value, list):
                if chunk_meta[key] not in value:
                    return False
            else:
                if chunk_meta[key] != value:
                    return False
        
        return True
    
    def _reconstruct_chunk(self, chunk_id: int) -> Optional[DocumentChunk]:
        """Reconstruct DocumentChunk from stored data.
        
        Args:
            chunk_id: Chunk ID
            
        Returns:
            Optional[DocumentChunk]: Reconstructed chunk or None
        """
        if chunk_id not in self.chunk_content or chunk_id not in self.chunk_metadata:
            return None
        
        content = self.chunk_content[chunk_id]
        metadata = self.chunk_metadata[chunk_id]
        
        return DocumentChunk(
            content=content,
            metadata=metadata,
            chunk_index=metadata.get("chunk_index", 0),
            source_file=metadata.get("source_file", ""),
            page_number=metadata.get("page_number"),
            start_char=metadata.get("start_char"),
            end_char=metadata.get("end_char"),
            quality_score=metadata.get("quality_score")
        )
    
    def _get_chunk_id_from_chunk(self, chunk: DocumentChunk) -> Optional[int]:
        """Get chunk ID from chunk object (reverse lookup).
        
        Args:
            chunk: DocumentChunk object
            
        Returns:
            Optional[int]: Chunk ID or None if not found
        """
        for chunk_id, content in self.chunk_content.items():
            if content == chunk.content:
                return chunk_id
        return None
    
    def save(self, index_name: str = "faiss_index") -> None:
        """Save vector store to disk.
        
        Args:
            index_name: Name for the saved index files
        """
        logger.info(f"Saving vector store to {self.store_path}")
        
        try:
            # Save FAISS index
            index_path = self.store_path / f"{index_name}.faiss"
            faiss.write_index(self.index, str(index_path))
            
            # Save metadata and content
            metadata_path = self.store_path / f"{index_name}_metadata.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump({
                    "chunk_metadata": self.chunk_metadata,
                    "chunk_content": self.chunk_content,
                    "document_metadata": self.document_metadata,
                    "embedding_model_name": self.embedding_model_name,
                    "dimension": self.dimension,
                    "index_type": self.index_type
                }, f)
            
            logger.info(f"Vector store saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")
            raise
    
    def load(self, index_name: str = "faiss_index") -> bool:
        """Load vector store from disk.
        
        Args:
            index_name: Name of the saved index files
            
        Returns:
            bool: True if loaded successfully
        """
        index_path = self.store_path / f"{index_name}.faiss"
        metadata_path = self.store_path / f"{index_name}_metadata.pkl"
        
        if not index_path.exists() or not metadata_path.exists():
            logger.info("No existing vector store found")
            return False
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(str(index_path))
            
            # Load metadata and content
            with open(metadata_path, 'rb') as f:
                data = pickle.load(f)
            
            self.chunk_metadata = data["chunk_metadata"]
            self.chunk_content = data["chunk_content"]
            self.document_metadata = data["document_metadata"]
            
            # Verify compatibility
            if data["embedding_model_name"] != self.embedding_model_name:
                logger.warning(
                    f"Loaded model ({data['embedding_model_name']}) differs from "
                    f"current model ({self.embedding_model_name})"
                )
            
            logger.info(f"Loaded vector store with {self.index.ntotal} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            return False
    
    def _load_if_exists(self) -> None:
        """Load existing vector store if available."""
        if self.store_path.exists():
            self.load()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get vector store statistics.
        
        Returns:
            Dict[str, Any]: Statistics about the vector store
        """
        stats = {
            "total_chunks": self.index.ntotal,
            "embedding_dimension": self.dimension,
            "index_type": self.index_type,
            "embedding_model": self.embedding_model_name,
            "total_documents": len(self.document_metadata),
        }
        
        # Add performance metrics
        for operation in ["embedding_generation", "index_addition", "query_search"]:
            operation_stats = self.performance_monitor.get_stats(operation)
            if operation_stats["count"] > 0:
                stats[f"{operation}_avg_time"] = operation_stats["avg"]
                stats[f"{operation}_total_calls"] = operation_stats["count"]
        
        return stats
    
    def clear(self) -> None:
        """Clear all data from vector store."""
        logger.warning("Clearing all data from vector store")
        
        # Recreate empty index
        self.index = self._create_index()
        
        # Clear metadata
        self.chunk_metadata.clear()
        self.chunk_content.clear()
        self.document_metadata.clear()
        
        # Reset performance monitor
        self.performance_monitor.reset()
    
    def remove_document(self, document_hash: str) -> int:
        """Remove all chunks belonging to a specific document.
        
        Args:
            document_hash: Hash of the document to remove
            
        Returns:
            int: Number of chunks removed
        """
        # Find chunks belonging to the document
        chunks_to_remove = []
        for chunk_id, metadata in self.chunk_metadata.items():
            if metadata.get("document_hash") == document_hash:
                chunks_to_remove.append(chunk_id)
        
        if not chunks_to_remove:
            logger.info(f"No chunks found for document {document_hash}")
            return 0
        
        # FAISS doesn't support efficient removal, so we need to rebuild
        logger.warning(f"Removing {len(chunks_to_remove)} chunks requires index rebuild")
        
        # Collect remaining chunks
        remaining_chunks = []
        for chunk_id in range(self.index.ntotal):
            if chunk_id not in chunks_to_remove:
                chunk = self._reconstruct_chunk(chunk_id)
                if chunk:
                    remaining_chunks.append(chunk)
        
        # Rebuild index with remaining chunks
        self.clear()
        if remaining_chunks:
            self.add_chunks(remaining_chunks)
        
        # Remove document metadata
        if document_hash in self.document_metadata:
            del self.document_metadata[document_hash]
        
        logger.info(f"Removed {len(chunks_to_remove)} chunks for document {document_hash}")
        return len(chunks_to_remove)