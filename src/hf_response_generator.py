"""HuggingFace response generator for DocuRAG system.

Replaces Ollama with direct HuggingFace transformers integration for better performance
and compatibility with job assignment requirements.
"""

import torch
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
except ImportError:
    logger.error("transformers not installed. Run: pip install transformers torch accelerate")
    raise

from .document_processor import DocumentChunk


class HuggingFaceResponseGenerator:
    """Generate responses using HuggingFace transformers."""
    
    def __init__(self, model_name: str = "google/gemma-2b", max_length: int = 1024, device: str = "auto"):
        """Initialize HuggingFace response generator.
        
        Args:
            model_name: HuggingFace model name
            max_length: Maximum token length for generation
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        self.model_name = model_name
        self.max_length = max_length
        
        # Determine device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Loading model: {model_name}")
        
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None
            )
            
            if self.device.type == "cpu":
                self.model = self.model.to(self.device)
            
            # Create pipeline for easier generation
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device if self.device.type == "cpu" else None,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
            )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def generate_response(self, 
                         query: str, 
                         context_chunks: List[DocumentChunk],
                         conversation_history: Optional[List] = None,
                         highlighted_contexts: Optional[List[str]] = None) -> Tuple[str, float, List[str]]:
        """Generate response using retrieved context.
        
        Args:
            query: User question
            context_chunks: Retrieved document chunks
            conversation_history: Previous conversation turns
            highlighted_contexts: Pre-highlighted contexts (optional)
            
        Returns:
            Tuple[str, float, List[str]]: Generated response, confidence score, and highlighted contexts
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
            response_text = self._generate_with_hf(prompt)
            confidence_score = self._calculate_confidence(response_text, context_chunks)
            
            # Generate highlighted contexts if not provided
            if highlighted_contexts is None:
                highlighted_contexts = self._generate_highlighted_contexts(response_text, context_chunks)
            
            return response_text, confidence_score, highlighted_contexts
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return "I apologize, but I encountered an error while generating the response. Please try again.", 0.0, []
    
    def _generate_with_hf(self, prompt: str) -> str:
        """Generate text using HuggingFace pipeline.
        
        Args:
            prompt: Input prompt
            
        Returns:
            str: Generated text
        """
        try:
            # Generate response
            outputs = self.pipeline(
                prompt,
                max_length=min(len(self.tokenizer.encode(prompt)) + 512, self.max_length),
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                truncation=True
            )
            
            # Extract generated text (remove the input prompt)
            full_response = outputs[0]['generated_text']
            response = full_response[len(prompt):].strip()
            
            # Clean up response
            response = self._clean_response(response)
            
            return response
            
        except Exception as e:
            logger.error(f"HuggingFace generation failed: {e}")
            raise
    
    def _clean_response(self, response: str) -> str:
        """Clean up generated response.
        
        Args:
            response: Raw generated response
            
        Returns:
            str: Cleaned response
        """
        # Remove common artifacts
        response = response.replace("<|endoftext|>", "")
        response = response.replace("</s>", "")
        response = response.replace("<unk>", "")
        
        # Split by newlines and take first coherent part
        lines = response.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('---'):
                cleaned_lines.append(line)
            elif cleaned_lines:  # Stop at first separator after we have content
                break
        
        return '\n'.join(cleaned_lines).strip()
    
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
    
    def _build_conversation_context(self, history: List) -> str:
        """Build conversation context from history.
        
        Args:
            history: Conversation history
            
        Returns:
            str: Formatted conversation context
        """
        if not history:
            return ""
        
        context_parts = []
        for turn in history[-3:]:  # Last 3 turns
            context_parts.append(f"Previous Question: {turn.question}")
            context_parts.append(f"Previous Answer: {turn.answer}")
        
        return "\n".join(context_parts)
    
    def _create_prompt(self, query: str, context: str, conversation_context: str = "") -> str:
        """Create prompt for the LLM.
        
        Args:
            query: User question
            context: Retrieved context
            conversation_context: Previous conversation
            
        Returns:
            str: Formatted prompt
        """
        prompt_parts = []
        
        if conversation_context:
            prompt_parts.append(f"Conversation History:\n{conversation_context}\n")
        
        prompt_parts.extend([
            "Context Information:",
            context,
            "",
            "Instructions:",
            "- Answer the question using only the provided context",
            "- Be concise and accurate",
            "- If the context doesn't contain relevant information, say so",
            "- Cite sources when possible",
            "",
            f"Question: {query}",
            "",
            "Answer:"
        ])
        
        return "\n".join(prompt_parts)
    
    def _calculate_confidence(self, response: str, context_chunks: List[DocumentChunk]) -> float:
        """Calculate confidence score for the response.
        
        Args:
            response: Generated response
            context_chunks: Retrieved chunks
            
        Returns:
            float: Confidence score between 0 and 1
        """
        if not response or not context_chunks:
            return 0.0
        
        # Simple confidence calculation based on response length and context relevance
        response_length = len(response.split())
        
        # Check if response contains source references
        has_sources = any(word in response.lower() for word in ['source', 'according', 'based on'])
        
        # Base confidence on response length (reasonable responses are 10-200 words)
        length_score = min(response_length / 100.0, 1.0) if response_length >= 10 else 0.3
        
        # Boost confidence if sources are referenced
        source_boost = 0.2 if has_sources else 0.0
        
        # Boost confidence based on number of context chunks
        context_boost = min(len(context_chunks) * 0.1, 0.3)
        
        confidence = min(length_score + source_boost + context_boost, 1.0)
        
        return round(confidence, 2)
    
    def _generate_highlighted_contexts(self, response: str, context_chunks: List[DocumentChunk]) -> List[str]:
        """Generate highlighted contexts showing which parts contributed to the answer.
        
        Args:
            response: Generated response
            context_chunks: Retrieved chunks
            
        Returns:
            List[str]: Highlighted context snippets
        """
        highlighted_contexts = []
        response_words = set(response.lower().split())
        
        for chunk in context_chunks:
            chunk_words = set(chunk.content.lower().split())
            
            # Find overlapping words (simple approach)
            overlap = response_words.intersection(chunk_words)
            
            if len(overlap) >= 3:  # Minimum overlap threshold
                # Highlight overlapping words in the chunk
                highlighted_chunk = chunk.content
                
                for word in overlap:
                    if len(word) > 3:  # Only highlight meaningful words
                        highlighted_chunk = highlighted_chunk.replace(
                            word, f"**{word}**", 
                            1  # Only first occurrence
                        )
                
                source = chunk.metadata.get("source_document", "Unknown")
                page = chunk.metadata.get("page_number")
                
                source_info = f"{source}"
                if page:
                    source_info += f" (Page {page})"
                
                highlighted_contexts.append(f"From {source_info}:\n{highlighted_chunk}")
        
        return highlighted_contexts