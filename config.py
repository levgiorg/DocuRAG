"""Configuration management for DocuRAG system."""

import os
from pathlib import Path
from typing import Dict, Any, Optional

import yaml
from pydantic import BaseModel, Field, validator
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class ModelConfig(BaseModel):
    """Model configuration settings."""
    
    # HuggingFace Model Settings (Primary)
    hf_model: str = Field(default="Qwen/Qwen2.5-1.5B-Instruct", description="HuggingFace model name")
    hf_max_length: int = Field(default=1024, description="HuggingFace max token length")
    hf_device: str = Field(default="auto", description="HuggingFace device (auto/cpu/cuda)")
    
    # Legacy Ollama Settings (Fallback)
    llm_model: str = Field(default="gemma2:2b", description="Ollama model name (fallback)")
    ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama API URL")
    
    # Common Settings
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model name"
    )
    max_tokens: int = Field(default=2048, description="Maximum tokens for generation")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Generation temperature")
    device: str = Field(default="auto", description="Device for embeddings (auto/cpu/cuda)")


class RetrievalConfig(BaseModel):
    """Retrieval configuration settings."""
    
    chunk_size: int = Field(default=1000, ge=100, description="Document chunk size")
    chunk_overlap: int = Field(default=200, ge=0, description="Chunk overlap size")
    top_k: int = Field(default=5, ge=1, description="Number of top chunks to retrieve")
    similarity_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Minimum similarity threshold"
    )
    score_threshold: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Minimum retrieval score threshold"
    )
    max_chunks_per_doc: int = Field(default=3, ge=1, description="Max chunks per document")
    rerank_top_k: int = Field(default=10, ge=1, description="Initial retrieval size for reranking")
    enable_metadata_filter: bool = Field(default=True, description="Enable metadata filtering")
    
    @validator('chunk_overlap')
    def validate_overlap(cls, v, values):
        if 'chunk_size' in values and v >= values['chunk_size']:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v


class AppConfig(BaseModel):
    """Application configuration settings."""
    
    app_name: str = Field(default="DocuRAG", description="Application name")
    version: str = Field(default="1.0.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    max_file_size_mb: int = Field(default=50, ge=1, description="Maximum file size in MB")
    allowed_extensions: list[str] = Field(
        default=[".pdf", ".txt", ".docx", ".md"],
        description="Allowed file extensions"
    )
    vector_store_path: str = Field(
        default="data/vector_store", description="Vector store directory"
    )
    documents_path: str = Field(default="data/documents", description="Documents directory")
    enable_conversation_history: bool = Field(
        default=True, description="Enable conversation history"
    )
    max_conversation_turns: int = Field(
        default=10, ge=1, description="Maximum conversation turns to keep"
    )


class DocuRAGConfig:
    """Main configuration manager for DocuRAG system."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """Initialize configuration manager.
        
        Args:
            config_dir: Directory containing config files. Defaults to 'configs/'.
        """
        self.config_dir = Path(config_dir or "configs")
        self.model_config = ModelConfig()
        self.retrieval_config = RetrievalConfig()
        self.app_config = AppConfig()
        
        self._load_configs()
        self._setup_logging()
    
    def _load_configs(self) -> None:
        """Load configuration from YAML files."""
        configs = {
            "model_config.yaml": self.model_config,
            "retrieval_config.yaml": self.retrieval_config,
            "app_config.yaml": self.app_config
        }
        
        for filename, config_obj in configs.items():
            config_path = self.config_dir / filename
            if config_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config_data = yaml.safe_load(f)
                    
                    # Update config object with loaded data
                    if config_data:
                        for key, value in config_data.items():
                            if hasattr(config_obj, key):
                                setattr(config_obj, key, value)
                    
                    logger.info(f"Loaded configuration from {filename}")
                except Exception as e:
                    logger.warning(f"Failed to load {filename}: {e}")
            else:
                logger.info(f"Config file {filename} not found, using defaults")
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        try:
            from loguru import logger as loguru_logger
            logger = loguru_logger
        except ImportError:
            import logging
            logger = logging.getLogger(__name__)
            return  # Skip loguru-specific setup
        
        # Remove default handler
        logger.remove()
        
        # Add console handler
        logger.add(
            lambda msg: print(msg, end=""),
            level=self.app_config.log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        )
        
        # Add file handler
        log_path = Path("data/logs") / "docurag.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            str(log_path),
            level=self.app_config.log_level,
            rotation="10 MB",
            retention="7 days",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
        )
    
    def save_configs(self) -> None:
        """Save current configurations to YAML files."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        configs = {
            "model_config.yaml": self.model_config.dict(),
            "retrieval_config.yaml": self.retrieval_config.dict(),
            "app_config.yaml": self.app_config.dict()
        }
        
        for filename, config_data in configs.items():
            config_path = self.config_dir / filename
            try:
                with open(config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config_data, f, default_flow_style=False, indent=2)
                logger.info(f"Saved configuration to {filename}")
            except Exception as e:
                logger.error(f"Failed to save {filename}: {e}")
    
    def get_env_var(self, key: str, default: Any = None) -> Any:
        """Get environment variable with optional default."""
        return os.getenv(key, default)
    
    def validate_paths(self) -> None:
        """Validate and create necessary directories."""
        paths = [
            self.app_config.vector_store_path,
            self.app_config.documents_path,
            "data/logs"
        ]
        
        for path_str in paths:
            path = Path(path_str)
            path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {path}")
    
    @property
    def is_debug(self) -> bool:
        """Check if debug mode is enabled."""
        return self.app_config.debug or self.get_env_var("DEBUG", "false").lower() == "true"


# Global configuration instance
config = DocuRAGConfig()