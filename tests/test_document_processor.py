"""Tests for document processor module."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from src.document_processor import DocumentProcessor, DocumentChunk, DocumentMetadata


class TestDocumentProcessor:
    """Test cases for DocumentProcessor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)
    
    def test_init(self):
        """Test DocumentProcessor initialization."""
        assert self.processor.chunk_size == 500
        assert self.processor.chunk_overlap == 100
        assert '.pdf' in self.processor.supported_formats
        assert '.txt' in self.processor.supported_formats
    
    def test_invalid_overlap(self):
        """Test that chunk_overlap cannot exceed chunk_size."""
        with pytest.raises(ValueError):
            DocumentProcessor(chunk_size=100, chunk_overlap=150)
    
    def test_split_into_sentences(self):
        """Test sentence splitting functionality."""
        text = "This is sentence one. This is sentence two! And this is three?"
        sentences = self.processor._split_into_sentences(text)
        
        assert len(sentences) == 3
        assert "This is sentence one." in sentences[0]
        assert "This is sentence two!" in sentences[1]
        assert "And this is three?" in sentences[2]
    
    def test_get_overlap_text(self):
        """Test overlap text extraction."""
        text = "This is a long text that should be used for testing overlap functionality"
        overlap = self.processor._get_overlap_text(text)
        
        assert len(overlap) <= self.processor.chunk_overlap
        assert overlap in text
    
    @patch('src.document_processor.calculate_file_hash')
    @patch('src.document_processor.Path.stat')
    def test_process_text_file(self, mock_stat, mock_hash):
        """Test processing of text files."""
        mock_stat.return_value.st_size = 1000
        mock_hash.return_value = "test_hash"
        
        # Create a temporary text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document. " * 50)  # Create substantial content
            temp_path = Path(f.name)
        
        try:
            chunks, metadata = self.processor.process_file(temp_path)
            
            assert isinstance(metadata, DocumentMetadata)
            assert metadata.filename == temp_path.name
            assert metadata.file_type == '.txt'
            assert len(chunks) > 0
            
            for chunk in chunks:
                assert isinstance(chunk, DocumentChunk)
                assert len(chunk.content) <= self.processor.chunk_size + 100  # Allow some flexibility
                assert chunk.source_file == str(temp_path)
        
        finally:
            temp_path.unlink()
    
    def test_process_nonexistent_file(self):
        """Test processing of non-existent file."""
        with pytest.raises(FileNotFoundError):
            self.processor.process_file("nonexistent_file.txt")
    
    def test_process_unsupported_format(self):
        """Test processing of unsupported file format."""
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                self.processor.process_file(temp_path)
        finally:
            temp_path.unlink()
    
    def test_create_chunks_small_text(self):
        """Test chunk creation with small text."""
        text = "Short text."
        metadata = DocumentMetadata(
            filename="test.txt",
            file_path="/test.txt",
            file_hash="hash",
            file_size=100,
            file_type=".txt",
            processed_at="2023-01-01T00:00:00",
            word_count=2,
            char_count=11
        )
        
        chunks = self.processor._create_chunks(text, metadata)
        
        assert len(chunks) == 1
        assert chunks[0].content == text
        assert chunks[0].chunk_index == 0
    
    def test_create_chunks_large_text(self):
        """Test chunk creation with large text."""
        text = "This is a sentence. " * 100  # Create text larger than chunk_size
        metadata = DocumentMetadata(
            filename="test.txt",
            file_path="/test.txt",
            file_hash="hash",
            file_size=2000,
            file_type=".txt",
            processed_at="2023-01-01T00:00:00",
            word_count=400,
            char_count=2000
        )
        
        chunks = self.processor._create_chunks(text, metadata)
        
        assert len(chunks) > 1
        
        # Check chunk sizes
        for chunk in chunks[:-1]:  # All but last chunk
            assert len(chunk.content) <= self.processor.chunk_size + 200  # Allow some flexibility
        
        # Check chunk indices
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
    
    def test_calculate_chunk_quality(self):
        """Test chunk quality calculation."""
        # Good quality chunk
        good_chunk = "This is a well-formed sentence with proper punctuation and reasonable length."
        good_score = self.processor._calculate_chunk_quality(good_chunk)
        
        # Poor quality chunk
        poor_chunk = "123 !@# $$$ %%% ^^^ &&&"
        poor_score = self.processor._calculate_chunk_quality(poor_chunk)
        
        # Empty chunk
        empty_score = self.processor._calculate_chunk_quality("")
        
        assert good_score > poor_score
        assert empty_score == 0.0
        assert 0.0 <= good_score <= 1.0
        assert 0.0 <= poor_score <= 1.0
    
    @patch('src.document_processor.PyPDF2')
    def test_extract_pdf_content_not_available(self, mock_pypdf2):
        """Test PDF processing when PyPDF2 is not available."""
        mock_pypdf2 = None
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            with pytest.raises(ImportError, match="PyPDF2 is required"):
                self.processor._extract_pdf_content(temp_path)
        finally:
            temp_path.unlink()
    
    @patch('src.document_processor.docx')
    def test_extract_docx_content_not_available(self, mock_docx):
        """Test DOCX processing when python-docx is not available."""
        mock_docx = None
        
        with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            with pytest.raises(ImportError, match="python-docx is required"):
                self.processor._extract_docx_content(temp_path)
        finally:
            temp_path.unlink()
    
    def test_process_directory_nonexistent(self):
        """Test processing of non-existent directory."""
        with pytest.raises(FileNotFoundError):
            self.processor.process_directory("nonexistent_directory")
    
    @patch('src.document_processor.calculate_file_hash')
    @patch('src.document_processor.Path.stat')
    def test_process_directory(self, mock_stat, mock_hash):
        """Test directory processing."""
        mock_stat.return_value.st_size = 1000
        mock_hash.return_value = "test_hash"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            (temp_path / "test1.txt").write_text("Test content 1")
            (temp_path / "test2.txt").write_text("Test content 2")
            (temp_path / "ignored.xyz").write_text("Should be ignored")
            
            all_chunks, all_metadata = self.processor.process_directory(temp_path)
            
            assert len(all_metadata) == 2  # Only .txt files processed
            assert len(all_chunks) >= 2  # At least one chunk per file
            
            filenames = [meta.filename for meta in all_metadata]
            assert "test1.txt" in filenames
            assert "test2.txt" in filenames


class TestDocumentMetadata:
    """Test cases for DocumentMetadata model."""
    
    def test_valid_metadata(self):
        """Test creating valid metadata."""
        metadata = DocumentMetadata(
            filename="test.pdf",
            file_path="/path/to/test.pdf",
            file_hash="abc123",
            file_size=1024,
            file_type=".pdf",
            processed_at="2023-01-01T00:00:00",
            word_count=100,
            char_count=500
        )
        
        assert metadata.filename == "test.pdf"
        assert metadata.file_size == 1024
        assert metadata.word_count == 100
    
    def test_missing_required_field(self):
        """Test metadata creation with missing required field."""
        with pytest.raises(ValidationError):
            DocumentMetadata(
                filename="test.pdf",
                # Missing required fields
            )


class TestDocumentChunk:
    """Test cases for DocumentChunk model."""
    
    def test_valid_chunk(self):
        """Test creating valid chunk."""
        chunk = DocumentChunk(
            content="Test content",
            metadata={"source": "test.pdf"},
            chunk_index=0,
            source_file="/path/to/test.pdf",
            quality_score=0.8
        )
        
        assert chunk.content == "Test content"
        assert chunk.chunk_index == 0
        assert chunk.quality_score == 0.8
    
    def test_chunk_with_optional_fields(self):
        """Test chunk with optional fields."""
        chunk = DocumentChunk(
            content="Test content",
            metadata={"source": "test.pdf"},
            chunk_index=1,
            source_file="/path/to/test.pdf",
            page_number=5,
            start_char=100,
            end_char=200
        )
        
        assert chunk.page_number == 5
        assert chunk.start_char == 100
        assert chunk.end_char == 200