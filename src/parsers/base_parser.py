"""Base parser for document parsing"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ParsedDocument:
    """Result of document parsing"""
    text: str                          # Full document text
    sections: Dict[str, str]           # Detected sections {"abstract": "...", "methods": "..."}
    metadata: Dict[str, Any]           # Document metadata (title, authors, etc.)
    word_count: int                    # Total word count
    page_count: Optional[int] = None   # Number of pages (if applicable)
    parse_time: Optional[float] = None # Parsing time in seconds
    imrad_sections: Optional[Dict[str, str]] = None  # IMRAD structured sections
    _spacy_doc: Optional[Any] = None   # Private: spaCy Doc object with full linguistic annotations

    def get_section(self, section_name: str) -> Optional[str]:
        """Get IMRAD section text by name (case-insensitive)"""
        if not self.imrad_sections:
            return None
        section_name_lower = section_name.lower()
        for key, value in self.imrad_sections.items():
            if key.lower() == section_name_lower:
                return value
        return None

    def get_sentences(self) -> list:
        """
        Get list of sentences as strings

        Returns:
            List of sentence strings
        """
        if self._spacy_doc:
            return [sent.text.strip() for sent in self._spacy_doc.sents]
        return []


class BaseParser(ABC):
    """Abstract base class for all document parsers"""

    def __init__(self):
        self.name = self.__class__.__name__

    @abstractmethod
    def parse(self, file_path: str) -> ParsedDocument:
        """
        Parse document from file path

        Args:
            file_path: Path to document file

        Returns:
            ParsedDocument with extracted text, sections, and metadata

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported
        """
        pass

    @abstractmethod
    def supports_format(self, file_extension: str) -> bool:
        """
        Check if parser supports given file format

        Args:
            file_extension: File extension (e.g., '.pdf', '.txt')

        Returns:
            True if format is supported
        """
        pass

    def _count_words(self, text: str) -> int:
        """Count words in text"""
        return len(text.split())

    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text from common artifacts

        Args:
            text: Raw extracted text

        Returns:
            Cleaned text
        """
        # Remove multiple spaces
        text = ' '.join(text.split())

        # Remove page numbers patterns
        import re
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)

        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()
