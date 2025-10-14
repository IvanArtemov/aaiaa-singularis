"""PDF parser using PyMuPDF (fitz) and pdfplumber"""

import re
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from .base_parser import BaseParser, ParsedDocument

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


class PDFParser(BaseParser):
    """
    Parser for PDF documents

    Uses PyMuPDF (fitz) as primary parser for speed and reliability.
    Falls back to pdfplumber for table extraction if needed.
    """

    # Common section headers in scientific papers
    SECTION_PATTERNS = {
        "abstract": r"^\s*abstract\s*$",
        "introduction": r"^\s*(?:introduction|background)\s*$",
        "methods": r"^\s*(?:methods?|materials?\s+and\s+methods?|methodology|experimental\s+(?:procedures?|methods?))\s*$",
        "results": r"^\s*results?\s*$",
        "discussion": r"^\s*discussion\s*$",
        "conclusion": r"^\s*(?:conclusion|conclusions?|concluding\s+remarks?)\s*$",
        "references": r"^\s*(?:references?|bibliography|works?\s+cited)\s*$",
        "acknowledgments": r"^\s*(?:acknowledgments?|acknowledgements?)\s*$",
    }

    def __init__(self, extract_tables: bool = False, enable_imrad: bool = True):
        """
        Initialize PDF parser

        Args:
            extract_tables: Whether to extract tables using pdfplumber (slower)
            enable_imrad: Whether to enable IMRAD section detection (recommended)
        """
        super().__init__()
        self.extract_tables = extract_tables
        self.enable_imrad = enable_imrad

        if not PYMUPDF_AVAILABLE:
            raise ImportError("PyMuPDF (fitz) is required. Install with: pip install PyMuPDF")

        # Initialize lemmatizer for IMRAD normalization
        if NLTK_AVAILABLE and enable_imrad:
            self.lemmatizer = WordNetLemmatizer()
        else:
            self.lemmatizer = None

    def parse(self, file_path: str) -> ParsedDocument:
        """Parse PDF from file path"""
        start_time = time.time()

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not file_path.suffix.lower() == '.pdf':
            raise ValueError(f"Not a PDF file: {file_path}")

        # Extract text using PyMuPDF
        doc = fitz.open(file_path)
        text = self._extract_text_pymupdf(doc)
        page_count = len(doc)

        # Extract metadata
        metadata = self._extract_metadata(doc)
        doc.close()

        # Clean text
        text = self._clean_text(text)

        # Detect sections
        sections = self._detect_sections(text)

        # IMRAD sections (enhanced for hybrid pipeline)
        imrad_sections = None
        if self.enable_imrad:
            imrad_sections = self._split_into_imrad_sections(text)

        # Word count
        word_count = self._count_words(text)

        parse_time = time.time() - start_time

        return ParsedDocument(
            text=text,
            sections=sections,
            metadata=metadata,
            word_count=word_count,
            page_count=page_count,
            parse_time=parse_time,
            imrad_sections=imrad_sections,
            sentences=None  # Will be populated by pattern extractors
        )

    def parse_from_bytes(self, file_bytes: bytes, filename: str = "") -> ParsedDocument:
        """Parse PDF from bytes"""
        start_time = time.time()

        # Open document from bytes
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = self._extract_text_pymupdf(doc)
        page_count = len(doc)

        # Extract metadata
        metadata = self._extract_metadata(doc)
        if filename:
            metadata["filename"] = filename
        doc.close()

        # Clean text
        text = self._clean_text(text)

        # Detect sections
        sections = self._detect_sections(text)

        # IMRAD sections (enhanced for hybrid pipeline)
        imrad_sections = None
        if self.enable_imrad:
            imrad_sections = self._split_into_imrad_sections(text)

        # Word count
        word_count = self._count_words(text)

        parse_time = time.time() - start_time

        return ParsedDocument(
            text=text,
            sections=sections,
            metadata=metadata,
            word_count=word_count,
            page_count=page_count,
            parse_time=parse_time,
            imrad_sections=imrad_sections,
            sentences=None  # Will be populated by pattern extractors
        )

    def supports_format(self, file_extension: str) -> bool:
        """Check if parser supports PDF format"""
        return file_extension.lower() in ['.pdf']

    def _extract_text_pymupdf(self, doc: fitz.Document) -> str:
        """
        Extract text from PDF using PyMuPDF

        Args:
            doc: fitz.Document object

        Returns:
            Extracted text
        """
        text_parts = []

        for page_num in range(len(doc)):
            page = doc[page_num]

            # Extract text with layout preservation
            text = page.get_text("text")

            if text.strip():
                text_parts.append(text)

        return "\n\n".join(text_parts)

    def _extract_metadata(self, doc: fitz.Document) -> Dict[str, Any]:
        """
        Extract metadata from PDF

        Args:
            doc: fitz.Document object

        Returns:
            Dictionary with metadata
        """
        metadata = {}

        # Get PDF metadata
        pdf_metadata = doc.metadata

        if pdf_metadata:
            metadata["title"] = pdf_metadata.get("title", "")
            metadata["author"] = pdf_metadata.get("author", "")
            metadata["subject"] = pdf_metadata.get("subject", "")
            metadata["keywords"] = pdf_metadata.get("keywords", "")
            metadata["creator"] = pdf_metadata.get("creator", "")
            metadata["producer"] = pdf_metadata.get("producer", "")
            metadata["creation_date"] = pdf_metadata.get("creationDate", "")
            metadata["modification_date"] = pdf_metadata.get("modDate", "")

        # Try to extract title from first page if not in metadata
        if not metadata.get("title"):
            first_page = doc[0]
            first_page_text = first_page.get_text("text")
            lines = [line.strip() for line in first_page_text.split('\n') if line.strip()]
            if lines:
                # First non-empty line is often the title
                metadata["title"] = lines[0]

        return metadata

    def _detect_sections(self, text: str) -> Dict[str, str]:
        """
        Detect and extract sections from paper text

        Args:
            text: Full paper text

        Returns:
            Dictionary mapping section names to their content
        """
        sections = {}
        lines = text.split('\n')

        # Find section boundaries
        section_starts = []

        for i, line in enumerate(lines):
            line_stripped = line.strip()

            # Check against section patterns
            for section_name, pattern in self.SECTION_PATTERNS.items():
                if re.match(pattern, line_stripped, re.IGNORECASE):
                    section_starts.append((i, section_name))
                    break

        # Extract section content
        for idx, (line_num, section_name) in enumerate(section_starts):
            # Find end of section (next section or end of document)
            if idx + 1 < len(section_starts):
                end_line = section_starts[idx + 1][0]
            else:
                end_line = len(lines)

            # Extract content (skip the header line itself)
            content_lines = lines[line_num + 1:end_line]
            content = '\n'.join(content_lines).strip()

            if content:
                sections[section_name] = content

        # If no sections detected, try to extract abstract (often at the beginning)
        if not sections and len(lines) > 5:
            # Look for "abstract" keyword in first 20 lines
            for i in range(min(20, len(lines))):
                if re.search(r'\babstract\b', lines[i], re.IGNORECASE):
                    # Extract next few lines as abstract
                    abstract_lines = []
                    for j in range(i + 1, min(i + 50, len(lines))):
                        line = lines[j].strip()
                        if not line:
                            continue
                        # Stop if we hit another section header
                        if re.match(r'^\d+\.\s+\w+|^[A-Z\s]{10,}$', line):
                            break
                        abstract_lines.append(line)

                    if abstract_lines:
                        sections["abstract"] = ' '.join(abstract_lines)
                    break

        return sections

    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text from PDF artifacts

        Args:
            text: Raw extracted text

        Returns:
            Cleaned text
        """
        # Basic cleaning from parent class
        text = super()._clean_text(text)

        # Remove hyphenation at line breaks
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)

        # Fix common PDF extraction issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between camelCase

        # Remove header/footer patterns (common in papers)
        # Pattern: single line with page numbers, journal names, etc.
        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            line_stripped = line.strip()

            # Skip likely header/footer lines
            if len(line_stripped) < 100:  # Headers/footers are usually short
                # Check if line looks like header/footer
                if re.match(r'^[\d\s\-]+$', line_stripped):  # Just numbers and dashes
                    continue
                if re.match(r'^\w+\s+\d{4}$', line_stripped):  # Month Year
                    continue

            cleaned_lines.append(line)

        text = '\n'.join(cleaned_lines)

        return text

    def extract_tables(self, file_path: str) -> List[List[List[str]]]:
        """
        Extract tables from PDF using pdfplumber

        Args:
            file_path: Path to PDF file

        Returns:
            List of tables, where each table is a list of rows,
            and each row is a list of cell values

        Raises:
            ImportError: If pdfplumber is not installed
        """
        if not PDFPLUMBER_AVAILABLE:
            raise ImportError("pdfplumber is required for table extraction. "
                            "Install with: pip install pdfplumber")

        tables = []

        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_tables = page.extract_tables()
                if page_tables:
                    tables.extend(page_tables)

        return tables

    def _clean_text_remove_figures_tables(self, text: str) -> str:
        """
        Remove figures and tables from text (from division_into_semantic_blocks.py)

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        # Remove non-breaking spaces and zero-width characters
        text = text.replace('\u00a0', ' ')
        text = re.sub(r'[\u200b\ufeff]', '', text)
        text = re.sub(r'\r\n?', '\n', text)

        # Remove figure captions
        text = re.sub(r'(?im)^\s*(figure|fig\.?)\s*\d+[^.\n]*[\.\n]?', '', text)

        # Remove table captions
        text = re.sub(r'(?im)^\s*(table|tab\.?)\s*\d+[^.\n]*[\.\n]?', '', text)

        # Filter tabular lines
        def is_tabular_line(line):
            stripped = line.strip()
            if not stripped:
                return False
            num_ratio = sum(c.isdigit() for c in stripped) / len(stripped)
            return (
                num_ratio > 0.4
                or '\t' in stripped
                or stripped.count('  ') > 3
                or re.search(r'\|', stripped)
            )

        lines = []
        for line in text.splitlines():
            if not is_tabular_line(line):
                lines.append(line)
            elif lines and lines[-1] != '':
                lines.append('')

        text = '\n'.join(lines)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def _normalize_section_name(self, name: str) -> str:
        """
        Normalize section name for IMRAD mapping (from division_into_semantic_blocks.py)

        Args:
            name: Raw section name

        Returns:
            Normalized section name
        """
        name = name.strip().lower()
        name = re.sub(r'[:.\-–—]+$', '', name)
        name = re.sub(r'\s+', ' ', name)

        # Lemmatize if available
        if self.lemmatizer:
            words = [self.lemmatizer.lemmatize(w) for w in name.split()]
            name = " ".join(words)

        # Normalization map
        normalization_map = {
            "introduction": "introduction",
            "background": "introduction",
            "method": "methods",
            "methods": "methods",
            "material and method": "methods",
            "materials and method": "methods",
            "materials and methods": "methods",
            "result": "results",
            "results": "results",
            "result and discussion": "results",
            "results and discussion": "results",
            "discussion": "discussion",
            "general discussion": "discussion",
            "conclusion": "conclusion",
            "conclusions": "conclusion",
            "summary": "conclusion",
            "abstract": "abstract",
            "acknowledgment": "acknowledgments",
            "acknowledgments": "acknowledgments",
            "reference": "references",
            "references": "references",
        }

        return normalization_map.get(name, name)

    def _split_into_imrad_sections(self, text: str) -> Dict[str, str]:
        """
        Split text into IMRAD sections (from division_into_semantic_blocks.py)

        Args:
            text: Full paper text

        Returns:
            Dictionary mapping section names to content
        """
        # Clean text first
        text = self._clean_text_remove_figures_tables(text)

        # Remove excessive newlines
        text = re.sub(r'\n{2,}', '\n\n', text)

        # Section headers
        headers = [
            r'abstract',
            r'introduction|background',
            r'materials\s+and\s+methods|methods?',
            r'results?',
            r'discussion',
            r'conclusion[s]?',
            r'references',
            r'acknowledg(?:ement|ements|e)?'
        ]
        headers_union = '|'.join(headers)

        pattern = rf'(?:^|\n)\s*(?:\d+\.?|[IVXLCM]+\.)?\s*(?P<header>{headers_union})\b[:.]?\s*'
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))

        if not matches:
            # No headers found, return full text
            return {"full_text": text.strip()}

        sections = {}
        for i, m in enumerate(matches):
            header_raw = m.group('header')
            normalized_header = self._normalize_section_name(header_raw)

            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section_text = text[start:end].lstrip()
            section_text = re.sub(r'\n{3,}', '\n\n', section_text)

            # Merge sections with same normalized name
            if normalized_header in sections:
                sections[normalized_header] += "\n\n" + section_text
            else:
                sections[normalized_header] = section_text

        return sections
