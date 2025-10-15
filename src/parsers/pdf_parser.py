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

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


class PDFParser(BaseParser):
    """
    Parser for PDF documents

    Uses PyMuPDF (fitz) as primary parser for speed and reliability.
    Falls back to pdfplumber for table extraction if needed.
    """

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

        # Initialize lemmatizer for section name normalization
        self.lemmatizer = None
        if NLTK_AVAILABLE:
            try:
                self.lemmatizer = WordNetLemmatizer()
            except:
                pass

        # Initialize spaCy for sentence splitting
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                print("Warning: spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")

        if not PYMUPDF_AVAILABLE:
            raise ImportError("PyMuPDF (fitz) is required. Install with: pip install PyMuPDF")

    # ========== IMRAD SECTION KEYWORDS ==========

    ABSTRACT_KEYWORDS = [
        "abstract",
        "summary",
        "synopsis",
        "précis",
        "overview",
        "graphical abstract",
        "structured abstract",
        # Multilingual
        "résumé",  # French
        "zusammenfassung",  # German
        "resumen",  # Spanish
    ]

    INTRODUCTION_KEYWORDS = [
        "introduction",
        "background",
        "preamble",
        "prologue",
        "rationale",
        "motivation",
        "study background",
        "introduction and background",
        "background and significance",
        # Specific domains
        "clinical background",
        "theoretical background",
    ]

    METHODS_KEYWORDS = [
        # Core terms
        "methods",
        "methodology",
        "materials and methods",
        "methods and materials",
        "experimental methods",
        "experimental procedures",
        # Variations
        "materials",
        "procedures",
        "protocol",
        "protocols",
        "approach",
        "techniques",
        # Specific types
        "study design",
        "experimental design",
        "research design",
        "study protocol",
        "experimental protocol",
        # Clinical
        "patients and methods",
        "subjects and methods",
        "participants",
        "study population",
        "patient population",
        # Data collection
        "data collection",
        "data acquisition",
        "sampling methods",
        "measurement methods",
        # Analysis methods
        "statistical methods",
        "statistical analysis",
        "analytical methods",
        "data analysis",
        "computational methods",
        # Ethics
        "ethics statement",
        "ethical approval",
        # Subsections
        "cell culture",
        "animal models",
        "reagents",
        "instruments",
        "equipment",
    ]

    RESULTS_KEYWORDS = [
        # Core
        "results",
        "findings",
        "observations",
        "outcomes",
        # Combined sections
        "results and discussion",
        "results and analysis",
        # Specific results
        "experimental results",
        "clinical results",
        "main results",
        "primary results",
        "secondary results",
        # Data presentation
        "data",
        "measurements",
        "output",
    ]

    DISCUSSION_KEYWORDS = [
        # Core
        "discussion",
        "interpretation",
        "analysis and discussion",
        # Combined
        "results and discussion",
        "discussion and conclusions",
        "discussion and conclusion",
        # Specific aspects
        "implications",
        "significance",
        "limitations",
        "study limitations",
        "limitations of the study",
        # Future work
        "future work",
        "future research",
        "future directions",
        "further research",
    ]

    CONCLUSION_KEYWORDS = [
        # Core
        "conclusion",
        "conclusions",
        "concluding remarks",
        "final remarks",
        "summary and conclusions",
        # Variations
        "conclusions and perspectives",
        "conclusions and future work",
        "conclusions and recommendations",
        # Specific
        "key findings",
        "take-home message",
        "clinical implications",
        "practical implications",
    ]

    ACKNOWLEDGMENTS_KEYWORDS = [
        "acknowledgments",
        "acknowledgements",
        "acknowledgment",
        "acknowledgement",
        "credits",
        "funding",
        "financial support",
        "grant support",
        "competing interests",
        "conflict of interest",
        "conflicts of interest",
        "author contributions",
        "contributions",
    ]

    REFERENCES_KEYWORDS = [
        "references",
        "bibliography",
        "citations",
        "works cited",
        "literature cited",
        "reference list",
        "bibliographic references",
    ]

    SUPPLEMENTARY_KEYWORDS = [
        "supplementary materials",
        "supplementary material",
        "supplementary information",
        "supplementary data",
        "supporting information",
        "appendix",
        "appendices",
        "supplemental data",
        "additional files",
        "supplementary figures",
        "supplementary tables",
    ]

    # Section order (for validation and priority)
    # Title has no keywords - extracted from document start to abstract
    SECTION_ORDER = [
        ("title", []),  # Special: from document start to abstract
        ("abstract", ABSTRACT_KEYWORDS),
        ("introduction", INTRODUCTION_KEYWORDS),
        ("methods", METHODS_KEYWORDS),
        ("results", RESULTS_KEYWORDS),
        ("discussion", DISCUSSION_KEYWORDS),
        ("conclusion", CONCLUSION_KEYWORDS),
        ("acknowledgments", ACKNOWLEDGMENTS_KEYWORDS),
        ("references", REFERENCES_KEYWORDS),
        ("supplementary", SUPPLEMENTARY_KEYWORDS),
    ]

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
        # text = self._clean_text(text)

        # Parse text with spaCy (returns Doc object)
        spacy_doc = self._parse_text_with_spacy(text)

        # Split into IMRAD sections if enabled (using spaCy Doc)
        imrad_sections = self._split_into_sections(spacy_doc)

        # Word count
        word_count = self._count_words(text)

        # Extract title from metadata or IMRAD sections
        title = metadata.get("title", "")
        if not title and imrad_sections and "title" in imrad_sections:
            title = imrad_sections["title"]

        parse_time = time.time() - start_time

        return ParsedDocument(
            text=text,
            sections={},  # Legacy, use imrad_sections instead
            metadata=metadata,
            word_count=word_count,
            page_count=page_count,
            parse_time=parse_time,
            imrad_sections=imrad_sections,
            _spacy_doc=spacy_doc,  # Private: full spaCy Doc object
            title=title
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

    def _parse_text_with_spacy(self, text: str):
        """
        Parse text with spaCy and return Doc object

        Args:
            text: Input text

        Returns:
            spacy.Doc object with full linguistic annotations

        Raises:
            RuntimeError: If spaCy is not available
        """
        if not self.nlp:
            raise RuntimeError(
                "spaCy is required for text parsing. "
                "Install with: pip install spacy && python -m spacy download en_core_web_sm"
            )

        # Parse text with spaCy (returns Doc object)
        return self.nlp(text)

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

        # Comprehensive normalization map for all keyword variations
        normalization_map = {
            # ABSTRACT section
            "abstract": "abstract",
            "summary": "abstract",
            "synopsis": "abstract",
            "précis": "abstract",
            "overview": "abstract",
            "graphical abstract": "abstract",
            "structured abstract": "abstract",
            "résumé": "abstract",
            "zusammenfassung": "abstract",
            "resumen": "abstract",

            # INTRODUCTION section
            "introduction": "introduction",
            "background": "introduction",
            "preamble": "introduction",
            "prologue": "introduction",
            "rationale": "introduction",
            "motivation": "introduction",
            "study background": "introduction",
            "introduction and background": "introduction",
            "background and significance": "introduction",
            "clinical background": "introduction",
            "theoretical background": "introduction",

            # METHODS section
            "method": "methods",
            "methods": "methods",
            "methodology": "methods",
            "material and method": "methods",
            "materials and method": "methods",
            "materials and methods": "methods",
            "methods and materials": "methods",
            "experimental method": "methods",
            "experimental methods": "methods",
            "experimental procedure": "methods",
            "experimental procedures": "methods",
            "material": "methods",
            "materials": "methods",
            "procedure": "methods",
            "procedures": "methods",
            "protocol": "methods",
            "protocols": "methods",
            "approach": "methods",
            "technique": "methods",
            "techniques": "methods",
            "study design": "methods",
            "experimental design": "methods",
            "research design": "methods",
            "study protocol": "methods",
            "experimental protocol": "methods",
            "patient and method": "methods",
            "patients and methods": "methods",
            "subject and method": "methods",
            "subjects and methods": "methods",
            "participant": "methods",
            "participants": "methods",
            "study population": "methods",
            "patient population": "methods",
            "data collection": "methods",
            "data acquisition": "methods",
            "sampling method": "methods",
            "sampling methods": "methods",
            "measurement method": "methods",
            "measurement methods": "methods",
            "statistical method": "methods",
            "statistical methods": "methods",
            "statistical analysis": "methods",
            "analytical method": "methods",
            "analytical methods": "methods",
            "data analysis": "methods",
            "computational method": "methods",
            "computational methods": "methods",
            "ethics statement": "methods",
            "ethical approval": "methods",

            # RESULTS section
            "result": "results",
            "results": "results",
            "finding": "results",
            "findings": "results",
            "observation": "results",
            "observations": "results",
            "outcome": "results",
            "outcomes": "results",
            "result and analysis": "results",
            "results and analysis": "results",
            "experimental result": "results",
            "experimental results": "results",
            "clinical result": "results",
            "clinical results": "results",
            "main result": "results",
            "main results": "results",
            "primary result": "results",
            "primary results": "results",
            "secondary result": "results",
            "secondary results": "results",
            "data": "results",
            "measurement": "results",
            "measurements": "results",
            "output": "results",

            # Combined RESULTS + DISCUSSION
            "result and discussion": "results",
            "results and discussion": "results",

            # DISCUSSION section
            "discussion": "discussion",
            "interpretation": "discussion",
            "analysis and discussion": "discussion",
            "discussion and conclusion": "discussion",
            "discussion and conclusions": "discussion",
            "implication": "discussion",
            "implications": "discussion",
            "significance": "discussion",
            "limitation": "discussion",
            "limitations": "discussion",
            "study limitation": "discussion",
            "study limitations": "discussion",
            "limitations of the study": "discussion",
            "future work": "discussion",
            "future research": "discussion",
            "future direction": "discussion",
            "future directions": "discussion",
            "further research": "discussion",
            "general discussion": "discussion",

            # CONCLUSION section
            "conclusion": "conclusion",
            "conclusions": "conclusion",
            "concluding remark": "conclusion",
            "concluding remarks": "conclusion",
            "final remark": "conclusion",
            "final remarks": "conclusion",
            "summary and conclusion": "conclusion",
            "summary and conclusions": "conclusion",
            "conclusion and perspective": "conclusion",
            "conclusions and perspectives": "conclusion",
            "conclusion and future work": "conclusion",
            "conclusions and future work": "conclusion",
            "conclusion and recommendation": "conclusion",
            "conclusions and recommendations": "conclusion",
            "key finding": "conclusion",
            "key findings": "conclusion",
            "take-home message": "conclusion",
            "clinical implication": "conclusion",
            "clinical implications": "conclusion",
            "practical implication": "conclusion",
            "practical implications": "conclusion",

            # ACKNOWLEDGMENTS section
            "acknowledgment": "acknowledgments",
            "acknowledgments": "acknowledgments",
            "acknowledgement": "acknowledgments",
            "acknowledgements": "acknowledgments",
            "credit": "acknowledgments",
            "credits": "acknowledgments",
            "funding": "acknowledgments",
            "financial support": "acknowledgments",
            "grant support": "acknowledgments",
            "competing interest": "acknowledgments",
            "competing interests": "acknowledgments",
            "conflict of interest": "acknowledgments",
            "conflicts of interest": "acknowledgments",
            "author contribution": "acknowledgments",
            "author contributions": "acknowledgments",
            "contribution": "acknowledgments",
            "contributions": "acknowledgments",

            # REFERENCES section
            "reference": "references",
            "references": "references",
            "bibliography": "references",
            "citation": "references",
            "citations": "references",
            "works cited": "references",
            "literature cited": "references",
            "reference list": "references",
            "bibliographic reference": "references",
            "bibliographic references": "references",

            # SUPPLEMENTARY section
            "supplementary material": "supplementary",
            "supplementary materials": "supplementary",
            "supplementary information": "supplementary",
            "supplementary data": "supplementary",
            "supporting information": "supplementary",
            "appendix": "supplementary",
            "appendices": "supplementary",
            "supplemental data": "supplementary",
            "additional file": "supplementary",
            "additional files": "supplementary",
            "supplementary figure": "supplementary",
            "supplementary figures": "supplementary",
            "supplementary table": "supplementary",
            "supplementary tables": "supplementary",
        }

        return normalization_map.get(name, name)

    def _create_section_pattern(self, keywords: List[str]) -> re.Pattern:
        """
        Create regex pattern for detecting section headers from keyword list

        Args:
            keywords: List of keywords for this section type

        Returns:
            Compiled regex pattern
        """
        # Escape special characters and sort by length (longest first for better matching)
        escaped_keywords = [re.escape(k) for k in sorted(keywords, key=len, reverse=True)]

        # Pattern explanation:
        # (?:^|\n)\s*           - Start of line
        # (?:\d+\.?|[IVXLCM]+\.?|[A-Z]\.?)?\s*  - Optional numbering (numeric, Roman, letter)
        # (?:keywords)\b        - One of the keywords (word boundary)
        # [:.\-–—\s]*          - Optional trailing punctuation/whitespace
        pattern_str = (
            r'(?:^|\n)\s*'
            r'(?:\d+\.?|[IVXLCM]+\.?|[A-Z]\.?)?\s*'
            r'(?:' + '|'.join(escaped_keywords) + r')\b'
            r'[:.\-–—\s]*'
        )

        return re.compile(pattern_str, re.IGNORECASE | re.MULTILINE)

    def _normalize_header(self, text: str) -> str:
        """
        Normalize section header text by removing numbering and punctuation

        Args:
            text: Raw header text

        Returns:
            Normalized header text
        """
        text = text.strip()

        # Remove numeric numbering (1., 2., etc.)
        text = re.sub(r'^\d+\.?\s*', '', text)

        # Remove Roman numerals (I., II., III., etc.)
        text = re.sub(r'^[IVXLCM]+\.?\s*', '', text, flags=re.IGNORECASE)

        # Remove letter numbering (A., B., etc.) - only when followed by period + space
        text = re.sub(r'^[A-Z]\.\s+', '', text)

        # Remove trailing punctuation
        text = re.sub(r'[:.\-–—]+$', '', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        return text.strip().lower()

    def _find_first_keyword_match(self, text: str, keywords: List[str], search_start_pos: int = 0) -> Optional[Dict[str, Any]]:
        """
        Find FIRST occurrence of ANY keyword in priority order (case-insensitive substring search)

        Keywords are searched in order of priority (as provided in the list).
        Returns the first match found, checking keywords one by one.

        Args:
            text: Text to search in
            keywords: List of keywords in priority order (e.g., ['introduction', 'background', ...])
            search_start_pos: Position to start searching from

        Returns:
            Dict with 'keyword', 'start', 'end', 'matched_text' or None if not found
        """
        text_lower = text.lower()

        # Search keywords in priority order
        for keyword in keywords:
            keyword_lower = keyword.lower()

            # Find first occurrence of this keyword
            pos = text_lower.find(keyword_lower, search_start_pos)

            if pos != -1:
                # Found! Return immediately (priority order)
                return {
                    'keyword': keyword,
                    'start': pos,
                    'end': pos + len(keyword),
                    'matched_text': text[pos:pos + len(keyword)]
                }

        # No keyword found
        return None

    def _validate_with_regex(self, text: str, pos: int, keyword: str) -> Dict[str, Any]:
        """
        Validate if keyword occurrence at position matches section header pattern

        Uses the same pattern as _create_section_pattern for consistency:
        - Optional line start
        - Optional numbering (1., I., A.)
        - Keyword (case-insensitive)
        - Optional punctuation/whitespace

        Args:
            text: Full text
            pos: Position of keyword occurrence
            keyword: The keyword found

        Returns:
            Dict with 'valid', 'match_start', 'match_end' keys
        """
        # Create regex pattern for this keyword (same as _create_section_pattern)
        escaped_keyword = re.escape(keyword)

        # Pattern: optional line start + optional numbering + keyword + optional punctuation
        pattern = re.compile(
            r'(?:^|\n)\s*'
            r'(?:\d+\.?|[IVXLCM]+\.?|[A-Z]\.?)?\s*'
            + escaped_keyword + r'\b'
            r'[:.\-–—\s]*',
            re.IGNORECASE | re.MULTILINE
        )

        # Search around the position (allow some padding for line start detection)
        search_start = max(0, pos - 20)
        search_end = min(len(text), pos + len(keyword) + 20)
        search_text = text[search_start:search_end]

        match = pattern.search(search_text)

        if match:
            # Calculate absolute positions
            match_abs_start = search_start + match.start()
            match_abs_end = search_start + match.end()

            # Valid if our keyword position is within the match
            is_valid = (match_abs_start <= pos < match_abs_end)

            if is_valid:
                return {
                    'valid': True,
                    'match_start': match_abs_start,
                    'match_end': match_abs_end
                }

        # No regex match - not a valid section header
        return {
            'valid': False,
            'match_start': pos,
            'match_end': pos + len(keyword)
        }

    def _split_into_sections(self, spacy_doc: Any) -> Dict[str, str]:
        """
        Split text into IMRAD sections using sequential keyword-based matching

        New Algorithm (as per requirements):
        1. Search sections in strict order: title → abstract → introduction → methods → ...
        2. For each section:
           - Search keywords in priority order (e.g., 'introduction' before 'background')
           - Find FIRST occurrence of ANY keyword
           - Validate with regex pattern
           - If valid → mark as section boundary
        3. Next section search starts AFTER current section boundary
        4. Extract content between boundaries

        Args:
            spacy_doc: spaCy Doc object with full linguistic annotations

        Returns:
            Dictionary mapping section names to content
        """
        # Extract text from spaCy Doc
        text = spacy_doc.text

        # Remove excessive newlines
        text = re.sub(r'\n{2,}', '\n\n', text)

        # List of section boundaries (position, name, end_of_header)
        section_boundaries = []

        # Current search position (sections must appear in strict order)
        search_start_pos = 0

        # Process each section in order
        for section_name, keywords in self.SECTION_ORDER:
            # Special handling for title section (no keywords)
            if section_name == "title":
                # Title is implicit: from start to first abstract keyword
                # Will be added after we find abstract
                continue

            if not keywords:
                continue

            # Find FIRST keyword match (in priority order)
            keyword_match = self._find_first_keyword_match(text, keywords, search_start_pos)

            if not keyword_match:
                # Section not found, continue to next
                continue

            # Validate with regex pattern
            validation = self._validate_with_regex(text, keyword_match['start'], keyword_match['keyword'])

            if not validation['valid']:
                # Keyword found but doesn't match section header pattern
                continue

            # Valid section boundary found!
            section_boundaries.append({
                'section_name': section_name,
                'start': validation['match_start'],
                'end': validation['match_end']
            })

            # Next search starts AFTER this section header
            search_start_pos = validation['match_end']

        # If no sections found at all, return full text
        if not section_boundaries:
            return {"full_text": text.strip()}

        # Sort boundaries by position (should already be sorted, but ensure correctness)
        section_boundaries.sort(key=lambda x: x['start'])

        # Add title section if we found abstract
        first_section = section_boundaries[0]
        if first_section['section_name'] == 'abstract' and first_section['start'] > 0:
            # Title exists from start to abstract
            section_boundaries.insert(0, {
                'section_name': 'title',
                'start': 0,
                'end': 0  # Title has no header, content starts immediately
            })

        # Extract section texts between boundaries
        return self._extract_section_texts(text, section_boundaries)

    def _extract_section_texts(self, text: str, section_boundaries: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Extract section texts between boundaries

        Args:
            text: Full document text
            section_boundaries: List of section boundaries with 'section_name', 'start', 'end' keys

        Returns:
            Dictionary mapping section names to extracted content
        """
        sections = {}

        for i, boundary in enumerate(section_boundaries):
            section_name = boundary['section_name']

            # Content starts after header (or at start for title)
            if section_name == 'title':
                content_start = 0
            else:
                content_start = boundary['end']

            # Content ends at next section header or end of text
            if i + 1 < len(section_boundaries):
                content_end = section_boundaries[i + 1]['start']
            else:
                content_end = len(text)

            # Extract section text
            section_text = text[content_start:content_end]

            # Clean up whitespace
            if section_name == 'title':
                # For title, preserve leading whitespace but remove trailing
                section_text = section_text.rstrip()
            else:
                # For other sections, strip both sides
                section_text = section_text.strip()

            # Remove excessive newlines
            section_text = re.sub(r'\n{3,}', '\n\n', section_text)

            # Merge if same section name appears multiple times (e.g., multiple Methods subsections)
            if section_name in sections:
                sections[section_name] += "\n\n" + section_text
            else:
                sections[section_name] = section_text

        return sections
