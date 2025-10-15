"""GROBID-based parser for scientific PDFs with structured extraction"""

import time
import xml.etree.ElementTree as ET
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from .base_parser import BaseParser, ParsedDocument

try:
    from grobid_client.grobid_client import GrobidClient
    GROBID_CLIENT_AVAILABLE = True
except ImportError:
    GROBID_CLIENT_AVAILABLE = False


class GrobidParser(BaseParser):
    """
    Parser for PDF documents using GROBID service

    GROBID (GeneRation Of BIbliographic Data) is a machine learning library
    for extracting and structuring raw documents (particularly PDFs) into
    structured TEI XML with >90% accuracy for scientific papers.

    Key features:
    - High-quality IMRAD section extraction
    - Structured metadata (title, authors, affiliations, abstract)
    - Bibliographic reference parsing
    - Figure and table detection with coordinates
    - TEI XML output with >55 label types

    Performance:
    - Speed: 2-5 seconds per page
    - Accuracy: >90% for scientific papers
    - Cost: FREE (runs locally in Docker)

    Prerequisites:
    - GROBID service running (Docker: https://grobid.readthedocs.io/en/latest/Grobid-docker/)
    - Default endpoint: http://localhost:8070
    """

    # TEI XML namespace
    TEI_NS = "{http://www.tei-c.org/ns/1.0}"

    def __init__(
        self,
        grobid_server: Optional[str] = None,
        consolidate_header: Optional[bool] = None,
        consolidate_citations: Optional[bool] = None,
        include_raw_citations: Optional[bool] = None,
        include_raw_affiliations: Optional[bool] = None,
        tei_coordinates: Optional[bool] = None,
        segment_sentences: Optional[bool] = None,
        timeout: Optional[int] = None
    ):
        """
        Initialize GROBID parser

        Args:
            grobid_server: GROBID service URL (default: from config)
            consolidate_header: Consolidate header metadata with external sources
            consolidate_citations: Consolidate citations with CrossRef/PubMed
            include_raw_citations: Include raw citation strings
            include_raw_affiliations: Include raw affiliation strings
            tei_coordinates: Include bounding box coordinates in TEI
            segment_sentences: Segment text into sentences
            timeout: Request timeout in seconds (default: from config)
        """
        super().__init__()

        if not GROBID_CLIENT_AVAILABLE:
            raise ImportError(
                "grobid-client-python is required. Install with: "
                "pip install grobid-client-python"
            )

        # Load config
        config = self._load_config()

        # Use parameters if provided, otherwise use config values
        self.grobid_server = grobid_server or config.get("grobid_server", "http://localhost:8070")
        self.timeout = timeout or config.get("timeout", 120)

        # Initialize GROBID client
        self.client = GrobidClient(grobid_server=self.grobid_server)

        # Processing options (use config defaults)
        consolidate_header = consolidate_header if consolidate_header is not None else config.get("consolidate_header", True)
        consolidate_citations = consolidate_citations if consolidate_citations is not None else config.get("consolidate_citations", False)
        include_raw_citations = include_raw_citations if include_raw_citations is not None else config.get("include_raw_citations", True)
        include_raw_affiliations = include_raw_affiliations if include_raw_affiliations is not None else config.get("include_raw_affiliations", True)
        tei_coordinates = tei_coordinates if tei_coordinates is not None else config.get("tei_coordinates", True)
        segment_sentences = segment_sentences if segment_sentences is not None else config.get("segment_sentences", True)

        self.processing_options = {
            "consolidateHeader": "1" if consolidate_header else "0",
            "consolidateCitations": "1" if consolidate_citations else "0",
            "includeRawCitations": "1" if include_raw_citations else "0",
            "includeRawAffiliations": "1" if include_raw_affiliations else "0",
            "teiCoordinates": tei_coordinates,
            "segmentSentences": "1" if segment_sentences else "0"
        }

    def _load_config(self) -> Dict[str, Any]:
        """
        Load GROBID configuration from grobid_config.yaml

        Returns:
            Configuration dictionary with defaults
        """
        config_path = Path(__file__).parent.parent / "config" / "grobid_config.yaml"

        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    return config or {}
            except Exception as e:
                print(f"Warning: Failed to load GROBID config from {config_path}: {e}")

        # Return defaults if config file not found
        return {
            "grobid_server": "http://localhost:8070",
            "timeout": 120,
            "consolidate_header": True,
            "consolidate_citations": False,
            "include_raw_citations": True,
            "include_raw_affiliations": True,
            "tei_coordinates": True,
            "segment_sentences": True
        }

    def parse(self, file_path: str) -> ParsedDocument:
        """
        Parse PDF using GROBID service

        Args:
            file_path: Path to PDF file

        Returns:
            ParsedDocument with extracted structure

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported
            ConnectionError: If GROBID service is unavailable
        """
        start_time = time.time()

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not file_path.suffix.lower() == '.pdf':
            raise ValueError(f"Not a PDF file: {file_path}")

        # Process PDF with GROBID
        try:
            tei_xml = self._process_with_grobid(file_path)
        except Exception as e:
            raise ConnectionError(
                f"Failed to process PDF with GROBID service at {self.grobid_server}. "
                f"Ensure GROBID Docker container is running. Error: {e}"
            )

        # Parse TEI XML
        root = ET.fromstring(tei_xml)

        # Extract metadata
        metadata = self._extract_metadata(root)

        # Extract title from metadata
        title = metadata.get("title", "")

        # Extract IMRAD sections
        imrad_sections = self._extract_imrad_sections(root)

        # Extract full text
        full_text = self._extract_full_text(root)

        # Count words and pages
        word_count = self._count_words(full_text)
        page_count = self._count_pages(root)

        parse_time = time.time() - start_time

        return ParsedDocument(
            text=full_text,
            sections={},  # Legacy, use imrad_sections instead
            metadata=metadata,
            word_count=word_count,
            page_count=page_count,
            parse_time=parse_time,
            imrad_sections=imrad_sections,
            _spacy_doc=None,  # GROBID provides pre-segmented text
            title=title
        )

    def supports_format(self, file_extension: str) -> bool:
        """Check if parser supports PDF format"""
        return file_extension.lower() in ['.pdf']

    def _process_with_grobid(self, file_path: Path) -> str:
        """
        Send PDF to GROBID service and get TEI XML response

        Args:
            file_path: Path to PDF file

        Returns:
            TEI XML string
        """
        import requests

        # Prepare file for upload
        with open(file_path, 'rb') as pdf_file:
            files = {
                'input': (file_path.name, pdf_file, 'application/pdf')
            }

            # Prepare form data
            data = self.processing_options.copy()

            # Call GROBID processFulltextDocument endpoint
            url = f"{self.grobid_server}/api/processFulltextDocument"

            response = requests.post(
                url,
                files=files,
                data=data,
                timeout=self.timeout
            )

            response.raise_for_status()

            return response.text

    def _extract_metadata(self, root: ET.Element) -> Dict[str, Any]:
        """
        Extract document metadata from TEI XML

        Args:
            root: TEI XML root element

        Returns:
            Dictionary with metadata
        """
        metadata = {}

        # Find teiHeader
        tei_header = root.find(f".//{self.TEI_NS}teiHeader")
        if tei_header is None:
            return metadata

        # Extract title
        title_elem = tei_header.find(f".//{self.TEI_NS}titleStmt/{self.TEI_NS}title")
        if title_elem is not None:
            metadata["title"] = self._get_text(title_elem)

        # Extract authors
        authors = []
        author_elems = tei_header.findall(f".//{self.TEI_NS}sourceDesc//{self.TEI_NS}author")
        for author_elem in author_elems:
            author_name = self._extract_author_name(author_elem)
            if author_name:
                authors.append(author_name)

        if authors:
            metadata["authors"] = authors
            metadata["author"] = ", ".join(authors)

        # Extract affiliations
        affiliations = []
        affiliation_elems = tei_header.findall(f".//{self.TEI_NS}affiliation")
        for aff_elem in affiliations:
            org_name = aff_elem.find(f".//{self.TEI_NS}orgName")
            if org_name is not None:
                affiliations.append(self._get_text(org_name))

        if affiliations:
            metadata["affiliations"] = affiliations

        # Extract abstract
        abstract_elem = tei_header.find(f".//{self.TEI_NS}abstract")
        if abstract_elem is not None:
            metadata["abstract"] = self._get_text(abstract_elem)

        # Extract keywords
        keywords_elem = tei_header.find(f".//{self.TEI_NS}keywords")
        if keywords_elem is not None:
            keywords = [self._get_text(term) for term in keywords_elem.findall(f".//{self.TEI_NS}term")]
            if keywords:
                metadata["keywords"] = ", ".join(keywords)

        # Extract publication info
        pub_stmt = tei_header.find(f".//{self.TEI_NS}publicationStmt")
        if pub_stmt is not None:
            date_elem = pub_stmt.find(f".//{self.TEI_NS}date")
            if date_elem is not None:
                metadata["publication_date"] = date_elem.get("when", "")

        # Extract DOI
        idno_elems = tei_header.findall(f".//{self.TEI_NS}idno")
        for idno in idno_elems:
            idno_type = idno.get("type", "").lower()
            if idno_type == "doi":
                metadata["doi"] = self._get_text(idno)
            elif idno_type == "pmid":
                metadata["pmid"] = self._get_text(idno)
            elif idno_type == "pmc":
                metadata["pmc"] = self._get_text(idno)

        return metadata

    def _map_section_name(self, title: str) -> Optional[str]:
        """
        Map section title to IMRAD section name

        Args:
            title: Section title from <head> element

        Returns:
            IMRAD section name or None if not mapped
        """
        mapping = {
            "introduction": "introduction",
            "background": "introduction",
            "methods": "methods",
            "materials and methods": "methods",
            "methodology": "methods",
            "study population": "methods",
            "immunohistochemistry": "methods",
            "image analysis": "methods",
            "statistical analysis": "methods",
            "experimental design": "methods",
            "data collection": "methods",
            "results": "results",
            "findings": "results",
            "observations": "results",
            "discussion": "discussion",
            "interpretation": "discussion",
            "conclusion": "conclusion",
            "conclusions": "conclusion",
            "concluding remarks": "conclusion",
            "acknowledgments": "acknowledgments",
            "acknowledgements": "acknowledgments",
            "acknowledgment": "acknowledgments",
            "acknowledgement": "acknowledgments",
            "funding": "acknowledgments",
            "references": "references",
            "bibliography": "references",
        }
        return mapping.get(title.strip().lower())

    def _get_div_content_without_head(self, div_elem: ET.Element) -> str:
        """
        Extract text from div excluding the head element

        Args:
            div_elem: Div XML element

        Returns:
            Text content without head
        """
        text_parts = []

        for elem in div_elem:
            # Skip <head> element
            if elem.tag == f"{self.TEI_NS}head":
                continue
            # Collect text from all other elements
            text_parts.append(self._get_text(elem))

        return "\n".join(text_parts).strip()

    def _extract_imrad_sections(self, root: ET.Element) -> Dict[str, str]:
        """
        Extract IMRAD sections from TEI XML

        GROBID structures documents using <div> elements with <head> tags.
        Format: <div><head>Introduction</head><p>...</p></div>

        Args:
            root: TEI XML root element

        Returns:
            Dictionary mapping section names to content
        """
        sections = {}

        # Find body element
        body = root.find(f".//{self.TEI_NS}text/{self.TEI_NS}body")
        if body is None:
            return sections

        # Extract sections from <div> elements with <head>
        for div in body.findall(f"{self.TEI_NS}div"):
            # Look for <head> element inside <div>
            head_elem = div.find(f"{self.TEI_NS}head")
            if head_elem is None:
                continue

            # Extract section title
            section_title = self._get_text(head_elem)
            section_name = self._map_section_name(section_title)

            if not section_name:
                continue

            # Extract text content (excluding head)
            section_text = self._get_div_content_without_head(div)

            # Merge if section already exists (e.g., multiple Methods subsections)
            if section_name in sections:
                sections[section_name] += "\n\n" + section_text
            else:
                sections[section_name] = section_text

        # Extract abstract from teiHeader if not in body
        if "abstract" not in sections:
            abstract_elem = root.find(f".//{self.TEI_NS}teiHeader//{self.TEI_NS}abstract")
            if abstract_elem is not None:
                sections["abstract"] = self._get_text(abstract_elem)

        return sections

    def _extract_full_text(self, root: ET.Element) -> str:
        """
        Extract full text from TEI XML

        Args:
            root: TEI XML root element

        Returns:
            Full document text
        """
        body = root.find(f".//{self.TEI_NS}text/{self.TEI_NS}body")
        if body is not None:
            return self._get_text(body)
        return ""

    def _extract_author_name(self, author_elem: ET.Element) -> Optional[str]:
        """
        Extract author name from author element

        Args:
            author_elem: Author XML element

        Returns:
            Formatted author name
        """
        # Find persName (person name)
        pers_name = author_elem.find(f".//{self.TEI_NS}persName")
        if pers_name is None:
            return None

        # Extract name parts
        forename = pers_name.find(f".//{self.TEI_NS}forename")
        surname = pers_name.find(f".//{self.TEI_NS}surname")

        parts = []
        if forename is not None:
            parts.append(self._get_text(forename))
        if surname is not None:
            parts.append(self._get_text(surname))

        return " ".join(parts) if parts else None

    def _get_text(self, element: ET.Element) -> str:
        """
        Extract text content from XML element

        Args:
            element: XML element

        Returns:
            Text content
        """
        return "".join(element.itertext()).strip()

    def _count_pages(self, root: ET.Element) -> Optional[int]:
        """
        Count pages from TEI XML

        Args:
            root: TEI XML root element

        Returns:
            Number of pages or None
        """
        # GROBID includes page coordinates in TEI
        # Count unique page numbers from coordinates
        pages = set()
        for elem in root.findall(f".//{self.TEI_NS}biblStruct"):
            coords = elem.get("coords", "")
            if coords:
                # Format: "page,x1,y1,x2,y2"
                parts = coords.split(",")
                if parts:
                    try:
                        pages.add(int(parts[0]))
                    except (ValueError, IndexError):
                        pass

        return len(pages) if pages else None

    def get_references(self, file_path: str) -> List[Dict[str, str]]:
        """
        Extract bibliographic references from PDF

        Args:
            file_path: Path to PDF file

        Returns:
            List of parsed references
        """
        tei_xml = self._process_with_grobid(Path(file_path))
        root = ET.fromstring(tei_xml)

        references = []

        # Find all biblStruct elements
        bibl_structs = root.findall(f".//{self.TEI_NS}listBibl/{self.TEI_NS}biblStruct")

        for bibl in bibl_structs:
            ref = {}

            # Extract title
            title_elem = bibl.find(f".//{self.TEI_NS}title")
            if title_elem is not None:
                ref["title"] = self._get_text(title_elem)

            # Extract authors
            authors = []
            author_elems = bibl.findall(f".//{self.TEI_NS}author")
            for author in author_elems:
                name = self._extract_author_name(author)
                if name:
                    authors.append(name)

            if authors:
                ref["authors"] = authors

            # Extract year
            date_elem = bibl.find(f".//{self.TEI_NS}date")
            if date_elem is not None:
                ref["year"] = date_elem.get("when", "")

            # Extract DOI
            idno_elems = bibl.findall(f".//{self.TEI_NS}idno")
            for idno in idno_elems:
                if idno.get("type", "").lower() == "doi":
                    ref["doi"] = self._get_text(idno)

            references.append(ref)

        return references
