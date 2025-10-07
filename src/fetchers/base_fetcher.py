"""Base fetcher for all paper sources"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class PaperMetadata:
    """Standard paper metadata structure"""
    pmid: Optional[str] = None          # PubMed ID
    pmc_id: Optional[str] = None        # PubMed Central ID
    doi: Optional[str] = None           # Digital Object Identifier
    title: str = ""
    authors: List[str] = None
    abstract: str = ""
    journal: str = ""
    publication_date: str = ""
    keywords: List[str] = None
    full_text_url: Optional[str] = None
    pdf_url: Optional[str] = None
    citations_count: int = 0

    def __post_init__(self):
        if self.authors is None:
            self.authors = []
        if self.keywords is None:
            self.keywords = []


class BaseFetcher(ABC):
    """Abstract base class for all paper fetchers"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_url = config.get("base_url", "")
        self.timeout = config.get("timeout", 30)
        self.rate_limit = config.get("rate_limit", {})

    @abstractmethod
    def search(
        self,
        query: str,
        max_results: int = 10,
        **kwargs
    ) -> List[str]:
        """
        Search for papers

        Args:
            query: Search query
            max_results: Maximum number of results
            **kwargs: Additional search parameters

        Returns:
            List of paper IDs
        """
        pass

    @abstractmethod
    def fetch_paper(self, paper_id: str) -> PaperMetadata:
        """
        Fetch paper metadata

        Args:
            paper_id: Paper identifier (PMID, DOI, etc.)

        Returns:
            PaperMetadata: Paper information
        """
        pass

    @abstractmethod
    def fetch_full_text(self, paper_id: str) -> Optional[str]:
        """
        Fetch full text of paper (if available)

        Args:
            paper_id: Paper identifier

        Returns:
            str: Full text content or None if not available
        """
        pass

    def fetch_multiple(self, paper_ids: List[str]) -> List[PaperMetadata]:
        """
        Fetch multiple papers

        Args:
            paper_ids: List of paper identifiers

        Returns:
            List of PaperMetadata
        """
        papers = []
        for paper_id in paper_ids:
            try:
                paper = self.fetch_paper(paper_id)
                papers.append(paper)
            except Exception as e:
                print(f"Error fetching paper {paper_id}: {e}")
                continue
        return papers
