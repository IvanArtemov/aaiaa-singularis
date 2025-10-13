"""Factory for creating paper fetchers"""

from typing import Optional
from .base_fetcher import BaseFetcher
from .pubmed_fetcher import PubMedFetcher
from .arxiv_fetcher import ArXivFetcher
from ..config.settings import settings


class FetcherFactory:
    """Factory for creating paper fetchers"""

    @staticmethod
    def create(fetcher_type: Optional[str] = None) -> BaseFetcher:
        """
        Create fetcher for specified type

        Args:
            fetcher_type: "pubmed", "pmc", etc. (if None, uses config default)

        Returns:
            BaseFetcher: Ready-to-use fetcher

        Raises:
            ValueError: If fetcher type is unknown
        """
        fetcher_type = fetcher_type or settings.active_fetcher
        config = settings.get_fetcher_config(fetcher_type)

        if fetcher_type == "pubmed":
            api_key = settings.get_fetcher_api_key("pubmed")
            return PubMedFetcher(config, api_key)

        elif fetcher_type == "arxiv":
            return ArXivFetcher(config)

        elif fetcher_type == "pmc":
            # Placeholder for PMC fetcher
            raise NotImplementedError("PMC fetcher not yet implemented")

        else:
            raise ValueError(
                f"Unknown fetcher type: {fetcher_type}. Supported: pubmed, arxiv, pmc"
            )


def get_fetcher(fetcher_type: Optional[str] = None) -> BaseFetcher:
    """
    Create and return paper fetcher

    Args:
        fetcher_type: "pubmed", "pmc", etc. (if None, uses config default)

    Returns:
        BaseFetcher: Ready-to-use fetcher

    Example:
        >>> from src.fetchers import get_fetcher
        >>>
        >>> # PubMed example
        >>> fetcher = get_fetcher("pubmed")
        >>> papers = fetcher.search("caloric restriction aging", max_results=5)
        >>> for pmid in papers:
        ...     paper = fetcher.fetch_paper(pmid)
        ...     print(paper.title)
        >>>
        >>> # arXiv example
        >>> arxiv_fetcher = get_fetcher("arxiv")
        >>> arxiv_ids = arxiv_fetcher.search("knowledge graph extraction", max_results=5)
        >>> for arxiv_id in arxiv_ids:
        ...     paper = arxiv_fetcher.fetch_paper(arxiv_id)
        ...     print(paper.title)
    """
    return FetcherFactory.create(fetcher_type)
