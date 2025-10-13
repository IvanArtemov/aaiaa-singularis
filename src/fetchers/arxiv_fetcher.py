"""arXiv fetcher using arxiv.py library"""

import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import arxiv
from .base_fetcher import BaseFetcher, PaperMetadata


class ArXivFetcher(BaseFetcher):
    """Fetcher for arXiv using arxiv.py library"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Rate limiting
        self.requests_per_second = config.get("rate_limit", {}).get("requests_per_second", 1)
        self.delay_between_requests = 1.0 / self.requests_per_second
        self.last_request_time = 0

        # Default parameters
        self.default_max_results = config.get("search", {}).get("max_results", 100)
        self.default_sort_by = config.get("search", {}).get("sort_by", "relevance")

        # Create reusable client
        self.client = arxiv.Client(
            page_size=100,
            delay_seconds=self.delay_between_requests,
            num_retries=3
        )

    def _rate_limit(self):
        """Enforce rate limiting"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.delay_between_requests:
            time.sleep(self.delay_between_requests - time_since_last_request)
        self.last_request_time = time.time()

    def _get_sort_criterion(self, sort_by: str) -> arxiv.SortCriterion:
        """Convert string sort parameter to arxiv.SortCriterion"""
        sort_map = {
            "relevance": arxiv.SortCriterion.Relevance,
            "last_updated": arxiv.SortCriterion.LastUpdatedDate,
            "submitted": arxiv.SortCriterion.SubmittedDate,
        }
        return sort_map.get(sort_by, arxiv.SortCriterion.Relevance)

    def search(
        self,
        query: str,
        max_results: int = 10,
        sort_by: str = "relevance",
        **kwargs
    ) -> List[str]:
        """
        Search arXiv using arxiv.py

        Args:
            query: Search query (e.g., "knowledge graph extraction")
                   Supports arXiv query operators:
                   - ti: title search
                   - au: author search
                   - abs: abstract search
                   - cat: category search
                   Example: "ti:knowledge graph AND cat:cs.CL"
            max_results: Maximum number of results
            sort_by: Sort order ("relevance", "last_updated", "submitted")
            **kwargs: Additional parameters (categories filter)

        Returns:
            List of arXiv IDs (e.g., ["2106.01167", "2502.14192"])
        """
        self._rate_limit()

        sort_criterion = self._get_sort_criterion(sort_by)

        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        arxiv_ids = []
        try:
            for result in self.client.results(search):
                # Extract ID without version (2106.01167v1 -> 2106.01167)
                arxiv_id = result.entry_id.split('/')[-1]
                if 'v' in arxiv_id:
                    arxiv_id = arxiv_id.split('v')[0]
                arxiv_ids.append(arxiv_id)
        except Exception as e:
            print(f"Error during arXiv search: {e}")

        return arxiv_ids

    def fetch_paper(self, paper_id: str) -> PaperMetadata:
        """
        Fetch paper metadata from arXiv

        Args:
            paper_id: arXiv ID (e.g., "2106.01167" or "2106.01167v1")

        Returns:
            PaperMetadata: Paper information
        """
        self._rate_limit()

        # Search by ID
        search = arxiv.Search(id_list=[paper_id])

        try:
            result = next(self.client.results(search))
            return self._parse_arxiv_result(result)
        except StopIteration:
            raise ValueError(f"Paper {paper_id} not found on arXiv")
        except Exception as e:
            raise ValueError(f"Error fetching paper {paper_id}: {e}")

    def _parse_arxiv_result(self, result: arxiv.Result) -> PaperMetadata:
        """Parse arxiv.Result to PaperMetadata"""

        # Extract clean arXiv ID without version
        arxiv_id = result.entry_id.split('/')[-1]
        if 'v' in arxiv_id:
            arxiv_id = arxiv_id.split('v')[0]

        # Extract authors
        authors = [author.name for author in result.authors]

        # Extract categories
        categories = result.categories

        # Format publication date
        pub_date = result.published.strftime("%Y-%m-%d") if result.published else ""

        # Extract DOI if available
        doi = result.doi if hasattr(result, 'doi') and result.doi else None

        # Get PDF URL
        pdf_url = result.pdf_url

        return PaperMetadata(
            pmid=None,  # arXiv doesn't have PMID
            pmc_id=None,  # arXiv doesn't have PMC ID
            arxiv_id=arxiv_id,  # arXiv ID
            doi=doi,
            title=result.title,
            authors=authors,
            abstract=result.summary,
            journal=result.journal_ref if result.journal_ref else f"arXiv:{arxiv_id}",
            publication_date=pub_date,
            keywords=categories,  # Use arXiv categories as keywords
            full_text_url=result.entry_id,
            pdf_url=pdf_url,
            citations_count=0,  # arXiv API doesn't provide citation count
            has_free_full_text=True  # All arXiv papers have free full text
        )

    def fetch_full_text(self, paper_id: str) -> Optional[str]:
        """
        Fetch full text from arXiv

        Note: arXiv provides PDFs, not plain text.
        This method returns None. Use download_pdf() instead.

        Args:
            paper_id: arXiv ID

        Returns:
            None (use download_pdf instead)
        """
        print("arXiv provides PDFs, not plain text. Use download_pdf() instead.")
        return None

    def download_pdf(self, paper_id: str, output_dir: str = "articles/pdfs") -> Optional[str]:
        """
        Download PDF from arXiv

        Args:
            paper_id: arXiv ID (e.g., "2106.01167")
            output_dir: Directory to save PDF

        Returns:
            Path to downloaded PDF file
        """
        self._rate_limit()

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            # Search by ID
            search = arxiv.Search(id_list=[paper_id])
            result = next(self.client.results(search))

            # Extract clean ID for filename
            clean_id = paper_id.replace('/', '_').replace('v', '_')
            filename = f"{clean_id}.pdf"
            output_file = output_path / filename

            # Download PDF
            result.download_pdf(dirpath=str(output_path), filename=filename)

            file_size = output_file.stat().st_size
            print(f"✓ Downloaded PDF to {output_file} ({file_size / 1024:.1f} KB)")

            return str(output_file)

        except StopIteration:
            print(f"✗ Paper {paper_id} not found on arXiv")
            return None
        except Exception as e:
            print(f"✗ Download failed for {paper_id}: {e}")
            return None

    def search_by_category(
        self,
        categories: List[str],
        additional_query: str = "",
        max_results: int = 10,
        sort_by: str = "relevance"
    ) -> List[str]:
        """
        Search arXiv by categories with optional additional query

        Args:
            categories: List of arXiv categories (e.g., ["cs.CL", "cs.AI"])
            additional_query: Additional search terms (e.g., "knowledge graph")
            max_results: Maximum number of results
            sort_by: Sort order

        Returns:
            List of arXiv IDs

        Example:
            >>> fetcher.search_by_category(
            ...     categories=["cs.CL", "cs.AI"],
            ...     additional_query="knowledge graph extraction",
            ...     max_results=50
            ... )
        """
        # # Build category query
        # if len(categories) == 1:
        #     cat_query = f"cat:{categories[0]}"
        # else:
        #     cat_parts = [f"cat:{cat}" for cat in categories]
        #     cat_query = f"({' OR '.join(cat_parts)})"

        # Combine with additional query
        # if additional_query:
        #     query = f"{additional_query} AND {cat_query}"
        # else:
        #     query = cat_query

        return self.search(additional_query, max_results=max_results, sort_by=sort_by)
