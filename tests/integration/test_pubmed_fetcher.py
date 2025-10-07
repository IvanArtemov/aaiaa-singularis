"""Integration tests for PubMed fetcher"""

import pytest
from src.fetchers import get_fetcher
from src.fetchers.base_fetcher import PaperMetadata


class TestPubMedFetcherIntegration:
    """Integration tests for PubMed fetcher (requires internet connection)"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup fetcher for all tests"""
        self.fetcher = get_fetcher("pubmed")

    def test_search_returns_results(self):
        """Test that search returns PMIDs"""
        pmids = self.fetcher.search("aging", max_results=2)

        assert isinstance(pmids, list)
        assert len(pmids) > 0
        assert len(pmids) <= 2

        # PMIDs should be strings of numbers
        for pmid in pmids:
            assert isinstance(pmid, str)
            assert pmid.isdigit()

    def test_fetch_paper_returns_metadata(self):
        """Test fetching paper metadata"""
        # Use a known PMID (this paper should exist)
        pmid = "29080073"  # Known aging research paper

        paper = self.fetcher.fetch_paper(pmid)

        assert isinstance(paper, PaperMetadata)
        assert paper.pmid == pmid
        assert len(paper.title) > 0
        assert len(paper.authors) > 0
        assert len(paper.abstract) > 0
        assert len(paper.journal) > 0

    def test_search_and_fetch_workflow(self):
        """Test complete workflow: search + fetch"""
        # Search for papers
        pmids = self.fetcher.search("caloric restriction", max_results=2)
        assert len(pmids) > 0

        # Fetch first paper
        paper = self.fetcher.fetch_paper(pmids[0])

        # Verify paper has required fields
        assert paper.pmid == pmids[0]
        assert paper.title is not None
        assert len(paper.title) > 0
        assert len(paper.authors) > 0

    def test_search_and_fetch_convenience_method(self):
        """Test search_and_fetch convenience method"""
        papers = self.fetcher.search_and_fetch("rapamycin", max_results=2)

        assert isinstance(papers, list)
        assert len(papers) > 0
        assert len(papers) <= 2

        # Check first paper
        paper = papers[0]
        assert isinstance(paper, PaperMetadata)
        assert paper.pmid is not None
        assert len(paper.title) > 0

    def test_paper_metadata_structure(self):
        """Test that returned paper has correct structure"""
        pmid = "29080073"
        paper = self.fetcher.fetch_paper(pmid)

        # Check required fields exist
        assert hasattr(paper, 'pmid')
        assert hasattr(paper, 'title')
        assert hasattr(paper, 'authors')
        assert hasattr(paper, 'abstract')
        assert hasattr(paper, 'journal')
        assert hasattr(paper, 'publication_date')
        assert hasattr(paper, 'keywords')
        assert hasattr(paper, 'doi')

        # Check types
        assert isinstance(paper.pmid, (str, type(None)))
        assert isinstance(paper.title, str)
        assert isinstance(paper.authors, list)
        assert isinstance(paper.abstract, str)
        assert isinstance(paper.journal, str)

    def test_rate_limiting_doesnt_fail(self):
        """Test that rate limiting doesn't cause failures"""
        # Make multiple requests quickly
        for i in range(3):
            pmids = self.fetcher.search(f"aging test{i}", max_results=1)
            assert len(pmids) >= 0  # May return 0 for nonsense queries

        # If we get here, rate limiting worked correctly

    @pytest.mark.parametrize("query,expected_min_results", [
        ("aging", 1),
        ("caloric restriction longevity", 1),
        ("rapamycin mtor", 1),
    ])
    def test_various_search_queries(self, query, expected_min_results):
        """Test different search queries"""
        pmids = self.fetcher.search(query, max_results=5)
        assert len(pmids) >= expected_min_results


@pytest.mark.slow
class TestPubMedFetcherSlow:
    """Slower integration tests (run with pytest -m slow or pytest --run-slow)"""

    def test_fetch_multiple_papers(self, run_slow):
        """Test fetching multiple papers"""
        if not run_slow:
            pytest.skip("Slow test - use --run-slow to run")

        fetcher = get_fetcher("pubmed")

        pmids = fetcher.search("aging", max_results=5)
        papers = fetcher.fetch_multiple(pmids)

        assert len(papers) > 0
        assert all(isinstance(p, PaperMetadata) for p in papers)
        assert all(p.pmid in pmids for p in papers)
