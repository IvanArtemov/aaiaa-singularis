"""Pytest configuration and fixtures"""

import pytest
import os
from pathlib import Path


@pytest.fixture
def project_root():
    """Get project root directory"""
    return Path(__file__).parent.parent


@pytest.fixture
def mock_fetcher_config():
    """Mock fetcher configuration"""
    return {
        "base_url": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
        "database": "pubmed",
        "endpoints": {
            "search": "esearch.fcgi",
            "fetch": "efetch.fcgi"
        },
        "rate_limit": {
            "requests_per_second": 3,
            "delay_between_requests": 0.35
        },
        "timeout": 30
    }

# 29080073
@pytest.fixture
def sample_paper_metadata():
    """Sample paper metadata for testing"""
    from src.fetchers.base_fetcher import PaperMetadata

    return PaperMetadata(
        pmid="12345678",
        title="Sample Paper on Aging Research",
        authors=["John Doe", "Jane Smith"],
        abstract="This is a sample abstract about aging research.",
        journal="Nature Aging",
        publication_date="2024-01-15",
        keywords=["aging", "longevity", "senescence"],
        doi="10.1234/nature.12345"
    )


@pytest.fixture
def skip_if_no_api_key():
    """Skip test if NCBI API key is not set"""
    api_key = os.getenv("NCBI_API_KEY")
    if not api_key:
        pytest.skip("NCBI_API_KEY not set - skipping integration test")
    return api_key


def pytest_addoption(parser):
    """Add custom pytest options"""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests"
    )


@pytest.fixture
def run_slow(request):
    """Fixture to check if --run-slow flag is set"""
    return request.config.getoption("--run-slow")
