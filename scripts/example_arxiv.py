"""
Example: arXiv Fetcher Usage

Demonstrates how to use ArXivFetcher to search and download papers from arXiv.

Usage:
    python scripts/example_arxiv.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fetchers import get_fetcher
from src.utils.article_registry import ArticleRegistry, ArticleRecord
import os


def print_separator(title: str = ""):
    """Print formatted separator"""
    if title:
        print(f"\n{'=' * 80}")
        print(f"  {title}")
        print(f"{'=' * 80}\n")
    else:
        print(f"{'=' * 80}\n")


def example_basic_search():
    """Example 1: Basic search"""
    print_separator("Example 1: Basic Search")

    # Create arXiv fetcher
    fetcher = get_fetcher("arxiv")

    # Search for papers
    query = "knowledge graph extraction"
    print(f"Searching for: '{query}'")
    print(f"Max results: 5\n")

    arxiv_ids = fetcher.search(query, max_results=5)

    print(f"Found {len(arxiv_ids)} papers:")
    for i, arxiv_id in enumerate(arxiv_ids, 1):
        print(f"  {i}. arXiv:{arxiv_id}")


def example_fetch_metadata():
    """Example 2: Fetch paper metadata"""
    print_separator("Example 2: Fetch Paper Metadata")

    fetcher = get_fetcher("arxiv")

    # Fetch a specific paper (example: "End-to-End NLP Knowledge Graph Construction")
    arxiv_id = "2106.01167"
    print(f"Fetching metadata for arXiv:{arxiv_id}\n")

    paper = fetcher.fetch_paper(arxiv_id)

    print(f"Title: {paper.title}")
    print(f"Authors: {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}")
    print(f"Publication Date: {paper.publication_date}")
    print(f"Journal: {paper.journal}")
    print(f"Categories: {', '.join(paper.keywords)}")
    print(f"DOI: {paper.doi or 'N/A'}")
    print(f"PDF URL: {paper.pdf_url}")
    print(f"\nAbstract (first 300 chars):")
    print(f"  {paper.abstract[:300]}...")


def example_category_search():
    """Example 3: Search by category"""
    print_separator("Example 3: Search by Category")

    fetcher = get_fetcher("arxiv")

    # Search in specific categories
    categories = ["cs.CL", "cs.AI"]
    query = "named entity recognition"

    print(f"Searching in categories: {', '.join(categories)}")
    print(f"Additional query: '{query}'")
    print(f"Max results: 5\n")

    arxiv_ids = fetcher.search_by_category(
        categories=categories,
        additional_query=query,
        max_results=5
    )

    print(f"Found {len(arxiv_ids)} papers:")
    for i, arxiv_id in enumerate(arxiv_ids, 1):
        paper = fetcher.fetch_paper(arxiv_id)
        print(f"\n  {i}. arXiv:{arxiv_id}")
        print(f"     Title: {paper.title}")
        print(f"     Categories: {', '.join(paper.keywords)}")


def example_download_pdf():
    """Example 4: Download PDF"""
    print_separator("Example 4: Download PDF")

    fetcher = get_fetcher("arxiv")

    # Download a specific paper
    arxiv_id = "2106.01167"
    output_dir = "articles/arxiv_examples"

    print(f"Downloading PDF for arXiv:{arxiv_id}")
    print(f"Output directory: {output_dir}\n")

    pdf_path = fetcher.download_pdf(arxiv_id, output_dir=output_dir)

    if pdf_path:
        file_size = os.path.getsize(pdf_path)
        print(f"\n✓ Successfully downloaded to: {pdf_path}")
        print(f"  File size: {file_size / 1024:.1f} KB")
    else:
        print(f"\n✗ Failed to download PDF")


def example_registry_integration():
    """Example 5: Integration with Article Registry"""
    print_separator("Example 5: Registry Integration")

    fetcher = get_fetcher("arxiv")
    registry = ArticleRegistry(registry_path="articles/metadata_example.json")

    # Search and register papers
    query = "large language models knowledge graph"
    print(f"Searching for: '{query}'")
    print(f"Max results: 3\n")

    arxiv_ids = fetcher.search(query, max_results=3)

    print(f"Found {len(arxiv_ids)} papers. Downloading and registering...\n")

    for i, arxiv_id in enumerate(arxiv_ids, 1):
        # Check if already registered
        if registry.exists(arxiv_id):
            print(f"  {i}. arXiv:{arxiv_id} - Already in registry, skipping")
            continue

        # Fetch metadata
        paper = fetcher.fetch_paper(arxiv_id)

        # Download PDF
        pdf_path = fetcher.download_pdf(arxiv_id, output_dir="articles/arxiv_examples")

        if pdf_path:
            # Create registry record
            record = ArticleRecord(
                pmid=None,
                pmc_id=None,
                arxiv_id=arxiv_id,
                doi=paper.doi,
                title=paper.title,
                authors=paper.authors,
                journal=paper.journal,
                publication_date=paper.publication_date,
                pdf_path=pdf_path,
                file_size=os.path.getsize(pdf_path),
                download_source="arXiv (Example)"
            )

            # Add to registry
            registry.add(record)
            print(f"  {i}. arXiv:{arxiv_id} - Downloaded and registered")
            print(f"     Title: {paper.title}")
        else:
            print(f"  {i}. arXiv:{arxiv_id} - Download failed")

    # Print registry stats
    print(f"\nRegistry statistics:")
    stats = registry.get_stats()
    print(f"  Total articles: {stats['total_articles']}")
    print(f"  Total size: {stats['total_size_mb']} MB")
    print(f"  With arXiv ID: {stats['with_arxiv_id']}")


def main():
    print_separator("arXiv Fetcher Examples")

    print("This script demonstrates various uses of the ArXivFetcher.\n")
    print("Examples:")
    print("  1. Basic search")
    print("  2. Fetch paper metadata")
    print("  3. Search by category")
    print("  4. Download PDF")
    print("  5. Registry integration")

    # Run all examples
    try:
        example_basic_search()
        example_fetch_metadata()
        example_category_search()
        example_download_pdf()
        example_registry_integration()

        print_separator("All Examples Completed")
        print("✓ All examples ran successfully!")
        print("\nNext steps:")
        print("  - Use batch_download_arxiv_kg.py to download 300-500 papers")
        print("  - Parse PDFs with src/parsers/pdf_parser.py")
        print("  - Extract entities with src/pipelines/")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
