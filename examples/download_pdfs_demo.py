"""
Demo: Download PDFs from PubMed Central with Article Registry

This script demonstrates how to:
1. Search for free full-text articles
2. Download PDFs
3. Track downloads in a registry
4. Avoid re-downloading

Usage:
    python examples/download_pdfs_demo.py
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fetchers.pubmed_fetcher import PubMedFetcher
from src.utils.article_registry import ArticleRegistry, ArticleRecord
import yaml


def print_separator(title: str = ""):
    """Print formatted separator"""
    if title:
        print(f"\n{'=' * 80}")
        print(f"  {title}")
        print(f"{'=' * 80}\n")
    else:
        print(f"{'=' * 80}\n")


def main():
    # Configuration
    SEARCH_QUERY = "crispr cas9"
    MAX_RESULTS = 5  # Download only 5 packages for demo
    OUTPUT_DIR = "articles/packages"  # tar.gz packages
    REGISTRY_PATH = "articles/metadata.json"

    print_separator("PubMed Package Download Demo with Registry")

    # Load configuration
    config_path = Path(__file__).parent.parent / "src/config/fetcher_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    fetcher_config = config.get("pubmed", {})

    # Get credentials
    api_key = os.getenv("NCBI_API_KEY", "")
    tool_name = os.getenv("NCBI_TOOL_NAME", "")
    email = os.getenv("NCBI_EMAIL", "")

    # Initialize fetcher and registry
    fetcher = PubMedFetcher(config=fetcher_config, api_key=api_key, tool_name=tool_name, email=email)
    registry = ArticleRegistry(registry_path=REGISTRY_PATH)

    print(f"Registry status:")
    print(f"  Total articles: {registry.count()}")
    print(f"  Registry file: {REGISTRY_PATH}")

    # Search for articles
    print_separator("Step 1: Searching for Free Full Text Articles")

    print(f"Query: '{SEARCH_QUERY}'")
    print(f"Max Results: {MAX_RESULTS}")
    print(f"Filter: Free Full Text ENABLED\n")

    try:
        pmids = fetcher.search(SEARCH_QUERY, max_results=MAX_RESULTS, free_full_text=True)
        print(f"✓ Found {len(pmids)} articles\n")
    except Exception as e:
        print(f"✗ Search failed: {e}")
        return

    if not pmids:
        print("No articles found.")
        return

    # Fetch metadata
    print_separator("Step 2: Fetching Article Metadata")

    papers = []
    for i, pmid in enumerate(pmids, 1):
        try:
            print(f"[{i}/{len(pmids)}] Fetching metadata for PMID {pmid}...")
            paper = fetcher.fetch_paper(pmid)
            papers.append(paper)
            print(f"  Title: {paper.title}")
            print(f"  PMC ID: {paper.pmc_id or 'N/A'}")
            print(f"  Free Full Text: {'✓' if paper.has_free_full_text else '✗'}\n")
        except Exception as e:
            print(f"  ✗ Error: {e}\n")

    # Download packages
    print_separator("Step 3: Downloading tar.gz Packages")

    downloaded = 0
    skipped = 0
    failed = 0

    for i, paper in enumerate(papers, 1):
        pmid = paper.pmid
        print(f"\n[{i}/{len(papers)}] Processing PMID {pmid}")
        print(f"  Title: {paper.title}")

        # Check if already downloaded
        if registry.exists(pmid):
            record = registry.get(pmid)
            print(f"  ⚠ Already downloaded: {record.pdf_path}")
            print(f"  Downloaded at: {record.downloaded_at}")
            skipped += 1
            continue

        # Check if has PMC ID
        if not paper.pmc_id:
            print(f"  ✗ No PMC ID - cannot download package")
            failed += 1
            continue

        # Download tar.gz package
        try:
            # Use PMC ID from metadata if available (faster), otherwise use PMID
            paper_id = paper.pmc_id if paper.pmc_id else pmid
            package_path = fetcher.download_pdf(paper_id, output_dir=OUTPUT_DIR)

            if package_path:
                # Get file size
                file_size = os.path.getsize(package_path)

                # Add to registry
                record = ArticleRecord(
                    pmid=paper.pmid,
                    pmc_id=paper.pmc_id,
                    doi=paper.doi,
                    title=paper.title,
                    authors=paper.authors,
                    journal=paper.journal,
                    publication_date=paper.publication_date,
                    pdf_path=package_path,  # stores tar.gz path
                    file_size=file_size,
                    download_source="PMC_OA"
                )

                registry.add(record)
                downloaded += 1
                print(f"  ✓ Added to registry")
            else:
                failed += 1
        except Exception as e:
            print(f"  ✗ Download failed: {e}")
            failed += 1

    # Summary
    print_separator("Download Summary")

    print(f"Total articles processed: {len(papers)}")
    print(f"  ✓ Downloaded: {downloaded}")
    print(f"  ⚠ Skipped (already exists): {skipped}")
    print(f"  ✗ Failed: {failed}")

    # Registry statistics
    print_separator("Registry Statistics")

    stats = registry.get_stats()
    print(f"Total articles in registry: {stats['total_articles']}")
    print(f"Total size: {stats['total_size_mb']} MB")
    print(f"Articles with PMC ID: {stats['with_pmc_id']}")
    print(f"Articles with DOI: {stats['with_doi']}")
    print(f"\nBy source:")
    for source, count in stats['by_source'].items():
        print(f"  {source}: {count}")

    # List all articles
    print_separator("All Downloaded Articles")

    all_articles = registry.list_all()
    for i, article in enumerate(all_articles, 1):
        print(f"[{i}] {article.title[:60]}...")
        print(f"    PMID: {article.pmid} | PMC: {article.pmc_id or 'N/A'}")
        print(f"    PDF: {article.pdf_path}")
        print(f"    Size: {article.file_size / 1024:.1f} KB")
        print(f"    Downloaded: {article.downloaded_at[:19]}\n")

    print_separator()

    print(f"\n✓ All tar.gz packages saved to: {OUTPUT_DIR}")
    print(f"✓ Registry saved to: {REGISTRY_PATH}")
    print(f"\nRun this script again to see deduplication in action!")
    print(f"\nNote: Packages are in tar.gz format. Extract them to get PDFs.")


if __name__ == "__main__":
    main()
