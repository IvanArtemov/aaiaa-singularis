"""
Demo: Searching PubMed for Free Full Text Articles

This script demonstrates how to search PubMed for articles with free full text
using the PubMedFetcher class with the free_full_text filter.

Usage:
    python examples/pubmed_free_full_text_demo.py

Modify the SEARCH_QUERY variable to change the search term.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fetchers.pubmed_fetcher import PubMedFetcher
import yaml


def print_separator(title: str = ""):
    """Print a formatted separator line"""
    if title:
        print(f"\n{'=' * 80}")
        print(f"  {title}")
        print(f"{'=' * 80}\n")
    else:
        print(f"{'=' * 80}\n")


def main():
    # Configuration
    SEARCH_QUERY = "crispr cas9"  # Change this to search for different topics
    MAX_RESULTS = 10
    USE_FREE_FULL_TEXT_FILTER = True  # Set to False to search all articles

    print_separator("PubMed Free Full Text Search Demo")

    # Load configuration from YAML
    config_path = Path(__file__).parent.parent / "src/config/fetcher_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    fetcher_config = config.get("pubmed", {})

    # Get API key from environment (optional but recommended)
    api_key = os.getenv("NCBI_API_KEY", "")
    tool_name = os.getenv("NCBI_TOOL_NAME", "")
    email = os.getenv("NCBI_EMAIL", "")

    print("Configuration Status:")
    if api_key:
        print("  ✓ NCBI API key found (10 requests/sec limit)")
    else:
        print("  ⚠ No API key (3 requests/sec limit)")
        print("    Get one at: https://www.ncbi.nlm.nih.gov/account/settings/")

    if tool_name and email:
        print(f"  ✓ Tool name: {tool_name}")
        print(f"  ✓ Email: {email}")
    else:
        print("  🚨 WARNING: NCBI_TOOL_NAME or NCBI_EMAIL not set!")
        print("     Your IP may be blocked without these!")
        print("     Set them in .env file immediately.")

    # Initialize fetcher
    fetcher = PubMedFetcher(config=fetcher_config, api_key=api_key, tool_name=tool_name, email=email)

    # Perform search
    print_separator("Step 1: Searching PubMed")

    print(f"Query: '{SEARCH_QUERY}'")
    print(f"Max Results: {MAX_RESULTS}")
    print(f"Free Full Text Filter: {'ENABLED' if USE_FREE_FULL_TEXT_FILTER else 'DISABLED'}")
    print()

    try:
        pmids = fetcher.search(
            query=SEARCH_QUERY,
            max_results=MAX_RESULTS,
            free_full_text=USE_FREE_FULL_TEXT_FILTER
        )

        print(f"✓ Found {len(pmids)} articles")

        if not pmids:
            print("\nNo articles found. Try:")
            print("  - Using a different search query")
            print("  - Disabling the free full text filter")
            print("  - Increasing MAX_RESULTS")
            return

        print(f"  PMIDs: {', '.join(pmids[:5])}" + ("..." if len(pmids) > 5 else ""))

    except Exception as e:
        print(f"✗ Error during search: {e}")
        return

    # Fetch article metadata
    print_separator("Step 2: Fetching Article Metadata")

    papers = []
    for i, pmid in enumerate(pmids, 1):
        try:
            print(f"[{i}/{len(pmids)}] Fetching PMID {pmid}...", end=" ")
            paper = fetcher.fetch_paper(pmid)
            papers.append(paper)
            print("✓")
        except Exception as e:
            print(f"✗ Error: {e}")
            continue

    print(f"\n✓ Successfully fetched {len(papers)} articles")

    # Display results
    print_separator("Step 3: Results")

    for i, paper in enumerate(papers, 1):
        print(f"\n[{i}] {paper.title}")
        print(f"    PMID: {paper.pmid}")
        print(f"    DOI: {paper.doi or 'N/A'}")
        print(f"    PMC ID: {paper.pmc_id or 'N/A'}")
        print(f"    Journal: {paper.journal}")
        print(f"    Date: {paper.publication_date}")
        print(f"    Authors: {', '.join(paper.authors[:3])}" +
              ("..." if len(paper.authors) > 3 else ""))
        print(f"    Free Full Text: {'✓ YES' if paper.has_free_full_text else '✗ NO'}")

        if paper.abstract:
            abstract_preview = paper.abstract[:200] + "..." if len(paper.abstract) > 200 else paper.abstract
            print(f"    Abstract: {abstract_preview}")

    # Statistics
    print_separator("Statistics")

    total = len(papers)
    with_full_text = sum(1 for p in papers if p.has_free_full_text)
    with_pmc = sum(1 for p in papers if p.pmc_id)
    with_doi = sum(1 for p in papers if p.doi)

    print(f"Total articles fetched: {total}")
    print(f"With free full text: {with_full_text} ({with_full_text/total*100:.1f}%)")
    print(f"With PMC ID: {with_pmc} ({with_pmc/total*100:.1f}%)")
    print(f"With DOI: {with_doi} ({with_doi/total*100:.1f}%)")

    # Additional examples
    print_separator("Additional Search Examples")

    print("You can modify the search query to explore different topics:")
    print()
    print("  # Recent reviews on aging:")
    print("  SEARCH_QUERY = 'aging AND review[pt] AND 2023:2024[pdat]'")
    print()
    print("  # Clinical trials on Alzheimer's:")
    print("  SEARCH_QUERY = 'alzheimer AND clinical trial[pt]'")
    print()
    print("  # Meta-analyses on cancer:")
    print("  SEARCH_QUERY = 'cancer AND meta-analysis[pt]'")
    print()
    print("  # Specific author:")
    print("  SEARCH_QUERY = 'diabetes AND Smith J[au]'")
    print()
    print("  # Specific journal:")
    print("  SEARCH_QUERY = 'crispr AND Nature[journal]'")
    print()

    print("See docs/pubmed_api_reference.md for complete API documentation")

    print_separator()


if __name__ == "__main__":
    main()
