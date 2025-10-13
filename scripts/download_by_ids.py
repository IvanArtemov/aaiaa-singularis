"""
Download arXiv Papers by ID List

This script downloads specific arXiv papers given their IDs.

Usage:
    python scripts/download_by_ids.py
"""

import sys
import os
from pathlib import Path
from typing import List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fetchers.arxiv_fetcher import ArXivFetcher
from src.utils.article_registry import ArticleRegistry, ArticleRecord
import yaml


# List of arXiv IDs to download
ARXIV_IDS = [
    # === Высокорелевантные: Knowledge Graph Extraction ===
    "2310.15572",  # Natural Language Processing for Drug Discovery Knowledge Graphs
    "2011.01103",  # Generating Knowledge Graphs by Employing NLP and ML Techniques

    # === Частично релевантные: Biomedical & Text-based KG ===
    "2207.14294",  # Knowledge-Driven Mechanistic Enrichment
    "2001.08392",  # Towards context in large scale biomedical knowledge graphs
    "1908.09354",  # Unsupervised Construction of Knowledge Graphs From Text and Code

    # === Высокая релевантность: Визуализация и академические графы ===
    "2506.17508",  # Mapping the Evolution of Research Contributions using KnoVo
    "2403.02576",  # AceMap: Knowledge Discovery through Academic Graph
    "1906.04800",  # Visualizing a Field of Research: Systematic Scientometric Reviews

    # === Средняя релевантность: Цитирование и коллаборации ===
    "2501.04015",  # Graph Analysis of Citation and Co-authorship Networks
    "2405.07267",  # How Researchers Browse Citation Network Visualizations
    "2204.11194",  # Co-citation and Co-authorship Networks of Statisticians
    "2009.13059",  # Visual Exploration and Knowledge Discovery from Biomedical Dark Data
    "2008.11844",  # Argo Lite: Interactive Graph Exploration and Visualization
    "1804.03261",  # Juniper: A Tree+Table Approach to Multivariate Graph Visualization
]

# Configuration
OUTPUT_DIR = "articles/arxiv_pdfs"
REGISTRY_PATH = "articles/metadata.json"


def print_separator(title: str = "", char: str = "="):
    """Print formatted separator"""
    if title:
        print(f"\n{char * 80}")
        print(f"  {title}")
        print(f"{char * 80}\n")
    else:
        print(f"{char * 80}\n")


def main():
    print_separator("Download arXiv Papers by ID")

    # Load configuration
    config_path = Path(__file__).parent.parent / "src/config/fetcher_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    fetcher_config = config.get("arxiv", {})

    # Initialize fetcher and registry
    fetcher = ArXivFetcher(config=fetcher_config)
    registry = ArticleRegistry(registry_path=REGISTRY_PATH)

    print(f"Configuration:")
    print(f"  Total papers to download: {len(ARXIV_IDS)}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Registry: {REGISTRY_PATH}")
    print(f"\nCurrent registry status:")
    print(f"  Total articles: {registry.count()}")

    # Statistics
    downloaded = 0
    skipped = 0
    failed = 0

    print_separator("Downloading Papers", char="-")

    # Process each arXiv ID
    for i, arxiv_id in enumerate(ARXIV_IDS, 1):
        print(f"\n[{i}/{len(ARXIV_IDS)}] Processing arXiv:{arxiv_id}")

        # Check if already downloaded
        if registry.exists(arxiv_id):
            print(f"  ✓ Already in registry, skipping")
            skipped += 1
            continue

        try:
            # Fetch metadata
            print(f"  Fetching metadata...")
            paper = fetcher.fetch_paper(arxiv_id)
            print(f"  Title: {paper.title}")
            print(f"  Authors: {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}")
            print(f"  Date: {paper.publication_date}")

            # Download PDF
            print(f"  Downloading PDF...")
            pdf_path = fetcher.download_pdf(arxiv_id, output_dir=OUTPUT_DIR)

            if pdf_path:
                # Get file size
                file_size = os.path.getsize(pdf_path)

                # Add to registry
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
                    file_size=file_size,
                    download_source="arXiv (Manual ID List)"
                )

                registry.add(record)
                downloaded += 1
                print(f"  ✓ Successfully downloaded ({file_size / 1024:.1f} KB)")
            else:
                failed += 1
                print(f"  ✗ Download failed")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            failed += 1

    # Final summary
    print_separator("DOWNLOAD COMPLETE")

    print(f"Statistics:")
    print(f"  Total papers: {len(ARXIV_IDS)}")
    print(f"  Downloaded: {downloaded}")
    print(f"  Skipped (already exists): {skipped}")
    print(f"  Failed: {failed}")

    if len(ARXIV_IDS) > 0:
        success_rate = (downloaded / len(ARXIV_IDS)) * 100
        print(f"  Success rate: {success_rate:.1f}%")

    # Registry statistics
    print(f"\nRegistry after download:")
    stats = registry.get_stats()
    print(f"  Total articles: {stats['total_articles']}")
    print(f"  Total size: {stats['total_size_mb']} MB")
    print(f"  Articles with arXiv ID: {stats['with_arxiv_id']}")

    print(f"\n✓ PDFs saved to: {OUTPUT_DIR}")
    print(f"✓ Registry saved to: {REGISTRY_PATH}")

    print(f"\nNext steps:")
    print(f"  1. Parse PDFs: python scripts/example_pdf_parser.py")
    print(f"  2. Extract entities: python scripts/example_llm_pipeline.py")
    print(f"  3. Generate knowledge graph: python scripts/generate_svg.py")


if __name__ == "__main__":
    main()