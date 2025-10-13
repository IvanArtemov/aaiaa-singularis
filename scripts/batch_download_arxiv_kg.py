"""
Batch Download: 300-500 Articles on Knowledge Graph Extraction from arXiv

This script downloads articles on knowledge graph extraction methods from arXiv
to improve the methodology for the Singularis Challenge hackathon.

Usage:
    python scripts/batch_download_arxiv_kg.py
"""

import sys
import os
from pathlib import Path
from typing import List, Dict
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fetchers.arxiv_fetcher import ArXivFetcher
from src.utils.article_registry import ArticleRegistry, ArticleRecord
import yaml


# 10 Interconnected Topics for Knowledge Graph Extraction Research
# NOTE: Queries simplified to improve search results on arXiv
TOPICS = [
    {
        "query": "knowledge graph",
        "categories": ["cs.CL", "cs.AI"],
        "description": "Knowledge graph construction from text",
        "block": "Core KG Construction"
    },
    {
        "query": "named entity recognition",
        "categories": ["cs.CL", "cs.IR"],
        "description": "NER in scientific and biomedical text",
        "block": "Entity Extraction"
    },
    {
        "query": "relation extraction",
        "categories": ["cs.CL", "cs.IR"],
        "description": "Relation extraction from scientific literature",
        "block": "Relation Extraction"
    },
    {
        "query": "information extraction",
        "categories": ["cs.DL", "cs.CL"],
        "description": "General information extraction from papers",
        "block": "Information Extraction"
    },
    {
        "query": "graph neural network",
        "categories": ["cs.LG", "cs.CL"],
        "description": "Graph neural networks for NLP tasks",
        "block": "Graph Learning"
    },
    {
        "query": "semantic parsing",
        "categories": ["cs.AI", "cs.CL"],
        "description": "Semantic parsing and knowledge representation",
        "block": "Semantic Understanding"
    },
    {
        "query": "language model knowledge",
        "categories": ["cs.CL", "cs.AI"],
        "description": "LLMs for knowledge graph construction",
        "block": "LLM-based KG"
    },
    {
        "query": "entity linking",
        "categories": ["cs.CL", "cs.DB"],
        "description": "Entity linking and KB population",
        "block": "Entity Linking"
    },
    {
        "query": "text mining",
        "categories": ["cs.DL", "cs.IR"],
        "description": "Mining scientific literature at scale",
        "block": "Literature Mining"
    },
    {
        "query": "biomedical text",
        "categories": ["cs.CL", "q-bio.QM"],
        "description": "Biomedical and biological knowledge graphs",
        "block": "Domain-Specific KG"
    }
]

# Configuration
ARTICLES_PER_TOPIC = 40  # 40 * 10 = 400 articles total
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


def print_topic_info(topic_num: int, topic: Dict[str, str]):
    """Print topic information"""
    print(f"\n{'─' * 80}")
    print(f"TOPIC {topic_num}/10: {topic['query']}")
    print(f"Block: {topic['block']}")
    print(f"Categories: {', '.join(topic['categories'])}")
    print(f"Description: {topic['description']}")
    print(f"{'─' * 80}")


def main():
    print_separator("Batch Download: arXiv Knowledge Graph Extraction Papers")

    # Load configuration
    config_path = Path(__file__).parent.parent / "src/config/fetcher_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    fetcher_config = config.get("arxiv", {})

    # Initialize fetcher and registry
    fetcher = ArXivFetcher(config=fetcher_config)
    registry = ArticleRegistry(registry_path=REGISTRY_PATH)

    print(f"Configuration:")
    print(f"  Total topics: {len(TOPICS)}")
    print(f"  Articles per topic: {ARTICLES_PER_TOPIC}")
    print(f"  Expected total: ~{len(TOPICS) * ARTICLES_PER_TOPIC} articles")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Registry: {REGISTRY_PATH}")
    print(f"\nCurrent registry status:")
    print(f"  Total articles: {registry.count()}")

    # Statistics
    total_searched = 0
    total_downloaded = 0
    total_skipped = 0
    total_failed = 0
    topic_stats = []

    # Process each topic
    for i, topic in enumerate(TOPICS, 1):
        print_topic_info(i, topic)

        topic_downloaded = 0
        topic_skipped = 0
        topic_failed = 0

        try:
            # Search for articles using simple query
            print(f"\n[1/3] Searching arXiv...")
            print(f"  Query: {topic['query']}")

            arxiv_ids = fetcher.search(
                query=topic["query"],
                max_results=ARTICLES_PER_TOPIC,
                sort_by="relevance"
            )

            total_searched += len(arxiv_ids)
            print(f"  ✓ Found {len(arxiv_ids)} articles")

            if not arxiv_ids:
                print(f"  ⚠ No articles found, skipping topic")
                continue

            # Fetch metadata and download PDFs
            print(f"\n[2/3] Fetching metadata and downloading PDFs...")

            for j, arxiv_id in enumerate(arxiv_ids, 1):
                # Check if already downloaded
                if registry.exists(arxiv_id):
                    topic_skipped += 1
                    if j % 10 == 0:
                        print(f"  Progress: {j}/{len(arxiv_ids)} (downloaded: {topic_downloaded}, skipped: {topic_skipped})")
                    continue

                try:
                    # Fetch metadata
                    paper = fetcher.fetch_paper(arxiv_id)

                    # Download PDF
                    pdf_path = fetcher.download_pdf(arxiv_id, output_dir=OUTPUT_DIR)

                    if pdf_path:
                        # Get file size
                        file_size = os.path.getsize(pdf_path)

                        # Add to registry with topic info
                        record = ArticleRecord(
                            pmid=None,  # arXiv doesn't have PMID
                            pmc_id=None,  # arXiv doesn't have PMC ID
                            arxiv_id=arxiv_id,
                            doi=paper.doi,
                            title=paper.title,
                            authors=paper.authors,
                            journal=paper.journal,
                            publication_date=paper.publication_date,
                            pdf_path=pdf_path,
                            file_size=file_size,
                            download_source=f"arXiv (Topic: {topic['query']})"
                        )

                        registry.add(record)
                        topic_downloaded += 1

                        if j % 10 == 0:
                            print(f"  Progress: {j}/{len(arxiv_ids)} (downloaded: {topic_downloaded}, skipped: {topic_skipped})")
                    else:
                        topic_failed += 1

                except Exception as e:
                    print(f"  ⚠ Error processing arXiv ID {arxiv_id}: {e}")
                    topic_failed += 1

            # Topic summary
            print(f"\n✓ Topic {i} completed:")
            print(f"  Searched: {len(arxiv_ids)}")
            print(f"  Downloaded: {topic_downloaded}")
            print(f"  Skipped (exists): {topic_skipped}")
            print(f"  Failed: {topic_failed}")

            # Update totals
            total_downloaded += topic_downloaded
            total_skipped += topic_skipped
            total_failed += topic_failed

            # Save topic stats
            topic_stats.append({
                "topic": topic["query"],
                "block": topic["block"],
                "categories": topic["categories"],
                "searched": len(arxiv_ids),
                "downloaded": topic_downloaded,
                "skipped": topic_skipped,
                "failed": topic_failed
            })

        except Exception as e:
            print(f"\n✗ Error processing topic: {e}")
            continue

        # Small delay between topics
        time.sleep(2)

    # Final summary
    print_separator("BATCH DOWNLOAD COMPLETE")

    print(f"Overall Statistics:")
    print(f"  Total articles searched: {total_searched}")
    print(f"  Total downloaded: {total_downloaded}")
    print(f"  Total skipped (already exists): {total_skipped}")
    print(f"  Total failed: {total_failed}")
    if total_searched > 0:
        print(f"  Success rate: {total_downloaded / total_searched * 100:.1f}%")

    # Registry statistics
    print_separator("Registry Statistics")
    stats = registry.get_stats()
    print(f"Total articles in registry: {stats['total_articles']}")
    print(f"Total size: {stats['total_size_mb']} MB")
    print(f"Articles with PMC ID: {stats['with_pmc_id']}")
    print(f"Articles with arXiv ID: {stats['with_arxiv_id']}")
    print(f"Articles with DOI: {stats['with_doi']}")

    # Topic breakdown
    print_separator("Downloads by Topic")
    for i, topic_stat in enumerate(topic_stats, 1):
        print(f"[{i}] {topic_stat['topic']}")
        print(f"    Block: {topic_stat['block']}")
        print(f"    Categories: {', '.join(topic_stat['categories'])}")
        print(f"    Searched: {topic_stat['searched']} | Downloaded: {topic_stat['downloaded']} | "
              f"Skipped: {topic_stat['skipped']} | Failed: {topic_stat['failed']}")
        print()

    print_separator()
    print(f"\n✓ All PDFs saved to: {OUTPUT_DIR}")
    print(f"✓ Registry saved to: {REGISTRY_PATH}")
    print(f"\nNext steps:")
    print(f"  1. Parse PDFs to extract text and structure")
    print(f"  2. Apply extraction pipelines (LLM/Regex/Hybrid)")
    print(f"  3. Build knowledge graphs from extracted entities")
    print(f"  4. Analyze KG extraction methodologies from downloaded papers")


if __name__ == "__main__":
    main()
