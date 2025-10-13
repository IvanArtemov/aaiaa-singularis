"""
Batch Download: 500 Cross-Referenced Articles from PubMed

This script downloads ~500 articles across 10 interconnected topics in aging research.
Topics are chosen to maximize cross-references for knowledge graph construction.

Usage:
    python scripts/batch_download_cross_referenced.py
"""

import sys
import os
from pathlib import Path
from typing import List, Dict
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fetchers.pubmed_fetcher import PubMedFetcher
from src.utils.article_registry import ArticleRegistry, ArticleRecord
import yaml


# 10 Interconnected Topics with High Cross-Reference Potential
# TOPICS = [
#     {
#         "query": "cellular senescence aging",
#         "description": "Cellular senescence and aging (foundational topic)",
#         "block": "Cellular Aging & Senolytics"
#     },
#     {
#         "query": "senolytics dasatinib quercetin",
#         "description": "Specific senolytics (references topic 1)",
#         "block": "Cellular Aging & Senolytics"
#     },
#     {
#         "query": "SASP senescence-associated secretory phenotype",
#         "description": "SASP phenotype (linked to topics 1, 2)",
#         "block": "Cellular Aging & Senolytics"
#     },
#     {
#         "query": "mTOR pathway aging",
#         "description": "mTOR signaling pathway",
#         "block": "Metabolism & Signaling"
#     },
#     {
#         "query": "rapamycin longevity lifespan",
#         "description": "Rapamycin as mTOR inhibitor (references topic 4)",
#         "block": "Metabolism & Signaling"
#     },
#     {
#         "query": "caloric restriction mTOR autophagy",
#         "description": "CR links to mTOR and autophagy (references topics 4, 5)",
#         "block": "Metabolism & Signaling"
#     },
#     {
#         "query": "mitochondrial dysfunction aging",
#         "description": "Mitochondrial dysfunction in aging",
#         "block": "Mitochondria & Metabolism"
#     },
#     {
#         "query": "NAD+ sirtuins mitochondria",
#         "description": "NAD+ and sirtuins (linked to topic 7)",
#         "block": "Mitochondria & Metabolism"
#     },
#     {
#         "query": "oxidative stress reactive oxygen species aging",
#         "description": "Oxidative stress and ROS (linked to topics 7, 8)",
#         "block": "Mitochondria & Metabolism"
#     },
#     {
#         "query": "epigenetic regulation aging reprogramming",
#         "description": "Epigenetics (connects all blocks)",
#         "block": "Epigenetics (Bridging Topic)"
#     }
# ]

# New 10 Interconnected Topics - Alternative Set
TOPICS = [
    {
        "query": "protein aggregation neurodegenerative diseases aging",
        "description": "Protein aggregation in neurodegeneration and aging",
        "block": "Proteostasis & Protein Aggregation"
    },
    # {
    #     "query": "autophagy lysosomal degradation aging",
    #     "description": "Autophagy and lysosomal degradation (clears aggregates from topic 1)",
    #     "block": "Proteostasis & Protein Aggregation"
    # },
    # {
    #     "query": "ubiquitin proteasome system aging",
    #     "description": "Ubiquitin-proteasome system (alternative clearance to topic 2)",
    #     "block": "Proteostasis & Protein Aggregation"
    # },
    # {
    #     "query": "DNA damage repair aging",
    #     "description": "DNA damage and repair systems",
    #     "block": "DNA Damage & Repair"
    # },
    # {
    #     "query": "telomere attrition telomerase aging",
    #     "description": "Telomere shortening and telomerase (linked to DNA damage)",
    #     "block": "DNA Damage & Repair"
    # },
    # {
    #     "query": "genome instability somatic mutations aging",
    #     "description": "Genome instability and somatic mutations (consequence of topics 4, 5)",
    #     "block": "DNA Damage & Repair"
    # },
    # {
    #     "query": "stem cell exhaustion aging",
    #     "description": "Stem cell exhaustion with age",
    #     "block": "Stem Cells & Regeneration"
    # },
    # {
    #     "query": "tissue regeneration aging inflammation",
    #     "description": "Tissue regeneration and inflammation (requires stem cells)",
    #     "block": "Stem Cells & Regeneration"
    # },
    # {
    #     "query": "hematopoietic stem cells aging",
    #     "description": "HSC aging as stem cell model (specific case of topic 7)",
    #     "block": "Stem Cells & Regeneration"
    # },
    # {
    #     "query": "intercellular communication aging extracellular vesicles",
    #     "description": "Intercellular communication via exosomes (bridges all blocks)",
    #     "block": "Intercellular Communication (Bridging)"
    # }
]

# Configuration
ARTICLES_PER_TOPIC = 50
OUTPUT_DIR = "articles/packages"
REGISTRY_PATH = "articles/metadata.json"
EXCLUDE_REVIEWS = True  # Exclude Review articles from search
EXCLUDE_META_ANALYSIS = True  # Exclude Meta-Analysis articles


def print_separator(title: str = "", char: str = "="):
    """Print formatted separator"""
    if title:
        print(f"\n{char * 80}")
        print(f"  {title}")
        print(f"{char * 80}\n")
    else:
        print(f"{char * 80}\n")


def build_search_query(base_query: str, free_full_text: bool = True,
                       exclude_reviews: bool = True, exclude_meta_analysis: bool = False) -> str:
    """
    Build PubMed search query with filters

    Args:
        base_query: Base search query
        free_full_text: Add free full text filter
        exclude_reviews: Exclude Review articles
        exclude_meta_analysis: Exclude Meta-Analysis articles

    Returns:
        Complete search query with filters
    """
    query = base_query

    # Add free full text filter
    if free_full_text:
        query += " AND free full text[filter]"

    # Exclude Review articles
    if exclude_reviews:
        query += " NOT Review[Publication Type]"

    # Exclude Meta-Analysis articles
    if exclude_meta_analysis:
        query += " NOT Meta-Analysis[Publication Type]"

    return query


def print_topic_info(topic_num: int, topic: Dict[str, str]):
    """Print topic information"""
    print(f"\n{'─' * 80}")
    print(f"TOPIC {topic_num}/10: {topic['query']}")
    print(f"Block: {topic['block']}")
    print(f"Description: {topic['description']}")
    print(f"{'─' * 80}")


def main():
    print_separator("Batch Download: 500 Cross-Referenced Articles")

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

    print(f"Configuration:")
    print(f"  Total topics: {len(TOPICS)}")
    print(f"  Articles per topic: {ARTICLES_PER_TOPIC}")
    print(f"  Expected total: ~{len(TOPICS) * ARTICLES_PER_TOPIC} articles")
    print(f"  Free full text filter: ENABLED")
    print(f"  Exclude Reviews: {'ENABLED' if EXCLUDE_REVIEWS else 'DISABLED'}")
    print(f"  Exclude Meta-Analysis: {'ENABLED' if EXCLUDE_META_ANALYSIS else 'DISABLED'}")
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
            # Build search query with filters
            search_query = build_search_query(
                topic["query"],
                free_full_text=True,
                exclude_reviews=EXCLUDE_REVIEWS,
                exclude_meta_analysis=EXCLUDE_META_ANALYSIS
            )

            # Search for articles
            print(f"\n[1/3] Searching for articles...")
            print(f"  Query: {search_query}")
            pmids = fetcher.search(
                search_query,
                max_results=ARTICLES_PER_TOPIC,
                free_full_text=False  # Already included in query
            )
            total_searched += len(pmids)
            print(f"  ✓ Found {len(pmids)} articles")

            if not pmids:
                print(f"  ⚠ No articles found, skipping topic")
                continue

            # Fetch metadata
            print(f"\n[2/3] Fetching metadata...")
            papers = []
            for j, pmid in enumerate(pmids, 1):
                try:
                    paper = fetcher.fetch_paper(pmid)
                    papers.append(paper)
                    if j % 10 == 0:
                        print(f"  Progress: {j}/{len(pmids)} papers fetched")
                except Exception as e:
                    print(f"  ⚠ Error fetching PMID {pmid}: {e}")
                    topic_failed += 1

            print(f"  ✓ Fetched metadata for {len(papers)} papers")

            # Download packages
            print(f"\n[3/3] Downloading tar.gz packages...")
            for j, paper in enumerate(papers, 1):
                pmid = paper.pmid

                # Check if already downloaded
                if registry.exists(pmid):
                    topic_skipped += 1
                    if j % 10 == 0:
                        print(f"  Progress: {j}/{len(papers)} (skipped: {topic_skipped})")
                    continue

                # Check if has PMC ID
                if not paper.pmc_id:
                    topic_failed += 1
                    continue

                # Download tar.gz package
                try:
                    paper_id = paper.pmc_id
                    package_path = fetcher.download_pdf(paper_id, output_dir=OUTPUT_DIR)

                    if package_path:
                        # Get file size
                        file_size = os.path.getsize(package_path)

                        # Add to registry with topic info
                        record = ArticleRecord(
                            pmid=paper.pmid,
                            pmc_id=paper.pmc_id,
                            doi=paper.doi,
                            title=paper.title,
                            authors=paper.authors,
                            journal=paper.journal,
                            publication_date=paper.publication_date,
                            pdf_path=package_path,
                            file_size=file_size,
                            download_source=f"PMC_OA (Topic: {topic['query']})"
                        )

                        registry.add(record)
                        topic_downloaded += 1

                        if j % 10 == 0:
                            print(f"  Progress: {j}/{len(papers)} (downloaded: {topic_downloaded}, skipped: {topic_skipped})")
                    else:
                        topic_failed += 1
                except Exception as e:
                    print(f"  ⚠ Download failed for PMID {pmid}: {e}")
                    topic_failed += 1

            # Topic summary
            print(f"\n✓ Topic {i} completed:")
            print(f"  Searched: {len(pmids)}")
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
                "searched": len(pmids),
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
    print(f"  Success rate: {total_downloaded / total_searched * 100:.1f}%")

    # Registry statistics
    print_separator("Registry Statistics")
    stats = registry.get_stats()
    print(f"Total articles in registry: {stats['total_articles']}")
    print(f"Total size: {stats['total_size_mb']} MB")
    print(f"Articles with PMC ID: {stats['with_pmc_id']}")
    print(f"Articles with DOI: {stats['with_doi']}")

    # Topic breakdown
    print_separator("Downloads by Topic")
    for i, topic_stat in enumerate(topic_stats, 1):
        print(f"[{i}] {topic_stat['topic']}")
        print(f"    Block: {topic_stat['block']}")
        print(f"    Searched: {topic_stat['searched']} | Downloaded: {topic_stat['downloaded']} | "
              f"Skipped: {topic_stat['skipped']} | Failed: {topic_stat['failed']}")
        print()

    print_separator()
    print(f"\n✓ All packages saved to: {OUTPUT_DIR}")
    print(f"✓ Registry saved to: {REGISTRY_PATH}")
    print(f"\nNext steps:")
    print(f"  1. Extract tar.gz packages to get PDFs")
    print(f"  2. Parse PDFs to extract structured information")
    print(f"  3. Build knowledge graph with cross-references")


if __name__ == "__main__":
    main()
