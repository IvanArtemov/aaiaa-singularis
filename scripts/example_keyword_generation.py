"""
Example script for LLM-based keyword generation

This script demonstrates the EntityKeywordGenerator that generates
context-specific keywords for entity extraction based on paper content.
"""

import sys
import os
from pathlib import Path

# Add project root to path
def main():
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from src.extractors import EntityKeywordGenerator
    from src.parsers import GrobidParser
    from src.models import EntityType

    print("=" * 80)
    print("LLM-Based Keyword Generation Example")
    print("=" * 80)

    pdf_path = project_root / "docs" / "sample_article.pdf"
    print(f"\nTesting with: {pdf_path.name}")

    # Parse document
    print("\n" + "=" * 80)
    print("STEP 1: Parsing Document")
    print("=" * 80)

    parser = GrobidParser()
    parsed_doc = parser.parse(str(pdf_path))

    print(f"\nDocument parsed:")
    print(f"  Title: {parsed_doc.title or 'Unknown'}")
    print(f"  Word count: {parsed_doc.word_count:,}")
    print(f"  Sections: {list(parsed_doc.imrad_sections.keys()) if parsed_doc.imrad_sections else []}")

    # Show context that will be sent to LLM
    print("\n" + "=" * 80)
    print("STEP 2: Context for Keyword Generation")
    print("=" * 80)

    title = parsed_doc.title or "Unknown"
    abstract = parsed_doc.get_section("abstract") or ""
    intro = parsed_doc.get_section("introduction") or ""

    print(f"\nTitle: {title[:100]}...")
    print(f"\nAbstract: {abstract[:200]}...")
    print(f"\nIntroduction (first 200 chars): {intro[:200]}...")

    # Initialize keyword generator
    print("\n" + "=" * 80)
    print("STEP 3: Generate Keywords for Each Entity Type")
    print("=" * 80)

    generator = EntityKeywordGenerator(
        llm_provider="openai",
        llm_model="gpt-5-mini"
    )

    # Test individual entity types
    entity_types = [
        EntityType.HYPOTHESIS,
        # EntityType.TECHNIQUE,
        # EntityType.RESULT,
        # EntityType.DATASET
    ]

    for entity_type in entity_types:
        print(f"\n--- {entity_type.value.upper()} ---")
        keywords = generator.generate_keywords(parsed_doc, entity_type)

        if keywords:
            print(f"Generated {len(keywords)} keywords:")
            for i, kw in enumerate(keywords, 1):
                print(f"  {i}. {kw}")
        else:
            print("  No keywords generated (error or empty response)")

    # Show metrics
    print("\n" + "=" * 80)
    print("STEP 4: Keyword Generation Metrics")
    print("=" * 80)

    metrics = generator.get_metrics()
    print(f"\nTotal tokens used: {metrics['total_tokens']}")
    print(f"Total cost: ${metrics['total_cost_usd']:.6f}")
    print(f"Cache hits: {metrics['cache_hits']}")
    print(f"Cache misses: {metrics['cache_misses']}")
    print(f"Cache hit rate: {metrics['cache_hit_rate']:.1%}")

    # Test generate_all_keywords
    print("\n" + "=" * 80)
    print("STEP 5: Generate All Keywords at Once")
    print("=" * 80)

    # Reset generator for clean test
    generator_2 = EntityKeywordGenerator()
    # all_keywords = generator_2.generate_all_keywords(parsed_doc)

    # print(f"\nGenerated keywords for {len(all_keywords)} entity types:")
    # for entity_type, keywords in all_keywords.items():
    #     print(f"\n{entity_type.value.upper()}: {len(keywords)} keywords")
    #     # Show first 5
    #     for kw in keywords[:5]:
    #         print(f"  - {kw}")
    #     if len(keywords) > 5:
    #         print(f"  ... and {len(keywords) - 5} more")

    # metrics_2 = generator_2.get_metrics()
    # print(f"\n\nFinal metrics:")
    # print(f"  Tokens: {metrics_2['total_tokens']}")
    # print(f"  Cost: ${metrics_2['total_cost_usd']:.6f}")
    # print(f"  Average cost per entity type: ${metrics_2['total_cost_usd'] / max(1, len(all_keywords)):.6f}")

    # Estimate cost for 50M papers
    print("\n" + "=" * 80)
    print("COST PROJECTION")
    print("=" * 80)

    cost_per_paper = metrics_2['total_cost_usd']
    print(f"\nCost per paper: ${cost_per_paper:.6f}")
    print(f"Cost per 1,000 papers: ${cost_per_paper * 1000:.2f}")
    print(f"Cost per 1M papers: ${cost_per_paper * 1_000_000:,.2f}")
    print(f"Cost for 50M papers: ${cost_per_paper * 50_000_000:,.2f}")

    if cost_per_paper < 0.005:
        print("\n✅ Target achieved! Cost is < $0.005/paper for keyword generation")
    else:
        print(f"\n⚠️  Cost is ${cost_per_paper:.6f}/paper (target: < $0.005)")

    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()