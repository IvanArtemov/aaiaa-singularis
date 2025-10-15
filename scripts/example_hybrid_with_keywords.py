"""
Example script for Hybrid Pipeline with LLM-based keyword generation

This script demonstrates the enhanced Hybrid Pipeline that uses
LLM-generated keywords for context-specific entity extraction.
"""

import sys
import os
import json
from pathlib import Path

def main():
    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from src.pipelines import HybridPipeline
    from src.parsers import GrobidParser
    from src.config.settings import settings

    print("=" * 80)
    print("Hybrid Pipeline with LLM Keyword Generation Example")
    print("=" * 80)

    pdf_path = project_root / "docs" / "sample_article.pdf"
    print(f"\nTesting with: {pdf_path.name}")

    # Parse document
    print("\n" + "=" * 80)
    print("STEP 1: Parsing Document with GROBID")
    print("=" * 80)

    parser = GrobidParser()
    parsed_doc = parser.parse(str(pdf_path))

    print(f"\nDocument parsed:")
    print(f"  Title: {parsed_doc.title or 'Unknown'}")
    print(f"  Word count: {parsed_doc.word_count:,}")
    print(f"  Parse time: {parsed_doc.parse_time:.2f}s")
    print(f"  Sections: {list(parsed_doc.imrad_sections.keys()) if parsed_doc.imrad_sections else []}")

    # Initialize pipeline
    print("\n" + "=" * 80)
    print("STEP 2: Initialize Hybrid Pipeline")
    print("=" * 80)

    config = {
        "pattern_confidence_threshold": 0.7,
        "nlp_confidence_threshold": 0.6
    }

    pipeline = HybridPipeline(
        config=config,
        llm_provider="openai",
        llm_model="gpt-5-mini"
    )

    print(f"\nPipeline: {pipeline.get_description()}")

    # Extract entities
    print("\n" + "=" * 80)
    print("STEP 3: Extract Entities with Keyword Generation")
    print("=" * 80)

    paper_id = pdf_path.stem
    result = pipeline.extract(parsed_doc, paper_id)

    # Show results
    print("\n" + "=" * 80)
    print("EXTRACTION RESULTS")
    print("=" * 80)

    print(f"\n📊 Summary:")
    print(f"  Total entities: {result.metrics.entities_extracted}")
    print(f"  Total relationships: {result.metrics.relationships_extracted}")

    print(f"\n📈 Entities by type:")
    for entity_type, entities in result.entities.items():
        print(f"  {entity_type}: {len(entities)}")

    # Show sample entities
    print("\n📝 Sample entities (first 2 per type):")
    for entity_type, entities in result.entities.items():
        print(f"\n  {entity_type.upper()}:")
        for entity in entities[:2]:
            text_preview = entity.text[:80] + "..." if len(entity.text) > 80 else entity.text
            used_keywords = entity.metadata.get("used_dynamic_keywords", False)
            keyword_marker = " [🔑 dynamic]" if used_keywords else ""
            print(f"    - {text_preview}")
            print(f"      (confidence: {entity.confidence:.2f}, section: {entity.source_section}){keyword_marker}")
        if len(entities) > 2:
            print(f"    ... and {len(entities) - 2} more")

    # Show metrics
    print("\n" + "=" * 80)
    print("PERFORMANCE METRICS")
    print("=" * 80)

    metrics = result.metrics
    print(f"\n⏱️  Processing time: {metrics.processing_time:.2f}s")
    print(f"💰 Cost: ${metrics.cost_usd:.6f}")
    print(f"🔤 Tokens used: {metrics.tokens_used}")

    # Show keyword generation metrics
    if "keyword_generation" in metrics.metadata:
        kg_metrics = metrics.metadata["keyword_generation"]
        print(f"\n🔑 Keyword Generation:")
        print(f"  Tokens: {kg_metrics['tokens_used']}")
        print(f"  Cost: ${kg_metrics['cost_usd']:.6f}")
        print(f"  Entity types covered: {kg_metrics['entity_types_covered']}")
        print(f"  Cache hit rate: {kg_metrics['cache_hit_rate']:.1%}")

        # Show breakdown
        total_cost = metrics.cost_usd
        kg_cost = kg_metrics['cost_usd']
        other_cost = total_cost - kg_cost

        print(f"\n💸 Cost breakdown:")
        print(f"  Keyword generation: ${kg_cost:.6f} ({kg_cost/total_cost*100:.1f}%)")
        print(f"  Other operations: ${other_cost:.6f} ({other_cost/total_cost*100:.1f}%)")

    # Show extraction methods
    if "extraction_methods" in metrics.metadata:
        print(f"\n🔧 Extraction methods used:")
        for method, count in metrics.metadata["extraction_methods"].items():
            print(f"  {method}: {count}")

    # Count entities found with dynamic keywords
    dynamic_keyword_count = 0
    for entities in result.entities.values():
        for entity in entities:
            if entity.metadata.get("used_dynamic_keywords", False):
                dynamic_keyword_count += 1

    print(f"\n🎯 Entities found using dynamic keywords: {dynamic_keyword_count} / {metrics.entities_extracted}")
    if metrics.entities_extracted > 0:
        print(f"   ({dynamic_keyword_count / metrics.entities_extracted * 100:.1f}%)")

    # Cost projection
    print("\n" + "=" * 80)
    print("COST PROJECTION FOR 50M PAPERS")
    print("=" * 80)

    cost_per_paper = metrics.cost_usd
    print(f"\nCost per paper: ${cost_per_paper:.6f}")
    print(f"Cost per 1,000 papers: ${cost_per_paper * 1000:.2f}")
    print(f"Cost per 1M papers: ${cost_per_paper * 1_000_000:,.2f}")
    print(f"Cost for 50M papers: ${cost_per_paper * 50_000_000:,.2f}")

    if cost_per_paper < 0.05:
        print(f"\n✅ Target achieved! Cost is < $0.05/paper")
    else:
        print(f"\n⚠️  Cost is ${cost_per_paper:.6f}/paper (target: < $0.05)")

    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    output_dir = project_root / "results"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"{paper_id}_hybrid_keywords.json"
    result_dict = result.to_dict()

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024:.1f} KB")

    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()