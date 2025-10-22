"""
Example: SciBERT-Nebius Hybrid Extraction Pipeline

Demonstrates the SciBERT-Nebius extraction workflow:
1. Parse GROBID TEI XML (pre-parsed scientific article)
2. Generate sentence embeddings with SciBERT (FREE, 768 dims)
3. Generate entity-specific keywords with Nebius (~$0.003)
4. Semantic retrieval with ChromaDB (FREE)
5. LLM validation in batches with Nebius (~$0.015)
6. Graph assembly with heuristics (FREE)

Total cost: ~$0.018 per paper
Embeddings: SciBERT (FREE, domain-optimized for scientific papers)
LLM: Nebius gpt-oss-120b (cost-efficient)

Note: This script reads pre-parsed GROBID XML instead of calling GROBID service
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parsers import GrobidParser
from src.pipelines import SciBertNebiusPipeline
from src.models import EntityType

# Load environment variables
load_dotenv()


def check_prerequisites():
    """Check if all required services are available"""
    issues = []

    # Check Nebius API key
    if not os.getenv("NEBIUS_API_KEY"):
        issues.append("❌ NEBIUS_API_KEY not found in .env")
        issues.append("   Get your API key from: https://studio.nebius.com/")
    else:
        print("✅ Nebius API key found")

    # Check if transformers/torch are installed (for SciBERT)
    try:
        import transformers
        import torch
        print("✅ SciBERT dependencies installed (transformers, torch)")
    except ImportError as e:
        issues.append(f"❌ SciBERT dependencies missing: {e}")
        issues.append("   Run: pip install transformers torch scikit-learn")

    if issues:
        print("\n⚠️  Prerequisites not met:")
        for issue in issues:
            print(issue)
        print("\nPlease resolve these issues before running the pipeline.")
        return False

    print("\n✅ All prerequisites met!\n")
    return True


def main():
    project_root = Path(__file__).parent.parent
    print("=" * 70)
    print("SciBERT-Nebius Pipeline Demo")
    print("=" * 70)

    # Check prerequisites
    if not check_prerequisites():
        return

    # ========== Step 1: Setup XML path ==========
    xml_path = project_root / "docs" / "sample_article.xml"

    if not xml_path.exists():
        print(f"\n❌ Error: Sample XML not found at {xml_path}")
        print("Please place a GROBID TEI XML file at docs/sample_article.xml")
        return

    print(f"\n📄 XML: {xml_path.name}")

    # ========== Step 2: Parse GROBID XML ==========
    print("\n" + "=" * 70)
    print("STEP 1: Parsing GROBID TEI XML")
    print("=" * 70)

    try:
        parser = GrobidParser()
        print("\nParsing XML...")
        parsed_doc = parser.parse_from_xml(str(xml_path))
    except Exception as e:
        print(f"\n❌ Error parsing XML: {e}")
        return

    print(f"\n✅ Document parsed successfully:")
    print(f"  Title: {parsed_doc.title or 'Unknown'}")
    print(f"  Word count: {parsed_doc.word_count:,}")
    print(f"  Parse time: {parsed_doc.parse_time:.2f}s")

    if parsed_doc.imrad_sections:
        print(f"  Sections: {list(parsed_doc.imrad_sections.keys())}")
    else:
        print("  ⚠️  Warning: No IMRAD sections found")
        return

    # ========== Step 3: Run SciBERT-Nebius Pipeline ==========
    print("\n" + "=" * 70)
    print("STEP 2: Running SciBERT-Nebius Pipeline")
    print("=" * 70)

    try:
        pipeline = SciBertNebiusPipeline()
        print(f"\nPipeline: {pipeline.get_description()}")
        print(f"Estimated cost: ${pipeline.get_estimated_cost():.4f} per paper")

        print("\n🚀 Starting extraction...\n")

        result = pipeline.extract(
            parsed_doc=parsed_doc,
            paper_id=xml_path.stem
        )

    except Exception as e:
        print(f"\n❌ Error running pipeline: {e}")
        import traceback
        traceback.print_exc()
        return

    # ========== Step 4: Display Results ==========
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    # Entity counts by type
    print("\n📊 Entities extracted:")

    # Collect all entities into a list
    all_entities_list = []
    for entity_type in EntityType:
        entities = result.entities.get(entity_type.value, [])
        all_entities_list.extend(entities)
        count = len(entities)
        if count > 0:
            print(f"  {entity_type.value.upper()}: {count}")

    total_entities = len(all_entities_list)
    print(f"\n  TOTAL: {total_entities} entities")
    print(f"  Relationships: {len(result.relationships)}")

    # Sample entities
    print("\n📝 Sample entities (first 3 per type):")
    for entity_type in EntityType:
        entities = result.entities.get(entity_type.value, [])
        if entities:
            print(f"\n  {entity_type.value.upper()}:")
            for i, entity in enumerate(entities[:3], 1):
                text_preview = entity.text[:70] + "..." if len(entity.text) > 70 else entity.text
                print(f"    {i}. {text_preview}")
                print(f"       Confidence: {entity.confidence:.2f}, Section: {entity.source_section}")

    # Relationships
    if result.relationships:
        print(f"\n🔗 Sample relationships (first 5):")
        for i, rel in enumerate(result.relationships[:5], 1):
            source = next((e for e in all_entities_list if e.id == rel.source_id), None)
            target = next((e for e in all_entities_list if e.id == rel.target_id), None)

            if source and target:
                source_text = source.text[:40] + "..." if len(source.text) > 40 else source.text
                target_text = target.text[:40] + "..." if len(target.text) > 40 else target.text

                print(f"\n  {i}. {rel.relationship_type.value}")
                print(f"     {source_text}")
                print(f"     → {target_text}")
                print(f"     Confidence: {rel.confidence:.2f}")

    # ========== Step 5: Display Metrics ==========
    print("\n" + "=" * 70)
    print("PERFORMANCE METRICS")
    print("=" * 70)

    metrics = result.metrics

    print(f"\nProcessing time: {metrics.processing_time:.2f}s")
    print(f"Tokens used: {metrics.tokens_used:,}")
    print(f"Cost: ${metrics.cost_usd:.6f}")

    # Phase breakdown
    if "phases" in metrics.metadata:
        phases = metrics.metadata["phases"]

        print("\n📈 Phase breakdown:")

        # Embeddings
        if "embeddings" in phases:
            emb = phases["embeddings"]
            print(f"\n  Phase 0.5 - SciBERT Embeddings:")
            print(f"    Sentences: {emb['sentences']}")
            print(f"    Cost: ${emb['cost_usd']:.6f} (FREE)")

        # Keywords
        if "keyword_generation" in phases:
            kg = phases["keyword_generation"]
            print(f"\n  Phase 1 - Keyword Generation (Nebius):")
            print(f"    Entity types: {kg['entity_types']}")
            print(f"    Tokens: {kg['tokens']:,}")
            print(f"    Cost: ${kg['cost_usd']:.6f}")

        # Retrieval
        if "semantic_retrieval" in phases:
            ret = phases["semantic_retrieval"]
            print(f"\n  Phase 4 - Semantic Retrieval:")
            print(f"    Queries: {ret['queries']}")
            print(f"    Candidates: {ret['candidates']}")
            print(f"    Cost: $0.000000 (FREE)")

        # Validation
        if "validation" in phases:
            val = phases["validation"]
            print(f"\n  Phase 5 - LLM Validation (Nebius):")
            print(f"    Candidates: {val['candidates']}")
            print(f"    Validated: {val['validated']}")
            print(f"    Rejected: {val['rejected']}")
            print(f"    Tokens: {val['tokens']:,}")
            print(f"    Cost: ${val['cost_usd']:.6f}")

        # Graph
        if "graph_assembly" in phases:
            graph = phases["graph_assembly"]
            print(f"\n  Phase 6 - Graph Assembly:")
            print(f"    Relationships: {graph['relationships']}")
            print(f"    Cost: $0.000000 (FREE)")

    # ========== Step 6: Cost Projection ==========
    print("\n" + "=" * 70)
    print("COST PROJECTION")
    print("=" * 70)

    cost_per_paper = metrics.cost_usd

    print(f"\nCost per paper: ${cost_per_paper:.6f}")
    print(f"Cost per 1,000 papers: ${cost_per_paper * 1000:.2f}")
    print(f"Cost per 1M papers: ${cost_per_paper * 1_000_000:,.2f}")
    print(f"Cost for 50M papers: ${cost_per_paper * 50_000_000:,.2f}")

    if cost_per_paper < 0.02:
        print(f"\n✅ Excellent! Cost is under $0.02/paper target")
    elif cost_per_paper < 0.05:
        print(f"\n✅ Good! Cost is under $0.05/paper target")
    else:
        print(f"\n⚠️  Cost is above target: ${cost_per_paper:.6f}/paper")

    # ========== Step 7: Save Results ==========
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    output_file = results_dir / f"{xml_path.stem}_scibert_nebius.json"

    # Convert result to dict
    result_dict = {
        "paper_id": result.paper_id,
        "entities": {
            entity_type: [
                {
                    "id": e.id,
                    "type": e.type.value,
                    "text": e.text,
                    "confidence": e.confidence,
                    "source_section": e.source_section,
                    "metadata": e.metadata
                }
                for e in entities
            ]
            for entity_type, entities in result.entities.items()
        },
        "relationships": [
            {
                "source_id": r.source_id,
                "target_id": r.target_id,
                "type": r.relationship_type.value,
                "confidence": r.confidence,
                "metadata": r.metadata
            }
            for r in result.relationships
        ],
        "metrics": {
            "processing_time": metrics.processing_time,
            "tokens_used": metrics.tokens_used,
            "cost_usd": metrics.cost_usd,
            "entities_extracted": metrics.entities_extracted,
            "relationships_extracted": metrics.relationships_extracted,
            "metadata": metrics.metadata
        }
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Results saved to: {output_file}")

    # ========== Summary ==========
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n✨ Pipeline completed successfully!")
    print(f"\n📊 Results:")
    print(f"  Entities: {total_entities}")
    print(f"  Relationships: {len(result.relationships)}")
    print(f"  Processing time: {metrics.processing_time:.2f}s")
    print(f"  Cost: ${metrics.cost_usd:.6f}")

    print(f"\n💡 Pipeline features:")
    print(f"  ✅ SciBERT embeddings (FREE, domain-optimized)")
    print(f"  ✅ Nebius gpt-oss-120b (cost-efficient LLM)")
    print(f"  ✅ Semantic retrieval with ChromaDB")
    print(f"  ✅ Batch LLM validation")
    print(f"  ✅ Automatic graph assembly")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
