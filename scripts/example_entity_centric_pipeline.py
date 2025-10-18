"""
Example: Entity-Centric Hybrid Extraction Pipeline

Demonstrates the complete Entity-Centric extraction workflow:
1. Parse PDF with GROBID (ML-based structured extraction)
2. Generate sentence embeddings with Ollama (FREE)
3. Generate entity-specific keywords with OpenAI (~$0.003)
4. Semantic retrieval with ChromaDB (FREE)
5. LLM validation in batches (~$0.015)
6. Graph assembly with heuristics (FREE)

Total cost: ~$0.018 per paper
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parsers import GrobidParser
from src.pipelines import EntityCentricPipeline
from src.models import EntityType

# Load environment variables
load_dotenv()


def check_prerequisites():
    """Check if all required services are available"""
    issues = []

    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        issues.append("❌ OPENAI_API_KEY not found in .env")
    else:
        print("✅ OpenAI API key found")

    # Check Ollama availability
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"✅ Ollama service running ({len(models)} models available)")
        else:
            issues.append("❌ Ollama service not responding")
    except Exception as e:
        issues.append(f"❌ Ollama not available: {e}")
        issues.append("   Run: ollama pull bge-m3")

    # Check GROBID service
    try:
        import requests
        response = requests.get("https://lfoppiano-grobid.hf.space/api/isalive", timeout=2)
        if response.status_code == 200:
            print("✅ GROBID service running")
        else:
            issues.append("❌ GROBID service not responding")
    except Exception:
        issues.append("❌ GROBID not available")
        issues.append("   Run: docker run --rm -p 8070:8070 lfoppiano/grobid:0.8.0")

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
    print("Entity-Centric Hybrid Extraction Pipeline Demo")
    print("=" * 70)
    print()

    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)

    # Configuration
    pdf_path = project_root / "docs" / "sample_article.pdf"
    paper_id = "demo_paper_001"

    # Check if PDF exists
    if not Path(pdf_path).exists():
        print(f"❌ PDF not found: {pdf_path}")
        print("Please provide a valid PDF path")
        sys.exit(1)

    print(f"📄 Input: {pdf_path}")
    print(f"🆔 Paper ID: {paper_id}")
    print()

    # ========== STEP 1: Parse PDF with GROBID ==========

    print("=" * 70)
    print("STEP 1: Parsing PDF with GROBID (ML-based extraction)")
    print("=" * 70)
    print()

    parser = GrobidParser(
        grobid_server="https://lfoppiano-grobid.hf.space",
        timeout=60
    )

    print("⚙️  Sending PDF to GROBID service...")
    parsed_doc = parser.parse(pdf_path)

    print(f"✅ Parsing complete!")
    print(f"   Title: {parsed_doc.title[:80]}...")

    if parsed_doc.imrad_sections:
        print(f"   IMRAD Sections:")
        for section_name, section_text in parsed_doc.imrad_sections.items():
            print(f"      • {section_name}: {len(section_text)} chars")

    print()

    # ========== STEP 2: Initialize Pipeline ==========

    print("=" * 70)
    print("STEP 2: Initializing Entity-Centric Pipeline")
    print("=" * 70)
    print()

    pipeline = EntityCentricPipeline(
        llm_provider="openai",
        llm_model="gpt-5-mini",
        embedding_provider="ollama"  # FREE local embeddings
    )

    print(f"✅ Pipeline initialized")
    print(f"   Description: {pipeline.get_description()}")
    print(f"   Estimated cost: ${pipeline.get_estimated_cost():.6f} per paper")
    print()

    # ========== STEP 3: Extract Entities and Relationships ==========

    print("=" * 70)
    print("STEP 3: Extracting Knowledge Graph")
    print("=" * 70)
    print()

    result = pipeline.extract(parsed_doc, paper_id)

    # ========== STEP 4: Display Results ==========

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()

    # Entity counts by type
    print("📊 Entities Extracted:")
    total_entities = 0
    for entity_type in EntityType:
        count = len(result.entities.get(entity_type.value, []))
        if count > 0:
            total_entities += count
            print(f"   • {entity_type.value}: {count}")
    print(f"   TOTAL: {total_entities} entities")
    print()

    # Relationships
    print(f"🕸️  Relationships: {len(result.relationships)}")
    if result.relationships:
        rel_types = {}
        for rel in result.relationships:
            rel_types[rel.relationship_type.value] = rel_types.get(rel.relationship_type.value, 0) + 1
        for rel_type, count in rel_types.items():
            print(f"   • {rel_type}: {count}")
    print()

    # ========== STEP 5: Display Detailed Metrics ==========

    print("=" * 70)
    print("DETAILED METRICS BY PHASE")
    print("=" * 70)
    print()

    metrics = result.metrics
    phases = metrics.metadata.get("phases", {})

    # Phase 0.5: Embeddings
    emb = phases.get("embeddings", {})
    print(f"📝 Phase 0.5: Sentence Embeddings (Ollama - FREE)")
    print(f"   Sentences: {emb.get('sentences', 0)}")
    print(f"   Tokens: {emb.get('tokens', 0)}")
    print(f"   Cost: ${emb.get('cost_usd', 0):.6f}")
    print()

    # Phase 1: Keywords
    kg = phases.get("keyword_generation", {})
    print(f"🔑 Phase 1: Keyword Generation (OpenAI)")
    print(f"   Entity types: {kg.get('entity_types', 0)}")
    print(f"   Tokens: {kg.get('tokens', 0)}")
    print(f"   Cost: ${kg.get('cost_usd', 0):.6f}")
    print(f"   Cache hit rate: {kg.get('cache_hit_rate', 0):.1%}")
    print()

    # Phase 4: Retrieval
    ret = phases.get("semantic_retrieval", {})
    print(f"🔍 Phase 4: Semantic Retrieval (ChromaDB - FREE)")
    print(f"   Queries: {ret.get('queries', 0)}")
    print(f"   Candidates retrieved: {ret.get('candidates', 0)}")
    print(f"   Collection size: {ret.get('collection_size', 0)}")
    print()

    # Phase 5: Validation
    val = phases.get("validation", {})
    print(f"✅ Phase 5: LLM Validation (OpenAI)")
    print(f"   Candidates: {val.get('candidates', 0)}")
    print(f"   Validated: {val.get('validated', 0)}")
    print(f"   Rejected: {val.get('rejected', 0)}")
    print(f"   Tokens: {val.get('tokens', 0)}")
    print(f"   Cost: ${val.get('cost_usd', 0):.6f}")
    print()

    # Phase 6: Graph
    graph = phases.get("graph_assembly", {})
    print(f"🕸️  Phase 6: Graph Assembly (Heuristics - FREE)")
    print(f"   Relationships: {graph.get('relationships', 0)}")
    if graph.get('by_type'):
        for rel_type, count in graph['by_type'].items():
            print(f"      • {rel_type}: {count}")
    print()

    # Overall metrics
    print("=" * 70)
    print("OVERALL METRICS")
    print("=" * 70)
    print()
    print(f"⏱️  Processing time: {metrics.processing_time:.2f}s")
    print(f"🪙  Total tokens: {metrics.tokens_used:,}")
    print(f"💰 Total cost: ${metrics.cost_usd:.6f}")
    print()

    # ========== STEP 6: Sample Entities ==========

    print("=" * 70)
    print("SAMPLE ENTITIES")
    print("=" * 70)
    print()

    for entity_type in EntityType:
        entities = result.entities.get(entity_type.value, [])
        if entities:
            print(f"\n{entity_type.value} (showing first 2):")
            for entity in entities[:2]:
                print(f"   • {entity.text[:100]}...")
                print(f"     Confidence: {entity.confidence:.2f}")
                print(f"     Section: {entity.source_section}")

    print()

    # ========== STEP 7: Save Results ==========

    print("=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    print()

    # Create output directory
    output_dir = Path(project_root / "results" )
    output_dir.mkdir(exist_ok=True)

    # Save full result as JSON
    output_path = output_dir / f"{paper_id}_entity_centric.json"

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
                "relationship_type": r.relationship_type.value,
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

    with open(output_path, 'w') as f:
        json.dump(result_dict, f, indent=2)

    print(f"✅ Results saved to: {output_path}")

    # Generate SVG visualization
    try:
        from src.visualization.generate_svg import generate_svg_from_json

        svg_path = output_dir / f"{paper_id}_graph.svg"
        generate_svg_from_json(str(output_path), str(svg_path))
        print(f"✅ Visualization saved to: {svg_path}")
    except Exception as e:
        print(f"⚠️  Could not generate visualization: {e}")

    print()
    print("=" * 70)
    print("✨ PIPELINE COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
