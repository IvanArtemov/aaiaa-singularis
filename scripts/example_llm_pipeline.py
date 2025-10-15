"""Test LLM Pipeline on COPD article"""

import os
from dotenv import load_dotenv

from src.parsers import PDFParser
from src.pipelines import LLMPipeline
from src.models import KnowledgeGraph

# Load environment variables
load_dotenv()

def main():
    print("=" * 80)
    print("LLM PIPELINE TEST - COPD Article Extraction")
    print("=" * 80)

    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ ERROR: OPENAI_API_KEY not found in environment")
        print("Please set it in .env file or export OPENAI_API_KEY=your_key")
        return

    # Step 1: Parse PDF
    print("\n📄 Step 1: Parsing PDF...")
    print("-" * 80)

    root_path = os.getenv("PROJECT_ROOT")
    pdf_path = root_path + "/docs/sample_article.pdf"
    parser = PDFParser()

    try:
        parsed_doc = parser.parse(pdf_path)
        print(f"✅ PDF parsed successfully")
        print(f"   Pages: {parsed_doc.page_count}")
        print(f"   Words: {parsed_doc.word_count}")
        print(f"   Sections detected: {len(parsed_doc.sections)}")
        print(f"   Parse time: {parsed_doc.parse_time:.2f}s")

        # Show detected sections
        if parsed_doc.sections:
            print(f"\n   Detected sections:")
            for section_name in parsed_doc.sections.keys():
                print(f"     - {section_name}")

    except Exception as e:
        print(f"❌ Failed to parse PDF: {e}")
        return

    # Step 2: Extract entities with LLM
    print("\n\n🤖 Step 2: Extracting entities with LLM...")
    print("-" * 80)
    print(f"Model: gpt")
    print(f"Estimated cost: ~$0.03 per paper")
    print(f"\nProcessing... (this may take 30-60 seconds)")

    config = {
        "temperature": 0.1,
        "max_tokens": 4000
    }

    pipeline = LLMPipeline(config=config, api_key=api_key, model="gpt-5-mini")

    try:
        result = pipeline.extract(
            parsed_doc=parsed_doc,
            paper_id="copd_bronchial_2025"
        )

        print(f"\n✅ Extraction completed!")

        # Display metrics
        print(f"\n📊 Extraction Metrics:")
        print(f"   Processing time: {result.metrics.processing_time:.2f}s")
        print(f"   Tokens used: {result.metrics.tokens_used:,}")
        print(f"   Cost: ${result.metrics.cost_usd:.4f}")
        print(f"   Entities extracted: {result.metrics.entities_extracted}")
        print(f"   Relationships extracted: {result.metrics.relationships_extracted}")

    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return

    # Step 3: Display results
    print("\n\n📦 Step 3: Extracted Entities")
    print("=" * 80)

    for entity_type, entities in result.entities.items():
        print(f"\n[{entity_type.upper()}] - {len(entities)} found")
        print("-" * 80)

        for i, entity in enumerate(entities[:3], 1):  # Show first 3 of each type
            preview = entity.text[:100] + "..." if len(entity.text) > 100 else entity.text
            print(f"{i}. {preview}")
            print(f"   Confidence: {entity.confidence:.2f} | Section: {entity.source_section}")

        if len(entities) > 3:
            print(f"   ... and {len(entities) - 3} more")

    # Display relationships
    print(f"\n\n🔗 Relationships")
    print("=" * 80)

    for i, rel in enumerate(result.relationships[:10], 1):  # Show first 10
        source_entity = result.get_entity_by_id(rel.source_id)
        target_entity = result.get_entity_by_id(rel.target_id)

        source_text = source_entity.text[:30] + "..." if source_entity else rel.source_id
        target_text = target_entity.text[:30] + "..." if target_entity else rel.target_id

        print(f"{i}. {source_text}")
        print(f"   --[{rel.relationship_type.value}]--> ")
        print(f"   {target_text}")
        print(f"   Confidence: {rel.confidence:.2f}\n")

    if len(result.relationships) > 10:
        print(f"... and {len(result.relationships) - 10} more relationships")

    # Step 4: Create Knowledge Graph
    print("\n\n🕸️  Step 4: Building Knowledge Graph")
    print("=" * 80)

    # Get all entities as flat list
    all_entities = []
    for entities_list in result.entities.values():
        all_entities.extend(entities_list)

    graph = KnowledgeGraph(
        paper_id=result.paper_id,
        entities=all_entities,
        relationships=result.relationships,
        metadata=result.metadata
    )

    stats = graph.statistics()
    print(f"Graph Statistics:")
    print(f"  Total entities: {stats['total_entities']}")
    print(f"  Total relationships: {stats['total_relationships']}")
    print(f"  Avg relationships per entity: {stats['avg_relationships_per_entity']:.2f}")

    print(f"\nEntity breakdown:")
    for entity_type, count in stats['entity_counts'].items():
        print(f"  {entity_type}: {count}")

    # Step 5: Save results
    print("\n\n💾 Step 5: Saving Results")
    print("=" * 80)

    # Generate output paths based on input PDF name
    pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]
    output_dir = f"{root_path}/results"
    os.makedirs(output_dir, exist_ok=True)

    result_path = f"{output_dir}/{pdf_basename}_result.json"
    graph_path = f"{output_dir}/{pdf_basename}_graph.json"
    svg_path = f"{output_dir}/{pdf_basename}_graph.svg"

    # Save extraction result
    result.to_json(result_path)
    print(f"✅ Saved extraction result: {result_path}")

    # Save knowledge graph
    graph.to_json(graph_path)
    print(f"✅ Saved knowledge graph: {graph_path}")

    # Generate SVG visualization (if matplotlib available)
    try:
        print(f"\n📊 Generating graph visualization...")
        graph.to_svg(svg_path, layout="hierarchical")
        print(f"✅ Saved visualization: {svg_path}")
    except ImportError:
        print(f"⚠️  Skipping visualization (matplotlib not installed)")
        print(f"   Install with: pip install matplotlib networkx")

    print("\n" + "=" * 80)
    print("✨ EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  • Extracted {result.total_entities()} entities")
    print(f"  • Found {result.total_relationships()} relationships")
    print(f"  • Processing time: {result.metrics.processing_time:.2f}s")
    print(f"  • Total cost: ${result.metrics.cost_usd:.4f}")
    print(f"\nThis extraction can now be used as Ground Truth for validating other pipelines!")


if __name__ == "__main__":
    main()
