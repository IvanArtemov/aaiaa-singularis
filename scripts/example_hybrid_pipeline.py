"""Example usage of Hybrid Pipeline for cost-optimized extraction"""

import sys
import os
import yaml
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelines import HybridPipeline
from src.parsers import PDFParser


def load_config():
    """Load pipeline configuration"""
    config_path = Path(__file__).parent.parent / "src" / "config" / "pipeline_config.yaml"

    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config.get('hybrid_pipeline', {})
    else:
        print("Warning: Config file not found. Using defaults.")
        return {}


def main():
    """Demonstrate hybrid pipeline extraction"""

    print("=" * 80)
    print("HYBRID PIPELINE DEMO - Cost-Optimized Extraction")
    print("=" * 80)
    print()

    # Load configuration
    config = load_config()

    # Initialize hybrid pipeline
    print("Initializing Hybrid Pipeline...")
    pipeline = HybridPipeline(
        config=config,
        confidence_threshold=config.get('confidence_threshold', 0.75),
        llm_provider=config.get('llm_provider', 'openai'),
        llm_model=config.get('llm_model', 'gpt-5-mini')
    )

    print(f"Pipeline: {pipeline.get_description()}")
    print()

    # Example 1: Extract from PDF file
    print("-" * 80)
    print("Example 1: Extract from PDF File")
    print("-" * 80)

    # Check for sample PDF
    articles_dir = Path(__file__).parent.parent / "articles" / "pdfs"
    pdf_files = list(articles_dir.glob("*.pdf")) if articles_dir.exists() else []

    pdf_path = pdf_files[0]
    print(f"Processing: {pdf_path.name}")
    print()

    # Parse PDF
    print("Step 1: Parsing PDF with IMRAD sections...")
    parser = PDFParser(enable_imrad=True)
    parsed_doc = parser.parse(str(pdf_path))

    print(f"  - Pages: {parsed_doc.page_count}")
    print(f"  - Words: {parsed_doc.word_count}")
    print(f"  - IMRAD sections: {list(parsed_doc.imrad_sections.keys()) if parsed_doc.imrad_sections else 'None'}")
    print()

    # Extract entities
    print("Step 2: Extracting entities using hybrid approach...")
    result = pipeline.extract(
        parsed_doc=parsed_doc,
        paper_id=pdf_path.stem
    )

    # Display results
    print()
    print("EXTRACTION RESULTS:")
    print("=" * 80)

    # Entities by type
    print("\nEntities extracted:")
    for entity_type, entities in result.entities.items():
        print(f"  - {entity_type.upper()}: {len(entities)} entities")
        for i, entity in enumerate(entities[:3], 1):  # Show first 3
            preview = entity.text[:80] + "..." if len(entity.text) > 80 else entity.text
            print(f"    {i}. [{entity.confidence:.2f}] {preview}")
        if len(entities) > 3:
            print(f"    ... and {len(entities) - 3} more")

    # Relationships
    print(f"\nRelationships: {len(result.relationships)} connections")
    for i, rel in enumerate(result.relationships[:5], 1):  # Show first 5
        print(f"  {i}. {rel.source_id} --[{rel.relationship_type.value}]--> {rel.target_id}")
    if len(result.relationships) > 5:
        print(f"  ... and {len(result.relationships) - 5} more")

    # Metrics
    print("\nPERFORMANCE METRICS:")
    print("=" * 80)
    metrics = result.metrics
    print(f"Processing time: {metrics.processing_time:.2f}s")
    print(f"Entities extracted: {metrics.entities_extracted}")
    print(f"Relationships: {metrics.relationships_extracted}")
    print(f"Tokens used: {metrics.tokens_used:,}")
    print(f"Cost: ${metrics.cost_usd:.4f}")
    print()

    # Extraction methods breakdown
    if "extraction_methods" in metrics.metadata:
        print("Extraction methods breakdown:")
        for method, count in metrics.metadata["extraction_methods"].items():
            print(f"  - {method}: {count} entities")

    print()

    # Cost comparison
    print("COST COMPARISON:")
    print("=" * 80)
    print(f"Hybrid Pipeline: ${metrics.cost_usd:.4f}")
    print(f"LLM-Only (estimated): ${0.03:.4f}")
    print(f"Savings: {((0.03 - metrics.cost_usd) / 0.03 * 100):.1f}%")
    print()

    # Save results
    output_path = Path(__file__).parent.parent / "results" / f"{pdf_path.stem}_hybrid.json"
    output_path.parent.mkdir(exist_ok=True)
    result.to_json(str(output_path))
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main()
