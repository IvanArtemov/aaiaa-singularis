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
        use_llm_fallback=config.get('use_llm_fallback', True),
        confidence_threshold=config.get('confidence_threshold', 0.75),
        llm_provider=config.get('llm_provider', 'openai'),
        llm_model=config.get('llm_model', 'gpt-5-mini')
    )

    print(f"Pipeline: {pipeline.get_description()}")
    print(f"Estimated cost: ${pipeline.get_estimated_cost():.4f} per paper")
    print()

    # Example 1: Extract from PDF file
    print("-" * 80)
    print("Example 1: Extract from PDF File")
    print("-" * 80)

    # Check for sample PDF
    articles_dir = Path(__file__).parent.parent / "articles" / "pdfs"
    pdf_files = list(articles_dir.glob("*.pdf")) if articles_dir.exists() else []

    if pdf_files:
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
            paper_text=parsed_doc.text,
            paper_id=pdf_path.stem,
            metadata={"title": parsed_doc.metadata.get("title", "Unknown")}
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

    else:
        print("No PDF files found in articles/pdfs/")
        print("Using sample text instead...")
        print()

        # Example 2: Extract from sample text
        sample_text = """
        Introduction

        Previous studies have shown that aging is associated with cellular senescence.
        The role of mitochondrial dysfunction in aging has been well established.

        We hypothesize that targeting senescent cells can slow down the aging process.
        The aim of this study is to investigate whether senolytics can extend lifespan.

        Materials and Methods

        We used flow cytometry to measure cellular markers.
        Participants were recruited from the local community.
        The statistical analysis was performed using R software.
        Data were obtained from the GEO database (accession: GSE12345).

        Results

        We found that senolytic treatment significantly increased lifespan (p < 0.01).
        The mean lifespan was 25% higher in the treatment group compared to control.

        Discussion and Conclusion

        These results suggest that targeting senescent cells is a promising strategy for extending lifespan.
        Further studies are needed to validate these findings in human populations.
        """

        print("Processing sample text...")
        print()

        result = pipeline.extract(
            paper_text=sample_text,
            paper_id="sample_001",
            metadata={"title": "Sample Paper on Aging"}
        )

        # Display results (same as above)
        print("EXTRACTION RESULTS:")
        print("=" * 80)

        print("\nEntities extracted:")
        for entity_type, entities in result.entities.items():
            print(f"  - {entity_type.upper()}: {len(entities)} entities")
            for i, entity in enumerate(entities, 1):
                preview = entity.text[:80] + "..." if len(entity.text) > 80 else entity.text
                print(f"    {i}. [{entity.confidence:.2f}] {preview}")

        print(f"\nRelationships: {len(result.relationships)} connections")

        print("\nMETRICS:")
        print(f"Time: {result.metrics.processing_time:.2f}s")
        print(f"Cost: ${result.metrics.cost_usd:.4f}")
        print(f"Entities: {result.metrics.entities_extracted}")


if __name__ == "__main__":
    main()
