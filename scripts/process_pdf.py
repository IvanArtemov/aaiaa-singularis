#!/usr/bin/env python3
"""
CLI Tool: Process PDF through SciBERT-Nebius Pipeline

Extracts structured knowledge graph from scientific PDF using:
1. GROBID parser (PDF → IMRAD sections)
2. SciBERT embeddings (FREE)
3. Nebius LLM (keyword generation + validation)
4. ChromaDB semantic search (FREE)

Usage:
    python scripts/process_pdf.py --pdf paper.pdf
    python scripts/process_pdf.py -p paper.pdf -o results --no-svg

Output:
    results/{paper_id}_entities.json  - Entities and relationships
    results/{paper_id}_metrics.json   - Performance metrics
    results/{paper_id}_graph.svg      - Knowledge graph visualization
"""

import os
import sys
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parsers import GrobidParser
from src.pipelines import SciBertNebiusPipeline
from src.models import EntityType
from src.visualization.generate_svg import generate_svg_from_json

# Load environment variables
load_dotenv()


def check_prerequisites() -> bool:
    """
    Check if all required services and dependencies are available

    Returns:
        bool: True if all prerequisites are met, False otherwise
    """
    issues = []

    # Check Nebius API key
    if not os.getenv("NEBIUS_API_KEY"):
        issues.append("❌ NEBIUS_API_KEY not found in .env")
        issues.append("   Get your API key from: https://studio.nebius.com/")
    else:
        print("✅ Nebius API key found")

    # Check SciBERT dependencies
    try:
        import transformers
        import torch
        print("✅ SciBERT dependencies installed (transformers, torch)")
    except ImportError as e:
        issues.append(f"❌ SciBERT dependencies missing: {e}")
        issues.append("   Run: pip install transformers torch scikit-learn")

    # Check ChromaDB
    try:
        import chromadb
        print("✅ ChromaDB installed")
    except ImportError as e:
        issues.append(f"❌ ChromaDB missing: {e}")
        issues.append("   Run: pip install chromadb")

    # Check spaCy
    try:
        import spacy
        # Try to load model
        try:
            nlp = spacy.load("en_core_web_sm")
            print("✅ spaCy model (en_core_web_sm) loaded")
        except OSError:
            issues.append("❌ spaCy model not found")
            issues.append("   Run: python -m spacy download en_core_web_sm")
    except ImportError as e:
        issues.append(f"❌ spaCy missing: {e}")
        issues.append("   Run: pip install spacy")

    if issues:
        print("\n⚠️  Prerequisites not met:")
        for issue in issues:
            print(issue)
        print("\nPlease resolve these issues before running the pipeline.")
        return False

    print("\n✅ All prerequisites met!\n")
    return True


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Extract knowledge graph from scientific PDF using SciBERT-Nebius pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/process_pdf.py --pdf paper.pdf

  # Custom output directory
  python scripts/process_pdf.py -p paper.pdf -o my_results

  # Skip SVG generation (faster)
  python scripts/process_pdf.py -p paper.pdf --no-svg

  # Custom paper ID
  python scripts/process_pdf.py -p paper.pdf --paper-id experiment_001

Output files:
  {paper_id}_entities.json  - Extracted entities and relationships
  {paper_id}_metrics.json   - Performance metrics (time, cost, tokens)
  {paper_id}_graph.svg      - Visual knowledge graph (if --no-svg not set)
        """
    )

    parser.add_argument(
        "--pdf", "-p",
        type=str,
        required=True,
        help="Path to PDF file to process"
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="results",
        help="Output directory for results (default: results/)"
    )

    parser.add_argument(
        "--paper-id",
        type=str,
        default=None,
        help="Paper ID for output files (default: PDF filename without extension)"
    )

    parser.add_argument(
        "--no-svg",
        action="store_true",
        help="Skip SVG visualization generation (faster)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    return parser.parse_args()


def process_pdf(
    pdf_path: str,
    output_dir: str,
    paper_id: str,
    generate_svg: bool = True,
    verbose: bool = False
):
    """
    Process PDF through SciBERT-Nebius pipeline

    Args:
        pdf_path: Path to PDF file
        output_dir: Directory to save results
        paper_id: Unique identifier for the paper
        generate_svg: Whether to generate SVG visualization
        verbose: Enable verbose output
    """
    pdf_file = Path(pdf_path)

    # Validate PDF exists
    if not pdf_file.exists():
        print(f"❌ Error: PDF file not found: {pdf_path}")
        sys.exit(1)

    if not pdf_file.suffix.lower() == ".pdf":
        print(f"❌ Error: File is not a PDF: {pdf_path}")
        sys.exit(1)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SciBERT-Nebius Knowledge Graph Extraction")
    print("=" * 70)
    print(f"\n📄 PDF: {pdf_file.name}")
    print(f"📁 Output: {output_path.absolute()}/")
    print(f"🆔 Paper ID: {paper_id}\n")

    # ========== Step 1: Parse PDF with GROBID ==========
    print("=" * 70)
    print("STEP 1: Parsing PDF with GROBID")
    print("=" * 70)

    try:
        parser = GrobidParser()
        print("\n🔄 Parsing PDF (this may take 10-30 seconds)...")
        parsed_doc = parser.parse(str(pdf_file))

        print(f"\n✅ PDF parsed successfully:")
        print(f"  Title: {parsed_doc.title or 'Unknown'}")
        print(f"  Word count: {parsed_doc.word_count:,}")
        print(f"  Parse time: {parsed_doc.parse_time:.2f}s")

        if parsed_doc.imrad_sections:
            sections = list(parsed_doc.imrad_sections.keys())
            print(f"  Sections: {sections}")
        else:
            print("  ⚠️  Warning: No IMRAD sections found")
            print("  The PDF might not be a scientific paper or parsing failed.")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error parsing PDF: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # ========== Step 2: Run SciBERT-Nebius Pipeline ==========
    print("\n" + "=" * 70)
    print("STEP 2: Running SciBERT-Nebius Pipeline")
    print("=" * 70)

    try:
        pipeline = SciBertNebiusPipeline()
        print(f"\nPipeline: {pipeline.get_description()}")
        print(f"Estimated cost: ${pipeline.get_estimated_cost():.4f} per paper")

        print("\n🚀 Extracting entities (this may take 60-90 seconds)...")
        print("  [Stage 1] LLM keyword generation (~$0.003)")
        print("  [Stage 2] Semantic retrieval with ChromaDB (FREE)")
        print("  [Stage 3] LLM validation (~$0.015)")
        print()

        result = pipeline.extract(
            parsed_doc=parsed_doc,
            paper_id=paper_id
        )

        print(f"\n✅ Extraction completed!")

    except Exception as e:
        print(f"\n❌ Error running pipeline: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # ========== Step 3: Display Results Summary ==========
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    # Count entities
    total_entities = 0
    entity_counts = {}

    for entity_type in EntityType:
        entities = result.entities.get(entity_type.value, [])
        count = len(entities)
        if count > 0:
            entity_counts[entity_type.value] = count
            total_entities += count

    print("\n📊 Entities extracted:")
    for entity_type, count in entity_counts.items():
        print(f"  {entity_type.upper()}: {count}")

    print(f"\n  TOTAL ENTITIES: {total_entities}")
    print(f"  RELATIONSHIPS: {len(result.relationships)}")

    # Metrics
    metrics = result.metrics
    print(f"\n⏱️  Processing time: {metrics.processing_time:.2f}s")
    print(f"💰 Cost: ${metrics.cost_usd:.6f}")
    print(f"🔢 Tokens used: {metrics.tokens_used:,}")

    # ========== Step 4: Save Results ==========
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    saved_files = []

    # 1. Save entities and relationships JSON
    entities_file = output_path / f"{paper_id}_entities.json"

    entities_data = {
        "paper_id": result.paper_id,
        "title": parsed_doc.title,
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
        ]
    }

    with open(entities_file, 'w', encoding='utf-8') as f:
        json.dump(entities_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Entities saved: {entities_file.name}")
    saved_files.append(entities_file)

    # 2. Save metrics JSON
    metrics_file = output_path / f"{paper_id}_metrics.json"

    metrics_data = {
        "paper_id": result.paper_id,
        "processing_time_seconds": metrics.processing_time,
        "tokens_used": metrics.tokens_used,
        "cost_usd": metrics.cost_usd,
        "entities_extracted": metrics.entities_extracted,
        "relationships_extracted": metrics.relationships_extracted,
        "entity_breakdown": entity_counts,
        "metadata": metrics.metadata
    }

    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Metrics saved: {metrics_file.name}")
    saved_files.append(metrics_file)

    # 3. Generate SVG visualization
    if generate_svg:
        svg_file = output_path / f"{paper_id}_graph.svg"

        try:
            print(f"\n🎨 Generating SVG visualization...")
            # Generate SVG from the entities JSON file
            generate_svg_from_json(str(entities_file), str(svg_file))
            print(f"✅ SVG saved: {svg_file.name}")
            saved_files.append(svg_file)
        except Exception as e:
            print(f"⚠️  Warning: SVG generation failed: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
    else:
        print("\n⏭️  Skipping SVG generation (--no-svg)")

    # ========== Final Summary ==========
    print("\n" + "=" * 70)
    print("COMPLETED")
    print("=" * 70)

    print(f"\n✨ Successfully processed: {pdf_file.name}")
    print(f"\n📁 Output files ({len(saved_files)}):")
    for file in saved_files:
        file_size = file.stat().st_size / 1024  # KB
        print(f"  {file.name} ({file_size:.1f} KB)")

    print(f"\n💡 Key Stats:")
    print(f"  Entities: {total_entities}")
    print(f"  Relationships: {len(result.relationships)}")
    print(f"  Cost: ${metrics.cost_usd:.6f}")
    print(f"  Time: {metrics.processing_time:.1f}s")

    # Cost projection
    if metrics.cost_usd > 0:
        cost_per_1k = metrics.cost_usd * 1000
        print(f"\n💰 Cost Projection:")
        print(f"  1,000 papers: ${cost_per_1k:.2f}")
        print(f"  10,000 papers: ${cost_per_1k * 10:.2f}")

    print("\n" + "=" * 70)


def main():
    """Main entry point"""
    args = parse_arguments()

    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)

    # Determine paper ID
    if args.paper_id:
        paper_id = args.paper_id
    else:
        paper_id = Path(args.pdf).stem

    # Process PDF
    try:
        process_pdf(
            pdf_path=args.pdf,
            output_dir=args.output_dir,
            paper_id=paper_id,
            generate_svg=not args.no_svg,
            verbose=args.verbose
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
