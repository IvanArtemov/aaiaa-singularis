"""Example usage of Adaptive Token Pipeline with LLM-guided token generation"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelines import AdaptiveRegexPipeline
from src.parsers import PDFParser
from src.config.settings import settings


def run_adaptive_pipeline(parsed_doc, paper_id: str):
    """
    Run Adaptive Token Pipeline and display results

    Args:
        parsed_doc: ParsedDocument from parser
        paper_id: Paper identifier
    """
    # Initialize pipeline with settings
    provider_config = settings.get_provider_config()
    pipeline = AdaptiveRegexPipeline(
        config={},
        llm_provider=settings.active_provider,
        llm_model=settings.get_model("chat"),
        temperature=provider_config.get("temperature", 0.2),
        confidence_threshold=0.7
    )

    # Run extraction
    result = pipeline.extract(
        parsed_doc=parsed_doc,
        paper_id=paper_id
    )

    # ========== DISPLAY RESULTS ==========

    print("\n" + "=" * 80)
    print("EXTRACTION RESULTS")
    print("=" * 80)

    # Metrics summary
    metrics = result.metrics
    print(f"\n📊 Metrics:")
    print(f"  Processing time: {metrics.processing_time:.2f}s")
    print(f"  Total cost: ${metrics.cost_usd:.4f}")
    print(f"  Tokens used: {metrics.tokens_used:,}")
    print(f"  Entities extracted: {metrics.entities_extracted}")
    print(f"  Relationships: {metrics.relationships_extracted}")

    # Generated tokens
    if "generated_tokens" in result.metrics.metadata:
        tokens_dict = result.metrics.metadata["generated_tokens"]
        required = tokens_dict.get("required_tokens", [])
        optional = tokens_dict.get("optional_tokens", [])

        print("\n" + "=" * 80)
        print("🔑 GENERATED TOKENS (Context-Aware)")
        print("=" * 80)

        print(f"\n✅ Required tokens ({len(required)}):")
        if required:
            for i, token in enumerate(required, 1):
                print(f"  {i}. {token}")
        else:
            print("  (none)")

        print(f"\n🔹 Optional tokens ({len(optional)}):")
        if optional:
            for i, token in enumerate(optional[:15], 1):  # Show first 15
                print(f"  {i}. {token}")
            if len(optional) > 15:
                print(f"  ... and {len(optional) - 15} more")
        else:
            print("  (none)")

    # Extracted facts
    print("\n" + "=" * 80)
    print("📚 EXTRACTED FACTS")
    print("=" * 80)
    facts = result.entities.get("fact", [])
    if facts:
        for i, fact in enumerate(facts, 1):
            preview = fact.text[:120] + "..." if len(fact.text) > 120 else fact.text
            print(f"\n{i}. [{fact.confidence:.2f}] {preview}")
    else:
        print("No facts extracted.")

    # Extracted hypotheses
    print("\n" + "=" * 80)
    print("💡 EXTRACTED HYPOTHESES")
    print("=" * 80)
    hypotheses = result.entities.get("hypothesis", [])
    if hypotheses:
        for i, hyp in enumerate(hypotheses, 1):
            preview = hyp.text[:120] + "..." if len(hyp.text) > 120 else hyp.text
            matched_req = hyp.metadata.get("matched_required_tokens", [])
            matched_opt = hyp.metadata.get("matched_optional_tokens", [])

            print(f"\n{i}. [{hyp.confidence:.2f}] {preview}")
            if matched_req:
                print(f"   ✅ Required: {', '.join(matched_req[:4])}")
            if matched_opt:
                print(f"   🔹 Optional: {', '.join(matched_opt[:4])}")
    else:
        print("No hypotheses extracted.")

    # Save results
    results_dir = settings.project_root / "results"
    results_dir.mkdir(exist_ok=True)

    output_path = results_dir / f"{paper_id}_adaptive_tokens.json"
    result.to_json(str(output_path))

    print("\n" + "=" * 80)
    print(f"✅ Results saved to: {output_path}")
    print("=" * 80)

    return result


def main():
    """Demonstrate Adaptive Token Pipeline"""

    print("=" * 80)
    print("ADAPTIVE TOKEN PIPELINE DEMO")
    print("LLM-Guided Token Generation for Context-Aware Extraction")
    print("=" * 80)
    print()
    print("Configuration:")
    print(f"  📁 Project root: {settings.project_root}")
    print(f"  🤖 LLM Provider: {settings.active_provider}")
    print(f"  🧠 Model: {settings.get_model('chat')}")
    print(f"  🌡️  Temperature: {settings.get_provider_config().get('temperature', 0.2)}")
    print()
    print("Strategy:")
    print("  1. Extract FACTS from Abstract + Introduction using LLM")
    print("  2. Generate context-aware TOKENS based on facts using LLM")
    print("  3. Apply token matching to extract HYPOTHESES (FREE)")
    print()
    print("Token advantages:")
    print("  ✅ Simpler than regex (no validation)")
    print("  ✅ Faster execution (substring matching)")
    print("  ✅ Flexible scoring (token combinations)")
    print("  ✅ More interpretable (see which tokens matched)")
    print()
    print("Target Cost: ~$0.01-0.012 per paper")
    print("=" * 80)

    # Use sample article from docs/
    sample_pdf = settings.project_root / "docs" / "sample_article.pdf"

    if not sample_pdf.exists():
        print(f"\n❌ ERROR: Sample article not found at {sample_pdf}")
        print("Please ensure docs/sample_article.pdf exists.")
        return

    print(f"\n📄 Using sample article: {sample_pdf.name}")
    print()

    # Parse PDF
    print("Parsing PDF with IMRAD sections...")
    parser = PDFParser(enable_imrad=True)
    parsed_doc = parser.parse(str(sample_pdf))

    print(f"  - Pages: {parsed_doc.page_count}")
    print(f"  - Words: {parsed_doc.word_count}")
    print(f"  - IMRAD sections: {list(parsed_doc.imrad_sections.keys()) if parsed_doc.imrad_sections else 'None'}")

    # Run pipeline
    run_adaptive_pipeline(
        parsed_doc=parsed_doc,
        paper_id=sample_pdf.stem
    )

    print("\n✅ Demo completed!")


if __name__ == "__main__":
    main()
