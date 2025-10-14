"""Extract hypotheses from PDF using HypothesisExtractor"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parsers import PDFParser
from src.extractors import HypothesisExtractor


def main():
    """Extract hypotheses from 2510.05495v1.pdf"""

    print("=" * 80)
    print("HYPOTHESIS EXTRACTION - Pattern-based Approach")
    print("=" * 80)
    print()

    # Path to PDF
    pdf_path = Path("/Users/ivanartemov/PycharmProjects/AAIAA/docs/articles/2510_05495v1/2510.05495v1.pdf")

    if not pdf_path.exists():
        print(f"ERROR: PDF file not found at {pdf_path}")
        print("\nPlease ensure the file exists at:")
        print(f"  {pdf_path}")
        return

    print(f"Processing: {pdf_path.name}")
    print()

    # Step 1: Parse PDF with IMRAD sections
    print("Step 1: Parsing PDF...")
    parser = PDFParser(enable_imrad=True)
    parsed_doc = parser.parse(str(pdf_path))

    print(f"  ✓ Pages: {parsed_doc.page_count}")
    print(f"  ✓ Words: {parsed_doc.word_count:,}")
    print(f"  ✓ Parse time: {parsed_doc.parse_time:.2f}s")

    if parsed_doc.imrad_sections:
        print(f"  ✓ IMRAD sections detected: {', '.join(parsed_doc.imrad_sections.keys())}")
    else:
        print("  ! No IMRAD sections detected")
    print()

    # Step 2: Extract hypotheses using pattern matcher
    print("Step 2: Extracting hypotheses from entire document...")
    extractor = HypothesisExtractor(confidence_threshold=0.6)

    print(f"  - Scanning full document ({len(parsed_doc.text.split())} words)...")
    print(f"  - Total characters: {len(parsed_doc.text):,}")

    # Extract from full document
    all_hypotheses = extractor.extract(
        parsed_doc.text,
        section_name="full_document",
        use_spacy=True
    )

    print(f"  ✓ Found {len(all_hypotheses)} hypotheses across entire document")
    print()

    # Also show breakdown by section if IMRAD sections are available
    if parsed_doc.imrad_sections:
        print("  Breakdown by section:")
        section_hypotheses = {}

        for section_name, section_text in parsed_doc.imrad_sections.items():
            section_hyps = extractor.extract(section_text, section_name=section_name, use_spacy=True)
            if section_hyps:
                section_hypotheses[section_name] = section_hyps
                print(f"    - {section_name}: {len(section_hyps)} hypotheses")

        print()

    print()

    # Step 3: Display results
    print("=" * 80)
    print(f"RESULTS: {len(all_hypotheses)} hypotheses extracted")
    print("=" * 80)
    print()

    if all_hypotheses:
        for i, hypothesis in enumerate(all_hypotheses, 1):
            print(f"Hypothesis {i}:")
            print(f"  Confidence: {hypothesis.confidence:.2f}")
            print(f"  Section: {hypothesis.source_section or 'unknown'}")
            print(f"  Text: {hypothesis.text}")
            print()
    else:
        print("No hypotheses found with pattern matching.")
        print("\nTip: This may mean:")
        print("  - The paper doesn't explicitly state hypotheses")
        print("  - Hypotheses are phrased differently than expected patterns")
        print("  - The patterns need to be adjusted for this paper")
        print("\nConsider using the Hybrid Pipeline with LLM fallback for better coverage.")

    print("=" * 80)


if __name__ == "__main__":
    main()