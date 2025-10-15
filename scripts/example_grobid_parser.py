"""Demo script for GROBID parser with comparison to PyMuPDF parser"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parsers import PDFParser, GrobidParser


def compare_parsers(pdf_path: str):
    """
    Compare GROBID parser with PyMuPDF parser

    Args:
        pdf_path: Path to PDF file
    """
    print("=" * 80)
    print("PDF PARSER COMPARISON: GROBID vs PyMuPDF")
    print("=" * 80)
    print()
    print(f"📄 Processing: {Path(pdf_path).name}")
    print()

    # ========== Parse with PyMuPDF ==========
    print("[1/2] Parsing with PyMuPDF (fitz)...")
    print("-" * 80)

    try:
        pymupdf_parser = PDFParser(enable_imrad=True)
        pymupdf_result = pymupdf_parser.parse(pdf_path)

        print(f"✅ PyMuPDF completed in {pymupdf_result.parse_time:.2f}s")
        print(f"   Pages: {pymupdf_result.page_count}")
        print(f"   Words: {pymupdf_result.word_count}")
        print(f"   IMRAD sections: {len(pymupdf_result.imrad_sections or {})}")
        if pymupdf_result.imrad_sections:
            print(f"   Detected sections: {', '.join(pymupdf_result.imrad_sections.keys())}")

    except Exception as e:
        print(f"❌ PyMuPDF parsing failed: {e}")
        pymupdf_result = None

    # ========== Parse with GROBID ==========
    print()
    print("[2/2] Parsing with GROBID...")
    print("-" * 80)

    try:
        grobid_parser = GrobidParser(
            grobid_server="https://lfoppiano-grobid.hf.space",
            consolidate_header=True,
            tei_coordinates=True,
            segment_sentences=True
        )
        grobid_result = grobid_parser.parse(pdf_path)

        print(f"✅ GROBID completed in {grobid_result.parse_time:.2f}s")
        print(f"   Pages: {grobid_result.page_count or 'N/A'}")
        print(f"   Words: {grobid_result.word_count}")
        print(f"   IMRAD sections: {len(grobid_result.imrad_sections or {})}")
        if grobid_result.imrad_sections:
            print(f"   Detected sections: {', '.join(grobid_result.imrad_sections.keys())}")

    except ConnectionError as e:
        print(f"❌ GROBID parsing failed: {e}")
        print()
        print("💡 To start GROBID Docker container:")
        print("   docker pull lfoppiano/grobid:0.8.0")
        print("   docker run --rm -p 8070:8070 lfoppiano/grobid:0.8.0")
        print()
        grobid_result = None

    except Exception as e:
        print(f"❌ GROBID parsing failed: {e}")
        grobid_result = None

    # ========== Compare Results ==========
    if pymupdf_result and grobid_result:
        print()
        print("=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)

        # Side-by-side metrics
        print()
        print(f"{'Metric':<30} | {'PyMuPDF':<20} | {'GROBID':<20}")
        print("-" * 80)
        print(f"{'Parse Time (s)':<30} | {pymupdf_result.parse_time:<20.2f} | {grobid_result.parse_time:<20.2f}")
        print(f"{'Word Count':<30} | {pymupdf_result.word_count:<20} | {grobid_result.word_count:<20}")
        print(f"{'IMRAD Sections':<30} | {len(pymupdf_result.imrad_sections or {}):<20} | {len(grobid_result.imrad_sections or {}):<20}")

        # Metadata comparison
        print()
        print("=" * 80)
        print("METADATA COMPARISON")
        print("=" * 80)

        print()
        print("PyMuPDF Metadata:")
        for key, value in pymupdf_result.metadata.items():
            if value:
                print(f"  {key}: {value}")

        print()
        print("GROBID Metadata:")
        for key, value in grobid_result.metadata.items():
            if value:
                print(f"  {key}: {value}")

        # Section content comparison
        print()
        print("=" * 80)
        print("SECTION CONTENT PREVIEW")
        print("=" * 80)

        sections_to_compare = ["abstract", "introduction", "methods", "results", "conclusion"]

        for section_name in sections_to_compare:
            pymupdf_section = (pymupdf_result.imrad_sections or {}).get(section_name, "")
            grobid_section = (grobid_result.imrad_sections or {}).get(section_name, "")

            if pymupdf_section or grobid_section:
                print()
                print(f"[{section_name.upper()}]")
                print("-" * 80)

                print(f"PyMuPDF ({len(pymupdf_section.split())} words):")
                preview = pymupdf_section[:200] + "..." if len(pymupdf_section) > 200 else pymupdf_section or "(not found)"
                print(f"  {preview}")

                print()
                print(f"GROBID ({len(grobid_section.split())} words):")
                preview = grobid_section[:200] + "..." if len(grobid_section) > 200 else grobid_section or "(not found)"
                print(f"  {preview}")

        # Quality assessment
        print()
        print("=" * 80)
        print("QUALITY ASSESSMENT")
        print("=" * 80)
        print()
        print("PyMuPDF:")
        print("  ✅ Fast parsing (local)")
        print("  ✅ No external dependencies")
        print("  ⚠️  IMRAD detection based on heuristics")
        print("  ⚠️  May miss section boundaries")
        print()
        print("GROBID:")
        print("  ✅ ML-based extraction (>90% accuracy)")
        print("  ✅ Structured TEI XML output")
        print("  ✅ Better metadata extraction")
        print("  ✅ Parsed bibliographic references")
        print("  ⚠️  Requires Docker service")
        print("  ⚠️  Slower than local parsing")

        # Recommendation
        print()
        print("=" * 80)
        print("RECOMMENDATION")
        print("=" * 80)
        print()
        print("Use GROBID parser for production pipeline:")
        print("  • Higher quality IMRAD extraction → better entity detection")
        print("  • Structured metadata → easier downstream processing")
        print("  • Free (runs locally) → no cost impact")
        print("  • ~3-5 sec/paper → acceptable for batch processing")
        print()
        print("Use PyMuPDF parser for:")
        print("  • Quick prototyping without Docker")
        print("  • Simple text extraction tasks")
        print("  • When GROBID service unavailable")

    elif grobid_result:
        # Show GROBID results only
        print()
        print("=" * 80)
        print("GROBID PARSING RESULTS")
        print("=" * 80)

        print()
        print("Metadata:")
        for key, value in grobid_result.metadata.items():
            if value:
                print(f"  {key}: {value}")

        print()
        print("IMRAD Sections:")
        for section_name, content in (grobid_result.imrad_sections or {}).items():
            word_count = len(content.split())
            print(f"  [{section_name.upper()}] - {word_count} words")

    elif pymupdf_result:
        # Show PyMuPDF results only
        print()
        print("=" * 80)
        print("PYMUPDF PARSING RESULTS")
        print("=" * 80)

        print()
        print("Metadata:")
        for key, value in pymupdf_result.metadata.items():
            if value:
                print(f"  {key}: {value}")

        print()
        print("IMRAD Sections:")
        for section_name, content in (pymupdf_result.imrad_sections or {}).items():
            word_count = len(content.split())
            print(f"  [{section_name.upper()}] - {word_count} words")


def main():
    """Demo GROBID parser with sample article"""

    print()
    print("=" * 80)
    print("GROBID PARSER DEMO")
    print("Structured PDF Extraction using GROBID TEI XML")
    print("=" * 80)
    print()
    print("GROBID (GeneRation Of BIbliographic Data)")
    print("- ML-based extraction for scientific PDFs")
    print("- >90% accuracy for scientific papers")
    print("- TEI XML output with >55 label types")
    print("- FREE (runs locally in Docker)")
    print()
    print("Prerequisites:")
    print("  docker pull lfoppiano/grobid:0.8.0")
    print("  docker run --rm -p 8070:8070 lfoppiano/grobid:0.8.0")
    print("=" * 80)

    # Use sample article from docs/
    sample_pdf = Path(__file__).parent.parent / "docs" / "articles" / "2508.03438v1.pdf"

    if not sample_pdf.exists():
        print()
        print(f"❌ ERROR: Sample article not found at {sample_pdf}")
        print("Please ensure docs/sample_article.pdf exists.")
        return

    # Compare parsers
    compare_parsers(str(sample_pdf))

    print()
    print("=" * 80)
    print("✅ Demo completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
