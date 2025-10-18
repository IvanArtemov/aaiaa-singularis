"""
Example: PDF Parsing with GROBID + spaCy Sentence Segmentation

This script demonstrates:
1. Parsing PDF with GROBID to extract IMRAD sections
2. Splitting each section into sentences using spaCy
3. Displaying all sentences with their full text

No embeddings - just document parsing and sentence extraction.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parsers import GrobidParser


def main():
    project_root = Path(__file__).parent.parent

    print("=" * 80)
    print("PDF Parsing with GROBID + spaCy Sentence Segmentation")
    print("=" * 80)
    print()

    # Configuration
    pdf_path = project_root / "docs" / "sample_article.pdf"

    # Check if PDF exists
    if not pdf_path.exists():
        print(f"❌ PDF not found: {pdf_path}")
        print("Please provide a valid PDF path")
        sys.exit(1)

    print(f"📄 Input: {pdf_path}")
    print()

    # ========== STEP 1: Parse PDF with GROBID ==========

    print("=" * 80)
    print("STEP 1: Parsing PDF with GROBID")
    print("=" * 80)
    print()

    parser = GrobidParser(
        grobid_server="https://lfoppiano-grobid.hf.space",
        timeout=60
    )

    print("⚙️  Sending PDF to GROBID service...")
    parsed_doc = parser.parse(pdf_path)

    print(f"✅ Parsing complete!")
    print()

    # Display metadata
    print("=" * 80)
    print("DOCUMENT METADATA")
    print("=" * 80)
    print(f"Title: {parsed_doc.title}")
    print()

    # Check if we have IMRAD sections
    if not parsed_doc.imrad_sections:
        print("❌ No IMRAD sections found!")
        sys.exit(1)

    print("=" * 80)
    print("IMRAD SECTIONS")
    print("=" * 80)
    for section_name, section_text in parsed_doc.imrad_sections.items():
        char_count = len(section_text)
        word_count = len(section_text.split())
        print(f"• {section_name.upper()}: {char_count} chars, {word_count} words")
    print()

    # ========== STEP 2: Initialize spaCy ==========

    print("=" * 80)
    print("STEP 2: Loading spaCy Model")
    print("=" * 80)
    print()

    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("✅ spaCy model 'en_core_web_sm' loaded")
    except OSError:
        print("❌ spaCy model 'en_core_web_sm' not found!")
        print("Install with: python -m spacy download en_core_web_sm")
        sys.exit(1)
    print()

    # ========== STEP 3: Extract Sentences from Each Section ==========

    print("=" * 80)
    print("STEP 3: Extracting Sentences by Section")
    print("=" * 80)
    print()

    min_sentence_length = 10  # Minimum character length

    total_sentences = 0

    for section_name, section_text in parsed_doc.imrad_sections.items():
        print("=" * 80)
        print(f"SECTION: {section_name.upper()}")
        print("=" * 80)
        print()
        print(f"Section text: {section_text}")
        # Process section with spaCy
        doc = nlp(section_text)

        # Extract sentences
        sentences = []
        for sent in doc.sents:
            text = sent.text.strip()

            # Filter: minimum length
            if len(text) < min_sentence_length:
                continue

            # Filter: must contain alphanumeric characters
            if not any(c.isalnum() for c in text):
                continue

            sentences.append(text)

        print(f"Found {len(sentences)} sentences:")
        print()

        # Display all sentences with numbering
        for i, sentence in enumerate(sentences, 1):
            print(f"[{i}] {sentence}")
            print()

        total_sentences += len(sentences)

    # ========== STEP 4: Summary ==========

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"Total sections: {len(parsed_doc.imrad_sections)}")
    print(f"Total sentences: {total_sentences}")
    print()

    # Section breakdown
    print("Sentences by section:")
    for section_name, section_text in parsed_doc.imrad_sections.items():
        doc = nlp(section_text)
        valid_sentences = 0

        for sent in doc.sents:
            text = sent.text.strip()
            if len(text) >= min_sentence_length and any(c.isalnum() for c in text):
                valid_sentences += 1

        print(f"  • {section_name}: {valid_sentences} sentences")

    print()
    print("=" * 80)
    print("✨ PARSING COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
