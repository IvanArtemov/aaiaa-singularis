"""Test script for PDF parser"""

from src.parsers import PDFParser

def main():
    # Initialize parser
    parser = PDFParser()

    # Parse the article
    pdf_path = "/Users/ivanartemov/PycharmProjects/AAIAA/docs/sample_article.pdf"

    print("📄 Parsing PDF...")
    result = parser.parse(pdf_path)

    print(f"\n✅ Parsing completed in {result.parse_time:.2f}s\n")

    # Display results
    print("=" * 80)
    print("METADATA")
    print("=" * 80)
    for key, value in result.metadata.items():
        if value:
            print(f"{key}: {value}")

    print(f"\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    print(f"Page count: {result.page_count}")
    print(f"Word count: {result.word_count}")
    print(f"Character count: {len(result.text)}")

    print(f"\n" + "=" * 80)
    print("DETECTED SECTIONS")
    print("=" * 80)
    for section_name, content in result.sections.items():
        word_count = len(content.split())
        print(f"\n[{section_name.upper()}] - {word_count} words")
        print("-" * 80)
        # Show first 200 characters of each section
        preview = content[:200] + "..." if len(content) > 200 else content
        print(preview)

    print(f"\n" + "=" * 80)
    print("FULL TEXT PREVIEW (first 500 chars)")
    print("=" * 80)
    print(result.text)

if __name__ == "__main__":
    main()
