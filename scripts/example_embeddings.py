"""
Example script demonstrating sentence embeddings creation

This script shows how to:
1. Parse a PDF document
2. Split sections into sentences
3. Create embeddings for all sentences
4. Display statistics and sample data
"""

import sys
import os
from pathlib import Path

def main():
    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from src.parsers import GrobidParser
    from src.extractors import SentenceEmbedder

    print("=" * 80)
    print("Sentence Embeddings Example")
    print("=" * 80)

    # Path to sample PDF
    pdf_path = project_root / "docs" / "sample_article.pdf"

    if not pdf_path.exists():
        print(f"\nError: Sample PDF not found at {pdf_path}")
        print("Please place a PDF file at docs/sample_article.pdf")
        return

    print(f"\nPDF: {pdf_path.name}")

    # ========== STEP 1: Parse PDF ==========
    print("\n" + "=" * 80)
    print("STEP 1: Parsing PDF with GROBID")
    print("=" * 80)

    try:
        parser = GrobidParser()
        parsed_doc = parser.parse(str(pdf_path))
    except Exception as e:
        print(f"\nError parsing PDF: {e}")
        print("\nMake sure GROBID server is running:")
        print("  docker run --rm -p 8070:8070 lfoppiano/grobid:0.8.0")
        return

    print(f"\nDocument parsed successfully:")
    print(f"  Title: {parsed_doc.title or 'Unknown'}")
    print(f"  Word count: {parsed_doc.word_count:,}")
    print(f"  Parse time: {parsed_doc.parse_time:.2f}s")

    if parsed_doc.imrad_sections:
        print(f"  Sections found: {list(parsed_doc.imrad_sections.keys())}")
        print(f"\nSection word counts:")
        for section_name, section_text in parsed_doc.imrad_sections.items():
            word_count = len(section_text.split())
            print(f"    {section_name}: {word_count:,} words")
    else:
        print("  Warning: No IMRAD sections found")
        return

    # ========== STEP 2: Create Sentence Embeddings ==========
    print("\n" + "=" * 80)
    print("STEP 2: Creating Sentence Embeddings")
    print("=" * 80)

    embedder = SentenceEmbedder(
        llm_provider="ollama",
        batch_size=100,
        min_sentence_length=10
    )

    print("\nProcessing document...")
    parsed_doc = embedder.process_document(parsed_doc)

    # ========== STEP 3: Display Results ==========
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print(f"\nTotal sentences extracted: {len(parsed_doc.sentences)}")

    # Sentences by section
    print("\nSentences by section:")
    for section_name in parsed_doc.imrad_sections.keys():
        section_sentences = parsed_doc.get_sentences_by_section(section_name)
        print(f"  {section_name}: {len(section_sentences)} sentences")

    # Sample sentences
    print("\nSample sentences (first 3):")
    for i, sentence in enumerate(parsed_doc.sentences[:3]):
        print(f"\n  [{i+1}] {sentence.section} (position {sentence.position}):")
        text_preview = sentence.text[:100] + "..." if len(sentence.text) > 100 else sentence.text
        print(f"      Text: {text_preview}")
        print(f"      Length: {len(sentence)} chars")
        print(f"      Has embedding: {sentence.has_embedding()}")
        if sentence.has_embedding():
            print(f"      Embedding dims: {len(sentence.embedding)}")
            print(f"      Embedding sample: [{sentence.embedding[0]:.4f}, {sentence.embedding[1]:.4f}, ...]")

    # ========== STEP 4: Display Metrics ==========
    print("\n" + "=" * 80)
    print("PERFORMANCE METRICS")
    print("=" * 80)

    metrics = embedder.get_metrics()

    print(f"\nTotal sentences: {metrics['total_sentences']}")
    print(f"Total tokens: {metrics['total_tokens']:,}")
    print(f"Embedding time: {metrics['embedding_time_seconds']:.2f}s")
    print(f"Sentences/second: {metrics['sentences_per_second']:.1f}")

    print(f"\nCache performance:")
    print(f"  Cache hits: {metrics['cache_hits']}")
    print(f"  Cache hit rate: {metrics['cache_hit_rate']:.1%}")

    print(f"\nCost analysis:")
    print(f"  Cost for this document: ${metrics['total_cost_usd']:.6f}")
    print(f"  Cost per sentence: ${metrics['total_cost_usd']/max(1, metrics['total_sentences']):.8f}")

    # ========== STEP 5: Cost Projection ==========
    print("\n" + "=" * 80)
    print("COST PROJECTION")
    print("=" * 80)

    cost_per_paper = metrics['total_cost_usd']

    print(f"\nCost per paper: ${cost_per_paper:.6f}")
    print(f"Cost per 1,000 papers: ${cost_per_paper * 1000:.2f}")
    print(f"Cost per 1M papers: ${cost_per_paper * 1_000_000:,.2f}")
    print(f"Cost for 50M papers: ${cost_per_paper * 50_000_000:,.2f}")

    if cost_per_paper < 0.0001:
        print(f"\nExcellent! Embedding cost is minimal (~$0.0001/paper)")
    elif cost_per_paper < 0.001:
        print(f"\nGood! Embedding cost is very low (~$0.001/paper)")
    else:
        print(f"\nNote: Cost is ${cost_per_paper:.6f}/paper")

    # ========== STEP 6: Verify Embeddings ==========
    print("\n" + "=" * 80)
    print("EMBEDDING VERIFICATION")
    print("=" * 80)

    has_embeddings = parsed_doc.has_embeddings()
    print(f"\nDocument has embeddings: {has_embeddings}")

    if has_embeddings:
        # Check all sentences have embeddings
        total_sentences = len(parsed_doc.sentences)
        sentences_with_embeddings = sum(1 for s in parsed_doc.sentences if s.has_embedding())

        print(f"Sentences with embeddings: {sentences_with_embeddings}/{total_sentences}")

        if sentences_with_embeddings == total_sentences:
            print("All sentences have embeddings!")
        else:
            print(f"Warning: {total_sentences - sentences_with_embeddings} sentences missing embeddings")

        # Check embedding dimensions
        if total_sentences > 0:
            first_embedding = parsed_doc.sentences[0].embedding
            if first_embedding:
                print(f"Embedding dimensions: {len(first_embedding)}")
                print(f"Expected: 1536 (text-embedding-3-small)")

    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)

    print("\nNext steps:")
    print("  1. Use embeddings for semantic search")
    print("  2. Find relevant sentences by keyword similarity")
    print("  3. Extract entities from top-K relevant sentences")


if __name__ == "__main__":
    main()
