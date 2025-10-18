"""
Test Script: FACT Entity Extraction with Semantic Retrieval

This script tests only the retrieval phase of the Entity-Centric Pipeline:
1. Parse PDF with GROBID (ML-based structured extraction)
2. Generate sentence embeddings with Ollama (FREE)
3. Generate keywords specifically for FACT entities with OpenAI (~$0.001)
4. Semantic retrieval with ChromaDB (FREE) - NO LLM validation

Goal: Test the effectiveness of keyword-based semantic search for facts
without the full LLM validation step.

Total cost: ~$0.001 per paper (only keyword generation)
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parsers import GrobidParser
from src.extractors import SentenceEmbedder, EntityKeywordGenerator
from src.components import SemanticRetriever
from src.models import EntityType
from src.llm_adapters import get_llm_adapter

# Load environment variables
load_dotenv()


def check_prerequisites():
    """Check if all required services are available"""
    issues = []

    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        issues.append("❌ OPENAI_API_KEY not found in .env")
    else:
        print("✅ OpenAI API key found")

    # Check Ollama availability
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"✅ Ollama service running ({len(models)} models available)")
        else:
            issues.append("❌ Ollama service not responding")
    except Exception as e:
        issues.append(f"❌ Ollama not available: {e}")
        issues.append("   Run: ollama pull bge-m3")

    # Check GROBID service
    try:
        import requests
        response = requests.get("https://lfoppiano-grobid.hf.space/api/isalive", timeout=2)
        if response.status_code == 200:
            print("✅ GROBID service running")
        else:
            issues.append("❌ GROBID service not responding")
    except Exception:
        issues.append("❌ GROBID not available")
        issues.append("   Run: docker run --rm -p 8070:8070 lfoppiano/grobid:0.8.0")

    if issues:
        print("\n⚠️  Prerequisites not met:")
        for issue in issues:
            print(issue)
        print("\nPlease resolve these issues before running the test.")
        return False

    print("\n✅ All prerequisites met!\n")
    return True


def main():
    project_root = Path(__file__).parent.parent

    print("=" * 70)
    print("FACT Entity Extraction with Semantic Retrieval Test")
    print("=" * 70)
    print()

    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)

    # Configuration
    pdf_path = project_root / "docs" / "2508.05666v1.pdf"
    paper_id = "test_fact_extraction_001"

    # Check if PDF exists
    if not Path(pdf_path).exists():
        print(f"❌ PDF not found: {pdf_path}")
        print("Please provide a valid PDF path")
        sys.exit(1)

    print(f"📄 Input: {pdf_path}")
    print(f"🆔 Paper ID: {paper_id}")
    print()

    # Track overall metrics
    total_tokens = 0
    total_cost = 0.0
    start_time = time.time()

    # ========== STEP 1: Parse PDF with GROBID ==========

    print("=" * 70)
    print("STEP 1: Parsing PDF with GROBID")
    print("=" * 70)
    print()

    parser = GrobidParser(
        grobid_server="https://lfoppiano-grobid.hf.space",
        timeout=60
    )

    print("⚙️  Sending PDF to GROBID service...")
    parsed_doc = parser.parse(pdf_path)

    print(f"✅ Parsing complete!")
    print(f"   Title: {parsed_doc.title[:80]}...")

    if parsed_doc.imrad_sections:
        print(f"   IMRAD Sections:")
        for section_name, section_text in parsed_doc.imrad_sections.items():
            print(f"      • {section_name}: {len(section_text)} chars")

    print()

    # ========== STEP 2: Generate Sentence Embeddings ==========

    print("=" * 70)
    print("STEP 2: Generating Sentence Embeddings (Ollama - FREE)")
    print("=" * 70)
    print()

    embedder = SentenceEmbedder(
        llm_provider="ollama",
        batch_size=100,
        cache_size=256
    )

    print("⚙️  Creating sentence embeddings...")
    parsed_doc = embedder.process_document(parsed_doc)

    emb_metrics = embedder.get_metrics()
    total_tokens += emb_metrics["total_tokens"]
    total_cost += emb_metrics["total_cost_usd"]

    print(f"✅ Embeddings created!")
    print(f"   Sentences: {emb_metrics['total_sentences']}")
    print(f"   Tokens: {emb_metrics['total_tokens']:,}")
    print(f"   Cost: ${emb_metrics['total_cost_usd']:.6f}")
    print(f"   Time: {emb_metrics['embedding_time_seconds']:.2f}s")
    print()

    # Show sample sentences from different sections
    print("📝 Sample Sentences by Section:")
    sections_shown = set()
    for sentence in parsed_doc.sentences[:10]:
        if sentence.section not in sections_shown:
            print(f"   [{sentence.section}] {sentence.text[:80]}...")
            sections_shown.add(sentence.section)
    print()

    # ========== STEP 3: Generate Keywords for FACT Entities ==========

    print("=" * 70)
    print("STEP 3: Generating Keywords for FACT Entities (OpenAI)")
    print("=" * 70)
    print()

    keyword_generator = EntityKeywordGenerator(
        llm_provider="openai",
        llm_model="gpt-5-mini",
        cache_size=128
    )

    print("⚙️  Generating FACT-specific keywords...")

    # Generate keywords only for FACT entity type
    fact_keywords = keyword_generator.generate_keywords(
        parsed_doc=parsed_doc,
        entity_type=EntityType.FACT
    )

    kg_metrics = keyword_generator.get_metrics()
    total_tokens += kg_metrics["total_tokens"]
    total_cost += kg_metrics["total_cost_usd"]

    print(f"✅ Keywords generated!")
    print(f"   Keywords for FACT: {len(fact_keywords)}")
    print(f"   Tokens: {kg_metrics['total_tokens']:,}")
    print(f"   Cost: ${kg_metrics['total_cost_usd']:.6f}")
    print()

    # Display generated keywords
    print("🔑 Generated FACT Keywords:")
    for i, keyword in enumerate(fact_keywords[:15], 1):
        print(f"   {i}. {keyword}")
    if len(fact_keywords) > 15:
        print(f"   ... and {len(fact_keywords) - 15} more")
    print()

    # ========== STEP 4: Semantic Retrieval ==========

    print("=" * 70)
    print("STEP 4: Semantic Retrieval of FACT Candidates (ChromaDB - FREE)")
    print("=" * 70)
    print()

    retriever = SemanticRetriever(
        collection_name="test_fact_retrieval",
        persist_directory="./chroma_db_test",
        distance_metric="cosine"
    )

    # Clear any previous data
    retriever.clear_collection()

    print("⚙️  Indexing document sentences...")
    retriever.index_segments(parsed_doc.sentences, paper_id)
    print(f"✅ Indexed {len(parsed_doc.sentences)} sentences")
    print()

    print("⚙️  Retrieving FACT candidates...")

    # Get embeddings for keywords using Ollama
    embedding_adapter = get_llm_adapter("ollama")
    keyword_embeddings = embedding_adapter.embed(fact_keywords)

    # Retrieve candidates
    # Use top_k=30 to get more candidates for analysis
    candidates = retriever.retrieve_by_keywords(
        keywords=fact_keywords,
        keyword_embeddings=keyword_embeddings,
        entity_type=EntityType.FACT,
        top_k=30,
        section_filter=["introduction", "abstract", "discussion", "results", "discussion"],  # FACT-relevant sections
        paper_id=paper_id
    )

    retrieval_metrics = retriever.get_metrics()

    print(f"✅ Retrieval complete!")
    print(f"   Queries: {retrieval_metrics['total_queries']}")
    print(f"   Candidates retrieved: {len(candidates)}")
    print(f"   Avg results per query: {retrieval_metrics['avg_results_per_query']:.2f}")
    print()

    # ========== STEP 5: Display Results ==========

    print("=" * 70)
    print("RETRIEVAL RESULTS")
    print("=" * 70)
    print()

    if not candidates:
        print("⚠️  No candidates retrieved!")
    else:
        # Group candidates by section
        by_section = {}
        for candidate in candidates:
            section = candidate["metadata"]["section"]
            if section not in by_section:
                by_section[section] = []
            by_section[section].append(candidate)

        print("📊 Candidates by Section:")
        for section, cands in sorted(by_section.items()):
            print(f"   • {section}: {len(cands)} candidates")
        print()

        # Show top 15 candidates with details
        print("🔍 Top 15 FACT Candidates (by similarity):")
        print()
        for i, candidate in enumerate(candidates[:15], 1):
            text = candidate["text"]
            section = candidate["metadata"]["section"]
            distance = candidate["distance"]

            # Similarity score (1 - cosine distance)
            similarity = 1.0 - distance

            print(f"{i}. [Section: {section}] [Similarity: {similarity:.3f}]")
            print(f"   {text}")
            print()

    # ========== STEP 6: Overall Metrics ==========

    print("=" * 70)
    print("OVERALL METRICS")
    print("=" * 70)
    print()

    total_time = time.time() - start_time

    print(f"⏱️  Total time: {total_time:.2f}s")
    print(f"🪙  Total tokens: {total_tokens:,}")
    print(f"💰 Total cost: ${total_cost:.6f}")
    print()

    print("📈 Phase Breakdown:")
    print(f"   1. GROBID Parsing: FREE")
    print(f"   2. Embeddings (Ollama): ${emb_metrics['total_cost_usd']:.6f}")
    print(f"   3. Keyword Generation (OpenAI): ${kg_metrics['total_cost_usd']:.6f}")
    print(f"   4. Semantic Retrieval (ChromaDB): FREE")
    print()

    # ========== STEP 7: Save Results ==========

    print("=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    print()

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / f"{paper_id}_fact_retrieval.json"

    results_dict = {
        "paper_id": paper_id,
        "pdf_path": str(pdf_path),
        "title": parsed_doc.title,
        "keywords": {
            "entity_type": "FACT",
            "keywords": fact_keywords
        },
        "candidates": [
            {
                "rank": i + 1,
                "text": c["text"],
                "section": c["metadata"]["section"],
                "distance": c["distance"],
                "similarity": 1.0 - c["distance"],
                "position": c["metadata"]["position"]
            }
            for i, c in enumerate(candidates[:30])
        ],
        "metrics": {
            "total_time_seconds": total_time,
            "total_tokens": total_tokens,
            "total_cost_usd": total_cost,
            "phases": {
                "embeddings": {
                    "sentences": emb_metrics["total_sentences"],
                    "tokens": emb_metrics["total_tokens"],
                    "cost_usd": emb_metrics["total_cost_usd"],
                    "time_seconds": emb_metrics["embedding_time_seconds"]
                },
                "keyword_generation": {
                    "keywords_count": len(fact_keywords),
                    "tokens": kg_metrics["total_tokens"],
                    "cost_usd": kg_metrics["total_cost_usd"]
                },
                "semantic_retrieval": {
                    "queries": retrieval_metrics["total_queries"],
                    "candidates": len(candidates),
                    "avg_per_query": retrieval_metrics["avg_results_per_query"]
                }
            }
        }
    }

    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"✅ Results saved to: {output_path}")
    print()

    # Clean up
    retriever.clear_collection()

    print("=" * 70)
    print("✨ TEST COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
