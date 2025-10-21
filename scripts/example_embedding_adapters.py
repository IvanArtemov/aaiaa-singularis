"""
Example script demonstrating the Embedding Adapters architecture

This script shows how to:
1. Use the factory pattern to create embedding adapters
2. Create embeddings with SciBERT
3. Calculate cosine similarity
4. Access model info and metrics
5. Compare with other embedding providers

Usage:
    python scripts/example_embedding_adapters.py
"""

import sys
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.embedding_adapters import get_embedding_adapter


def main():
    print("=" * 80)
    print("Embedding Adapters Example")
    print("=" * 80)

    # ========== STEP 1: Create Embedding Adapter ==========
    print("\n" + "=" * 80)
    print("STEP 1: Creating SciBERT Embedding Adapter")
    print("=" * 80)

    # Create adapter using factory (uses active_provider from config)
    print("\nLoading SciBERT adapter...")
    embedder = get_embedding_adapter("scibert")

    # Display model info
    model_info = embedder.get_model_info()
    print("\nModel Information:")
    print(f"  Provider: {model_info['provider']}")
    print(f"  Model: {model_info['model_name']}")
    print(f"  Embedding dimension: {model_info['embedding_dim']}")
    print(f"  Max length: {model_info['max_length']} tokens")
    print(f"  Description: {model_info['description']}")
    print(f"  Cost: {model_info['cost']}")
    print(f"  Best for: {model_info['best_for']}")

    # ========== STEP 2: Create Embeddings ==========
    print("\n" + "=" * 80)
    print("STEP 2: Creating Embeddings for Scientific Texts")
    print("=" * 80)

    # Scientific texts from different entity types
    scientific_texts = {
        "Hypothesis 1": "Aging is driven by accumulation of mitochondrial dysfunction and oxidative stress.",
        "Hypothesis 2": "Caloric restriction extends lifespan through activation of sirtuins and autophagy.",
        "Method 1": "We used RNA-seq to analyze gene expression profiles in aging tissues.",
        "Method 2": "Protein levels were quantified using Western blot and mass spectrometry.",
        "Result 1": "Treatment with rapamycin reduced inflammation markers by 30% in aged mice.",
        "Result 2": "Gene expression analysis revealed upregulation of DNA repair pathways.",
    }

    print(f"\nProcessing {len(scientific_texts)} texts...")
    for label, text in list(scientific_texts.items())[:2]:
        preview = text[:60] + "..." if len(text) > 60 else text
        print(f"  {label}: {preview}")
    print(f"  ... and {len(scientific_texts) - 2} more")

    # Create embeddings
    texts_list = list(scientific_texts.values())
    labels_list = list(scientific_texts.keys())

    print("\nCreating embeddings...")
    embeddings = embedder.embed(texts_list)

    print(f"\nEmbeddings created!")
    print(f"  Shape: {len(embeddings)} x {len(embeddings[0])}")
    print(f"  First embedding (first 5 dims): {embeddings[0][:5]}")

    # ========== STEP 3: Calculate Similarity ==========
    print("\n" + "=" * 80)
    print("STEP 3: Calculating Cosine Similarity")
    print("=" * 80)

    # Convert to numpy array for sklearn
    embeddings_array = np.array(embeddings)

    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(embeddings_array)

    print("\nSimilarity Matrix:")
    print("=" * 80)

    # Print header
    header = "".ljust(15)
    for i in range(min(4, len(labels_list))):
        header += labels_list[i][:12].ljust(14)
    print(header)
    print("-" * 80)

    # Print similarity scores
    for i in range(min(4, len(labels_list))):
        row = labels_list[i][:15].ljust(15)
        for j in range(min(4, len(labels_list))):
            row += f"{similarity_matrix[i][j]:.3f}".ljust(14)
        print(row)

    # ========== STEP 4: Find Most Similar Pairs ==========
    print("\n" + "=" * 80)
    print("STEP 4: Most Similar Text Pairs")
    print("=" * 80)

    # Find top 3 similar pairs (excluding self-similarity)
    similarities = []
    for i in range(len(labels_list)):
        for j in range(i + 1, len(labels_list)):
            similarities.append((
                labels_list[i],
                labels_list[j],
                similarity_matrix[i][j]
            ))

    # Sort by similarity
    similarities.sort(key=lambda x: x[2], reverse=True)

    print("\nTop 3 most similar pairs:")
    for i, (label1, label2, score) in enumerate(similarities[:3], 1):
        print(f"\n{i}. Similarity: {score:.4f}")
        print(f"   {label1} <-> {label2}")
        print(f"     - {scientific_texts[label1][:60]}...")
        print(f"     - {scientific_texts[label2][:60]}...")

    # ========== STEP 5: Semantic Search ==========
    print("\n" + "=" * 80)
    print("STEP 5: Semantic Search Example")
    print("=" * 80)

    # Query text
    query = "How does mitochondrial dysfunction contribute to aging?"
    print(f"\nQuery: {query}")

    # Create query embedding
    query_embedding = embedder.embed([query])

    # Calculate similarity with all texts
    query_similarities = cosine_similarity(query_embedding, embeddings_array)[0]

    # Get top 3 results
    top_indices = np.argsort(query_similarities)[::-1][:3]

    print("\nTop 3 relevant texts:")
    for i, idx in enumerate(top_indices, 1):
        print(f"\n{i}. {labels_list[idx]} (similarity: {query_similarities[idx]:.4f})")
        print(f"   {scientific_texts[labels_list[idx]]}")

    # ========== STEP 6: Performance Metrics ==========
    print("\n" + "=" * 80)
    print("PERFORMANCE METRICS")
    print("=" * 80)

    metrics = embedder.get_metrics()

    print(f"\nTotal embeddings created: {metrics['total_embeddings']}")
    print(f"Total time: {metrics['total_time_seconds']:.2f}s")
    print(f"Average time per text: {metrics['average_time_per_text']:.4f}s")
    print(f"Throughput: {metrics['texts_per_second']:.1f} texts/second")
    print(f"Total tokens processed: {metrics['total_tokens']:,}")

    # ========== STEP 7: Batch Processing ==========
    print("\n" + "=" * 80)
    print("STEP 7: Batch Processing Example")
    print("=" * 80)

    # Create more texts
    batch_texts = [f"Scientific text number {i}" for i in range(20)]

    print(f"\nProcessing {len(batch_texts)} texts in batches...")
    batch_embeddings = embedder.embed_batch(
        batch_texts,
        batch_size=8,
        show_progress=False
    )

    print(f"\nBatch embeddings created: {len(batch_embeddings)} x {len(batch_embeddings[0])}")

    # ========== STEP 8: Cost Comparison ==========
    print("\n" + "=" * 80)
    print("COST COMPARISON")
    print("=" * 80)

    print("\n╔═══════════════╦════════════╦═══════════╦══════════════╦═══════════════════╗")
    print("║ Embedder      ║ Dimensions ║ Cost      ║ Quality      ║ Best For          ║")
    print("╠═══════════════╬════════════╬═══════════╬══════════════╬═══════════════════╣")
    print("║ SciBERT       ║ 768        ║ FREE      ║ ⭐⭐⭐⭐⭐    ║ Scientific papers ║")
    print("║ OpenAI        ║ 1536       ║ $0.02/1M  ║ ⭐⭐⭐⭐      ║ General text      ║")
    print("║ Nebius BGE    ║ 768        ║ $0.01/1M  ║ ⭐⭐⭐⭐      ║ General text      ║")
    print("║ Ollama bge-m3 ║ 1024       ║ FREE      ║ ⭐⭐⭐       ║ Local deployment  ║")
    print("╚═══════════════╩════════════╩═══════════╩══════════════╩═══════════════════╝")

    # Cost projection for 50M papers
    avg_sentences_per_paper = 200
    total_texts = 50_000_000 * avg_sentences_per_paper

    print(f"\nFor 50M papers with ~{avg_sentences_per_paper} sentences each:")
    print(f"  Total texts: {total_texts:,}")
    print(f"  SciBERT: $0 (FREE)")
    print(f"  OpenAI: ${total_texts * 0.00000002:,.2f}")
    print(f"  Nebius: ${total_texts * 0.00000001:,.2f}")

    print("\n✅ RECOMMENDATION: Use SciBERT for scientific papers (FREE + domain-optimized)")

    # ========== STEP 9: Architecture Overview ==========
    print("\n" + "=" * 80)
    print("ARCHITECTURE OVERVIEW")
    print("=" * 80)

    print("\nEmbedding Adapters Architecture:")
    print("  ├── BaseEmbeddingAdapter (abstract)")
    print("  │   ├── embed(texts) -> embeddings")
    print("  │   ├── get_embedding_dimension() -> int")
    print("  │   ├── get_model_info() -> dict")
    print("  │   └── get_metrics() -> dict")
    print("  │")
    print("  ├── SciBertAdapter (implemented)")
    print("  │   └── allenai/scibert_scivocab_uncased")
    print("  │")
    print("  ├── OpenAIEmbeddingAdapter (future)")
    print("  │   └── text-embedding-3-small")
    print("  │")
    print("  └── NebiusEmbeddingAdapter (future)")
    print("      └── BAAI/bge-en-icl")

    print("\nFactory Pattern:")
    print("  get_embedding_adapter(provider) -> BaseEmbeddingAdapter")

    print("\nConfiguration:")
    print("  src/config/embedding_config.yaml")
    print("  - Switch providers via 'active_provider'")
    print("  - Configure model parameters")
    print("  - Manage batch sizes and device settings")

    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)

    print("\nNext steps:")
    print("  1. Integrate into Entity-Centric Pipeline")
    print("  2. Replace SentenceEmbedder with SciBertAdapter")
    print("  3. Measure impact on extraction quality")
    print("  4. Add support for GPU acceleration (device='cuda')")

    print("\nUsage in your code:")
    print("  from src.embedding_adapters import get_embedding_adapter")
    print("  embedder = get_embedding_adapter('scibert')")
    print("  embeddings = embedder.embed(['text1', 'text2'])")


if __name__ == "__main__":
    main()
