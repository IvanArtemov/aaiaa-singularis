"""
Example script demonstrating SciBERT embeddings

This script shows how to:
1. Load the SciBERT model from HuggingFace
2. Create embeddings for scientific texts
3. Calculate cosine similarity between texts
4. Compare performance with other embedders

SciBERT: https://huggingface.co/allenai/scibert_scivocab_uncased
- Pre-trained on scientific papers (1.14M papers from Semantic Scholar)
- Based on BERT-base architecture (768 dimensions)
- Optimized for scientific domain (biology, CS, etc.)
- FREE (local execution)
"""

import time
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


def mean_pooling(model_output, attention_mask):
    """
    Perform mean pooling on token embeddings to get sentence embedding

    Args:
        model_output: Output from the model (last_hidden_state)
        attention_mask: Attention mask to ignore padding tokens

    Returns:
        Mean-pooled sentence embedding
    """
    token_embeddings = model_output[0]  # First element contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def create_embeddings(texts, tokenizer, model):
    """
    Create embeddings for a list of texts using SciBERT

    Args:
        texts: List of text strings
        tokenizer: SciBERT tokenizer
        model: SciBERT model

    Returns:
        numpy array of embeddings (shape: [num_texts, 768])
    """
    # Tokenize sentences
    encoded_input = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings (optional but recommended for cosine similarity)
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

    return sentence_embeddings.cpu().numpy()


def main():
    print("=" * 80)
    print("SciBERT Embeddings Example")
    print("=" * 80)

    # ========== STEP 1: Load Model ==========
    print("\n" + "=" * 80)
    print("STEP 1: Loading SciBERT Model")
    print("=" * 80)

    model_name = "allenai/scibert_scivocab_uncased"
    print(f"\nModel: {model_name}")
    print("Downloading/loading model (first run may take a few minutes)...")

    start_time = time.time()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Set model to evaluation mode
    model.eval()

    load_time = time.time() - start_time

    print(f"\nModel loaded successfully in {load_time:.2f}s")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Embedding dimensions: 768 (BERT-base)")

    # ========== STEP 2: Prepare Scientific Texts ==========
    print("\n" + "=" * 80)
    print("STEP 2: Preparing Scientific Texts")
    print("=" * 80)

    # Scientific texts from different entity types
    scientific_texts = {
        "Hypothesis 1": "Aging is driven by accumulation of mitochondrial dysfunction and oxidative stress.",
        "Hypothesis 2": "Caloric restriction extends lifespan through activation of sirtuins and autophagy.",
        "Method 1": "We used RNA-seq to analyze gene expression profiles in aging tissues.",
        "Method 2": "Protein levels were quantified using Western blot and mass spectrometry.",
        "Result 1": "Treatment with rapamycin reduced inflammation markers by 30% in aged mice.",
        "Result 2": "Gene expression analysis revealed upregulation of DNA repair pathways.",
        "Conclusion 1": "Our findings suggest that targeting mitochondrial health may slow aging.",
        "Conclusion 2": "These results demonstrate the therapeutic potential of caloric restriction mimetics.",
    }

    print(f"\nPrepared {len(scientific_texts)} scientific texts:")
    for label, text in scientific_texts.items():
        preview = text[:60] + "..." if len(text) > 60 else text
        print(f"  {label}: {preview}")

    # ========== STEP 3: Create Embeddings ==========
    print("\n" + "=" * 80)
    print("STEP 3: Creating Embeddings")
    print("=" * 80)

    texts_list = list(scientific_texts.values())
    labels_list = list(scientific_texts.keys())

    print(f"\nProcessing {len(texts_list)} texts...")

    start_time = time.time()
    embeddings = create_embeddings(texts_list, tokenizer, model)
    embed_time = time.time() - start_time

    print(f"\nEmbeddings created in {embed_time:.2f}s")
    print(f"Throughput: {len(texts_list) / embed_time:.1f} texts/second")
    print(f"Shape: {embeddings.shape}")
    print(f"Embedding sample (first 5 dims): {embeddings[0][:5]}")

    # ========== STEP 4: Calculate Similarity ==========
    print("\n" + "=" * 80)
    print("STEP 4: Calculating Cosine Similarity")
    print("=" * 80)

    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(embeddings)

    print("\nSimilarity Matrix (top 4x4):")
    print("=" * 80)

    # Print header
    header = "".ljust(20)
    for i in range(min(4, len(labels_list))):
        header += labels_list[i][:12].ljust(14)
    print(header)
    print("-" * 80)

    # Print similarity scores
    for i in range(min(4, len(labels_list))):
        row = labels_list[i][:20].ljust(20)
        for j in range(min(4, len(labels_list))):
            row += f"{similarity_matrix[i][j]:.3f}".ljust(14)
        print(row)

    # ========== STEP 5: Find Most Similar Pairs ==========
    print("\n" + "=" * 80)
    print("STEP 5: Most Similar Text Pairs")
    print("=" * 80)

    # Find top 5 similar pairs (excluding self-similarity)
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

    print("\nTop 5 most similar pairs:")
    for i, (label1, label2, score) in enumerate(similarities[:5], 1):
        print(f"\n{i}. Similarity: {score:.4f}")
        print(f"   {label1}")
        print(f"   {label2}")
        print(f"   Texts:")
        print(f"     - {scientific_texts[label1][:70]}...")
        print(f"     - {scientific_texts[label2][:70]}...")

    # ========== STEP 6: Semantic Search Example ==========
    print("\n" + "=" * 80)
    print("STEP 6: Semantic Search Example")
    print("=" * 80)

    # Query text
    query = "How does mitochondrial dysfunction contribute to aging?"
    print(f"\nQuery: {query}")

    # Create query embedding
    query_embedding = create_embeddings([query], tokenizer, model)

    # Calculate similarity with all texts
    query_similarities = cosine_similarity(query_embedding, embeddings)[0]

    # Get top 3 results
    top_indices = np.argsort(query_similarities)[::-1][:3]

    print("\nTop 3 relevant texts:")
    for i, idx in enumerate(top_indices, 1):
        print(f"\n{i}. {labels_list[idx]} (similarity: {query_similarities[idx]:.4f})")
        print(f"   {scientific_texts[labels_list[idx]]}")

    # ========== STEP 7: Performance Metrics ==========
    print("\n" + "=" * 80)
    print("PERFORMANCE METRICS")
    print("=" * 80)

    print(f"\nModel loading time: {load_time:.2f}s")
    print(f"Embedding creation time: {embed_time:.2f}s")
    print(f"Throughput: {len(texts_list) / embed_time:.1f} texts/second")
    print(f"Average time per text: {embed_time / len(texts_list):.4f}s")

    # Memory usage
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    print(f"\nModel memory: {model_size_mb:.1f} MB")
    print(f"Embeddings memory: {embeddings.nbytes / 1024:.1f} KB")

    # ========== STEP 8: Cost Analysis ==========
    print("\n" + "=" * 80)
    print("COST ANALYSIS")
    print("=" * 80)

    print(f"\nSciBERT embeddings: $0.00 (FREE - local execution)")
    print(f"OpenAI embeddings: ~$0.00002 per text (text-embedding-3-small)")
    print(f"Nebius embeddings: ~$0.00001 per text (BAAI/bge-en-icl)")

    # Cost projection for 50M papers
    avg_sentences_per_paper = 200
    total_texts = 50_000_000 * avg_sentences_per_paper

    print(f"\nFor 50M papers with ~{avg_sentences_per_paper} sentences each:")
    print(f"  SciBERT: $0 (FREE)")
    print(f"  OpenAI: ${total_texts * 0.00000002:,.2f}")
    print(f"  Nebius: ${total_texts * 0.00000001:,.2f}")

    # ========== STEP 9: Comparison with Other Embedders ==========
    print("\n" + "=" * 80)
    print("COMPARISON: SciBERT vs Other Embedders")
    print("=" * 80)

    print("\n╔═══════════════╦════════════╦═══════════╦══════════════╦═══════════════════╗")
    print("║ Embedder      ║ Dimensions ║ Cost      ║ Quality      ║ Best For          ║")
    print("╠═══════════════╬════════════╬═══════════╬══════════════╬═══════════════════╣")
    print("║ SciBERT       ║ 768        ║ FREE      ║ ⭐⭐⭐⭐⭐    ║ Scientific papers ║")
    print("║ OpenAI        ║ 1536       ║ $0.02/1M  ║ ⭐⭐⭐⭐      ║ General text      ║")
    print("║ Nebius BGE    ║ 768        ║ $0.01/1M  ║ ⭐⭐⭐⭐      ║ General text      ║")
    print("║ Ollama bge-m3 ║ 1024       ║ FREE      ║ ⭐⭐⭐       ║ Local deployment  ║")
    print("╚═══════════════╩════════════╩═══════════╩══════════════╩═══════════════════╝")

    print("\n✅ RECOMMENDATIONS:")
    print("  1. Use SciBERT for scientific papers (FREE + domain-optimized)")
    print("  2. Use OpenAI for general text or when higher dimensions are needed")
    print("  3. Use Nebius for cost-efficient cloud embeddings")
    print("  4. Use Ollama for completely offline deployments")

    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)

    print("\nNext steps:")
    print("  1. Integrate SciBERT into SentenceEmbedder")
    print("  2. Create SciBertAdapter for consistent interface")
    print("  3. Compare embedding quality on real papers")
    print("  4. Measure impact on entity extraction accuracy")

    print("\nInstallation:")
    print("  pip install transformers torch scikit-learn")


if __name__ == "__main__":
    main()
