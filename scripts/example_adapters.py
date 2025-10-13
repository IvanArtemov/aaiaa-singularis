"""Example usage of LLM llm_adapters"""

from src.llm_adapters import get_llm_adapter


def main():
    """Demonstrate adapter usage"""

    print("=" * 60)
    print("LLM Adapter Example")
    print("=" * 60)

    # Get adapter (automatically uses active_provider from config)
    llm = get_llm_adapter()

    print(f"\nUsing provider: {llm.__class__.__name__}")
    print(f"Model: {llm.chat_model}")

    # Example 1: Simple generation
    print("\n" + "=" * 60)
    print("Example 1: Text Generation")
    print("=" * 60)

    result = llm.generate(
        prompt="What are the main causes of aging?",
        system_prompt="You are a scientific expert on aging research."
    )

    print(f"\nResponse: {result['content'][:200]}...")
    print(f"\nTokens used: {result['usage']['input_tokens']} input, {result['usage']['output_tokens']} output")
    print(f"Cost: ${result['cost']:.6f}")

    # Example 2: Embeddings
    print("\n" + "=" * 60)
    print("Example 2: Embeddings")
    print("=" * 60)

    texts = [
        "Caloric restriction extends lifespan",
        "Exercise improves health",
        "The sky is blue"
    ]

    embeddings = llm.embed(texts)

    print(f"\nCreated {len(embeddings)} embeddings")
    print(f"Embedding dimension: {len(embeddings[0])}")
    print(f"First embedding (first 5 values): {embeddings[0][:5]}")

    # Example 3: Streaming
    print("\n" + "=" * 60)
    print("Example 3: Streaming Generation")
    print("=" * 60)
    print("\nStreaming response:")

    for chunk in llm.stream_generate(
        prompt="List 3 key interventions for healthy aging.",
        system_prompt="You are a longevity expert."
    ):
        print(chunk, end="", flush=True)

    print("\n\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
