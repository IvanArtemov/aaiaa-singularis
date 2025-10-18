"""
Example: Testing Nebius AI Studio Adapter

This script demonstrates:
1. Initializing Nebius LLM adapter
2. Generating text via chat completion
3. Creating embeddings
4. Displaying usage metrics and costs
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm_adapters import get_llm_adapter

# Load environment variables
load_dotenv()


def main():
    print("=" * 80)
    print("Nebius AI Studio Adapter Demo")
    print("=" * 80)
    print()

    # Initialize Nebius adapter
    print("⚙️  Initializing Nebius adapter...")
    try:
        llm = get_llm_adapter("nebius")
        print("✅ Nebius adapter initialized successfully!")
        print(f"   Chat model: {llm.chat_model}")
        print(f"   Embedding model: {llm.embedding_model}")
        print()
    except ValueError as e:
        print(f"❌ Error: {e}")
        print()
        print("Please set NEBIUS_API_KEY in your .env file")
        print("Get your API key from: https://studio.nebius.com/")
        sys.exit(1)

    # ========== Test 1: Chat Completion ==========
    print("=" * 80)
    print("TEST 1: Chat Completion")
    print("=" * 80)
    print()

    prompt = "Extract the main hypothesis from this text: 'We hypothesize that metformin extends lifespan by activating AMPK pathway in mice.'"
    system_prompt = "You are a scientific entity extraction assistant. Extract entities concisely."

    print(f"Prompt: {prompt}")
    print(f"System: {system_prompt}")
    print()
    print("🔄 Sending request to Nebius...")

    result = llm.generate(
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=0.1
    )

    print("✅ Response received!")
    print()
    print("Generated text:")
    print("-" * 80)
    print(result["content"])
    print("-" * 80)
    print()
    print("Usage metrics:")
    print(f"  • Input tokens: {result['usage']['input_tokens']}")
    print(f"  • Output tokens: {result['usage']['output_tokens']}")
    print(f"  • Cost: ${result['cost']:.6f}")
    print()

    # ========== Test 2: Embeddings ==========
    print("=" * 80)
    print("TEST 2: Embeddings")
    print("=" * 80)
    print()

    texts = [
        "Metformin extends lifespan in mice",
        "AMPK activation promotes longevity",
        "The experiment used 200 mg/kg daily dosage"
    ]

    print(f"Generating embeddings for {len(texts)} texts:")
    for i, text in enumerate(texts, 1):
        print(f"  {i}. {text}")
    print()
    print("🔄 Sending request to Nebius...")

    embeddings = llm.embed(texts)

    print("✅ Embeddings received!")
    print()
    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Embedding dimensions: {len(embeddings[0])}")
    print()
    print("First embedding (first 10 dimensions):")
    print(embeddings[0][:10])
    print()

    # ========== Summary ==========
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("✅ Nebius adapter is working correctly!")
    print()
    print("Configuration:")
    print(f"  • Provider: nebius")
    print(f"  • Chat model: {llm.chat_model}")
    print(f"  • Embedding model: {llm.embedding_model}")
    print(f"  • Base URL: {llm.base_url}")
    print()
    print("Cost breakdown:")
    print(f"  • Chat input: ${llm.config['costs']['chat_input']}/1M tokens")
    print(f"  • Chat output: ${llm.config['costs']['chat_output']}/1M tokens")
    print(f"  • Embeddings: ${llm.config['costs']['embeddings']}/1M tokens")
    print()
    print("To use Nebius as default provider:")
    print("  1. Edit src/config/llm_config.yaml")
    print("  2. Set active_provider: \"nebius\"")
    print()


if __name__ == "__main__":
    main()
