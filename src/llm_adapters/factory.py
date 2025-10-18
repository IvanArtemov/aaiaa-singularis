"""Factory for creating LLM llm_adapters"""

from typing import Optional
from .base_adapter import BaseLLMAdapter
from .openai_adapter import OpenAIAdapter
from .ollama_adapter import OllamaAdapter
from .nebius_adapter import NebiusAdapter
from ..config.settings import settings


class AdapterFactory:
    """Factory for creating LLM llm_adapters"""

    @staticmethod
    def create(provider: Optional[str] = None) -> BaseLLMAdapter:
        """
        Create adapter for specified provider

        Args:
            provider: "openai" or "ollama" (if None, uses config default)

        Returns:
            BaseLLMAdapter: Ready-to-use adapter

        Raises:
            ValueError: If provider is unknown
        """
        provider = provider or settings.active_provider
        config = settings.get_provider_config(provider)

        if provider == "openai":
            api_key = settings.get_api_key("openai")
            if not api_key:
                raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
            return OpenAIAdapter(config, api_key)

        elif provider == "nebius":
            api_key = settings.get_api_key("nebius")
            if not api_key:
                raise ValueError("Nebius API key not found. Set NEBIUS_API_KEY environment variable.")
            return NebiusAdapter(config, api_key)

        elif provider == "ollama":
            return OllamaAdapter(config)

        else:
            raise ValueError(f"Unknown provider: {provider}. Supported: openai, nebius, ollama")


def get_llm_adapter(provider: Optional[str] = None) -> BaseLLMAdapter:
    """
    Create and return LLM adapter

    Args:
        provider: "openai", "nebius", or "ollama" (if None, uses config default)

    Returns:
        BaseLLMAdapter: Ready-to-use adapter

    Example:
        >>> from llm_adapters import get_llm_adapter
        >>> llm = get_llm_adapter()  # Uses active_provider from config
        >>> result = llm.generate("Extract facts from...")
        >>> print(result["content"])
    """
    return AdapterFactory.create(provider)
