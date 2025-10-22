"""Factory for creating embedding adapters"""

from typing import Optional, Dict, Any
from .base_embedding_adapter import BaseEmbeddingAdapter
from .scibert_adapter import SciBertAdapter


class EmbeddingAdapterFactory:
    """Factory for creating embedding adapters"""

    @staticmethod
    def create(provider: str, config: Optional[Dict[str, Any]] = None) -> BaseEmbeddingAdapter:
        """
        Create adapter for specified provider

        Args:
            provider: Provider name ("scibert", "openai", "nebius", etc.)
            config: Provider configuration (if None, will be loaded from settings)

        Returns:
            BaseEmbeddingAdapter: Ready-to-use embedding adapter

        Raises:
            ValueError: If provider is unknown
        """
        # If no config provided, try to load from settings
        if config is None:
            try:
                from ..config.settings import settings
                config = settings.get_embedding_config(provider)
            except Exception:
                # Fallback to default config
                config = {}

        if provider == "scibert":
            return SciBertAdapter(config)

        # Future providers can be added here:
        # elif provider == "openai":
        #     return OpenAIEmbeddingAdapter(config, api_key)
        # elif provider == "nebius":
        #     return NebiusEmbeddingAdapter(config, api_key)

        else:
            raise ValueError(
                f"Unknown embedding provider: {provider}. "
                f"Supported providers: scibert"
            )


def get_embedding_adapter(
    provider: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> BaseEmbeddingAdapter:
    """
    Create and return embedding adapter

    Args:
        provider: Provider name ("scibert", "openai", etc.)
                 If None, uses active_provider from config
        config: Provider configuration (optional)

    Returns:
        BaseEmbeddingAdapter: Ready-to-use embedding adapter

    Example:
        >>> from embedding_adapters import get_embedding_adapter
        >>> embedder = get_embedding_adapter("scibert")
        >>> embeddings = embedder.embed(["Text 1", "Text 2"])
        >>> print(len(embeddings))  # 2
        >>> print(len(embeddings[0]))  # 768
    """
    # Use default provider if not specified
    if provider is None:
        try:
            from ..config.settings import settings
            provider = settings.active_embedding_provider
        except Exception:
            # Fallback to scibert
            provider = "scibert"

    return EmbeddingAdapterFactory.create(provider, config)
