"""Base adapter for all embedding providers"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import time


class BaseEmbeddingAdapter(ABC):
    """Abstract base class for all embedding adapters"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Performance metrics
        self._total_embeddings = 0
        self._total_time = 0.0
        self._total_tokens = 0

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for texts

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
            [[0.1, 0.2, ...], [0.3, 0.4, ...]]
        """
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """
        Get embedding vector dimension

        Returns:
            int: Dimension of embedding vectors (e.g., 768 for SciBERT)
        """
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the embedding model

        Returns:
            Dictionary with model information:
            {
                "provider": "scibert",
                "model_name": "allenai/scibert_scivocab_uncased",
                "embedding_dim": 768,
                "max_length": 512,
                "description": "Scientific BERT for research papers"
            }
        """
        return {
            "provider": self.__class__.__name__,
            "config": self.config
        }

    def embed_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = False
    ) -> List[List[float]]:
        """
        Create embeddings with batching support

        Args:
            texts: List of texts to embed
            batch_size: Batch size (if None, uses config default)
            show_progress: Print progress information

        Returns:
            List of embedding vectors
        """
        batch_size = batch_size or self.config.get("batch_size", 32)
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            if show_progress:
                self.logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")

            batch_embeddings = self.embed(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics

        Returns:
            Dictionary with metrics:
            {
                "total_embeddings": 1000,
                "total_time_seconds": 10.5,
                "average_time_per_text": 0.0105,
                "texts_per_second": 95.2,
                "total_tokens": 50000
            }
        """
        return {
            "total_embeddings": self._total_embeddings,
            "total_time_seconds": self._total_time,
            "average_time_per_text": self._total_time / max(1, self._total_embeddings),
            "texts_per_second": self._total_embeddings / max(0.001, self._total_time),
            "total_tokens": self._total_tokens
        }

    def reset_metrics(self):
        """Reset performance metrics"""
        self._total_embeddings = 0
        self._total_time = 0.0
        self._total_tokens = 0

    def _track_embedding(self, num_texts: int, elapsed_time: float, num_tokens: int = 0):
        """
        Track embedding metrics

        Args:
            num_texts: Number of texts embedded
            elapsed_time: Time taken in seconds
            num_tokens: Number of tokens processed
        """
        self._total_embeddings += num_texts
        self._total_time += elapsed_time
        self._total_tokens += num_tokens
