"""Base adapter for all LLM providers"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseLLMAdapter(ABC):
    """Abstract base class for all LLM llm_adapters"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_url = config.get("base_url", "")
        self.temperature = config.get("temperature", 0.1)
        self.timeout = config.get("timeout", 30)
        self.log_requests = config.get("log_requests", True)
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate text response

        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            temperature: Sampling temperature (optional)
            max_tokens: Maximum tokens to generate (optional)

        Returns:
            {
                "content": "Generated text",
                "usage": {"input_tokens": 100, "output_tokens": 50},
                "cost": 0.001
            }
        """
        pass

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
    def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ):
        """
        Stream text generation

        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)

        Yields:
            str: Chunks of generated text
        """
        pass

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate request cost

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            float: Cost in USD
        """
        costs = self.config.get("costs", {})
        input_cost = (input_tokens / 1_000_000) * costs.get("chat_input", 0)
        output_cost = (output_tokens / 1_000_000) * costs.get("chat_output", 0)
        return input_cost + output_cost
