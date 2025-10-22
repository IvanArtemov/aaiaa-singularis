"""Nebius AI Studio adapter using OpenAI-compatible API"""

from openai import OpenAI
from typing import List, Dict, Any, Optional
from .base_adapter import BaseLLMAdapter


class NebiusAdapter(BaseLLMAdapter):
    """Adapter for Nebius AI Studio API using OpenAI-compatible SDK"""

    def __init__(self, config: Dict[str, Any], api_key: str):
        super().__init__(config)
        self.client = OpenAI(
            base_url="https://api.studio.nebius.com/v1/",
            api_key=api_key
        )
        self.chat_model = config["models"]["chat"]
        self.embedding_model = config["models"]["embeddings"]

    def generate(
            self,
            prompt: str,
            system_prompt: Optional[str] = None,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate text via Nebius Chat Completions API"""

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Log request
        if self.log_requests:
            self.logger.info(f"Nebius Chat Request - Model: {self.chat_model}")
            if system_prompt:
                self.logger.info(f"System prompt: {system_prompt}")
            self.logger.info(f"User prompt: {prompt}")

        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=messages,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.config.get("max_tokens", 2000),
            extra_body={"reasoning_enabled": False},
            reasoning_effort="low"
        )

        # Extract data
        content = response.choices[0].message.content
        usage = response.usage
        cost = self.calculate_cost(
            usage.prompt_tokens,
            usage.completion_tokens
        )

        # Log response
        if self.log_requests:
            self.logger.info(
                f"Nebius Chat Response - Tokens: {usage.prompt_tokens}/{usage.completion_tokens}, "
                f"Cost: ${cost:.6f}"
            )
            self.logger.info(f"Response: {content}")

        return {
            "content": content,
            "usage": {
                "input_tokens": usage.prompt_tokens,
                "output_tokens": usage.completion_tokens
            },
            "cost": cost
        }

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings via Nebius Embeddings API"""

        # Log request
        if self.log_requests:
            self.logger.info(f"Nebius Embeddings Request - Model: {self.embedding_model}, Texts: {len(texts)}")
            for i, text in enumerate(texts[:5], 1):  # Log first 5
                preview = text[:50] + "..." if len(text) > 50 else text
                self.logger.info(f"  [{i}] {preview}")
            if len(texts) > 5:
                self.logger.info(f"  ... and {len(texts) - 5} more texts")

        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=texts
        )

        # Extract embeddings
        embeddings = [item.embedding for item in response.data]
        return embeddings

    def stream_generate(
            self,
            prompt: str,
            system_prompt: Optional[str] = None
    ):
        """Stream text generation"""

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Log request
        if self.log_requests:
            self.logger.info(f"Nebius Stream Request - Model: {self.chat_model}")
            if system_prompt:
                self.logger.info(f"System prompt: {system_prompt}")
            self.logger.info(f"User prompt: {prompt}")

        stream = self.client.chat.completions.create(
            model=self.chat_model,
            messages=messages,
            temperature=self.temperature,
            stream=True
        )

        # Accumulate response for logging
        accumulated_response = []
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                accumulated_response.append(content)
                yield content

        # Log accumulated response
        if self.log_requests and accumulated_response:
            full_response = "".join(accumulated_response)
            self.logger.info(f"Nebius Stream Response: {full_response}")
