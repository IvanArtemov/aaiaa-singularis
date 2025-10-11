"""OpenAI adapter using official OpenAI SDK"""

from openai import OpenAI
from typing import List, Dict, Any, Optional
from .base_adapter import BaseLLMAdapter


class OpenAIAdapter(BaseLLMAdapter):
    """Adapter for OpenAI API using official SDK"""

    def __init__(self, config: Dict[str, Any], api_key: str):
        super().__init__(config)
        self.client = OpenAI(api_key=api_key)
        self.chat_model = config["models"]["chat"]
        self.embedding_model = config["models"]["embeddings"]

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate text via OpenAI Chat Completions API"""

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=messages,
        )

        # Extract data
        content = response.choices[0].message.content
        usage = response.usage

        return {
            "content": content,
            "usage": {
                "input_tokens": usage.prompt_tokens,
                "output_tokens": usage.completion_tokens
            },
            "cost": self.calculate_cost(
                usage.prompt_tokens,
                usage.completion_tokens
            )
        }

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings via OpenAI Embeddings API"""

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

        stream = self.client.chat.completions.create(
            model=self.chat_model,
            messages=messages,
            temperature=self.temperature,
            stream=True
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
