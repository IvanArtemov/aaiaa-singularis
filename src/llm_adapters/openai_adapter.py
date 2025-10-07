"""OpenAI adapter using requests"""

import requests
import json
from typing import List, Dict, Any, Optional
from .base_adapter import BaseLLMAdapter


class OpenAIAdapter(BaseLLMAdapter):
    """Adapter for OpenAI API using requests"""

    def __init__(self, config: Dict[str, Any], api_key: str):
        super().__init__(config)
        self.api_key = api_key
        self.chat_model = config["models"]["chat"]
        self.embedding_model = config["models"]["embeddings"]

    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for OpenAI API"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate text via OpenAI Chat Completions API"""

        url = f"{self.base_url}/chat/completions"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.chat_model,
            "messages": messages,
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.config.get("max_tokens", 2000)
        }

        response = requests.post(
            url,
            headers=self._get_headers(),
            json=payload,
            timeout=self.timeout
        )

        response.raise_for_status()
        data = response.json()

        # Extract data
        content = data["choices"][0]["message"]["content"]
        usage = data["usage"]

        return {
            "content": content,
            "usage": {
                "input_tokens": usage["prompt_tokens"],
                "output_tokens": usage["completion_tokens"]
            },
            "cost": self.calculate_cost(
                usage["prompt_tokens"],
                usage["completion_tokens"]
            )
        }

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings via OpenAI Embeddings API"""

        url = f"{self.base_url}/embeddings"

        payload = {
            "model": self.embedding_model,
            "input": texts
        }

        response = requests.post(
            url,
            headers=self._get_headers(),
            json=payload,
            timeout=self.timeout
        )

        response.raise_for_status()
        data = response.json()

        # Extract embeddings
        embeddings = [item["embedding"] for item in data["data"]]
        return embeddings

    def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ):
        """Stream text generation"""

        url = f"{self.base_url}/chat/completions"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.chat_model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": True
        }

        response = requests.post(
            url,
            headers=self._get_headers(),
            json=payload,
            stream=True,
            timeout=self.timeout
        )

        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data_str = line[6:]
                    if data_str == '[DONE]':
                        break
                    try:
                        data = json.loads(data_str)
                        if 'choices' in data and len(data['choices']) > 0:
                            delta = data['choices'][0].get('delta', {})
                            content = delta.get('content', '')
                            if content:
                                yield content
                    except json.JSONDecodeError:
                        continue
