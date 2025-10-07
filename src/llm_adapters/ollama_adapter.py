"""Ollama adapter using requests"""

import requests
import json
from typing import List, Dict, Any, Optional
from .base_adapter import BaseLLMAdapter


class OllamaAdapter(BaseLLMAdapter):
    """Adapter for Ollama (local) using requests"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.chat_model = config["models"]["chat"]
        self.embedding_model = config["models"]["embeddings"]

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate text via Ollama API"""

        url = f"{self.base_url}/api/generate"

        # Combine system and user prompts
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        payload = {
            "model": self.chat_model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": temperature or self.temperature
            }
        }

        response = requests.post(
            url,
            json=payload,
            timeout=self.timeout
        )

        response.raise_for_status()
        data = response.json()

        return {
            "content": data["response"],
            "usage": {
                "input_tokens": data.get("prompt_eval_count", 0),
                "output_tokens": data.get("eval_count", 0)
            },
            "cost": 0.0  # Local = free!
        }

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings via Ollama"""

        url = f"{self.base_url}/api/embeddings"
        embeddings = []

        for text in texts:
            payload = {
                "model": self.embedding_model,
                "prompt": text
            }

            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout
            )

            response.raise_for_status()
            data = response.json()
            embeddings.append(data["embedding"])

        return embeddings

    def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ):
        """Stream text generation"""

        url = f"{self.base_url}/api/generate"

        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        payload = {
            "model": self.chat_model,
            "prompt": full_prompt,
            "stream": True,
            "options": {
                "temperature": self.temperature
            }
        }

        response = requests.post(
            url,
            json=payload,
            stream=True,
            timeout=self.timeout
        )

        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if not data.get("done", False):
                    yield data.get("response", "")
