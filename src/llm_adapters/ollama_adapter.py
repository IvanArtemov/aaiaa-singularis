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

        # Log request
        if self.log_requests:
            self.logger.info(f"Ollama Chat Request - Model: {self.chat_model}")
            if system_prompt:
                self.logger.info(f"System prompt: {system_prompt}")
            self.logger.info(f"User prompt: {prompt}")

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

        content = data["response"]
        input_tokens = data.get("prompt_eval_count", 0)
        output_tokens = data.get("eval_count", 0)

        # Log response
        if self.log_requests:
            self.logger.info(
                f"Ollama Chat Response - Tokens: {input_tokens}/{output_tokens}, "
                f"Cost: $0.00 (local)"
            )
            self.logger.info(f"Response: {content}")

        return {
            "content": content,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            },
            "cost": 0.0  # Local = free!
        }

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings via Ollama"""

        url = f"{self.base_url}/api/embeddings"
        embeddings = []

        # Log request
        if self.log_requests:
            self.logger.info(f"Ollama Embeddings Request - Model: {self.embedding_model}, Texts: {len(texts)}")
            for i, text in enumerate(texts[:5], 1):  # Log first 5
                preview = text[:50] + "..." if len(text) > 50 else text
                self.logger.info(f"  [{i}] {preview}")
            if len(texts) > 5:
                self.logger.info(f"  ... and {len(texts) - 5} more texts")

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

        # Log request
        if self.log_requests:
            self.logger.info(f"Ollama Stream Request - Model: {self.chat_model}")
            if system_prompt:
                self.logger.info(f"System prompt: {system_prompt}")
            self.logger.info(f"User prompt: {prompt}")

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

        # Accumulate response for logging
        accumulated_response = []
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if not data.get("done", False):
                    content = data.get("response", "")
                    accumulated_response.append(content)
                    yield content

        # Log accumulated response
        if self.log_requests and accumulated_response:
            full_response = "".join(accumulated_response)
            self.logger.info(f"Ollama Stream Response: {full_response}")
