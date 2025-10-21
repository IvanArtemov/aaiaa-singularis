"""SciBERT adapter for scientific text embeddings"""

import time
import torch
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModel

from .base_embedding_adapter import BaseEmbeddingAdapter


class SciBertAdapter(BaseEmbeddingAdapter):
    """
    Adapter for SciBERT embeddings

    SciBERT: https://huggingface.co/allenai/scibert_scivocab_uncased
    - Pre-trained on 1.14M scientific papers from Semantic Scholar
    - Based on BERT-base architecture (768 dimensions)
    - Optimized for scientific domain (biology, CS, physics, etc.)
    - FREE (local execution, no API costs)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Model configuration
        self.model_name = config.get("model_name", "allenai/scibert_scivocab_uncased")
        self.max_length = config.get("max_length", 512)
        self.normalize = config.get("normalize", True)
        self.device = config.get("device", "cpu")

        # Load model and tokenizer
        self.logger.info(f"Loading SciBERT model: {self.model_name}")
        start_time = time.time()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)

        # Move model to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()

        load_time = time.time() - start_time
        self.logger.info(f"Model loaded in {load_time:.2f}s")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for texts using SciBERT

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors (shape: [num_texts, 768])
        """
        if not texts:
            return []

        start_time = time.time()

        # Tokenize sentences
        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Move to device
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform mean pooling
        sentence_embeddings = self._mean_pooling(
            model_output,
            encoded_input['attention_mask']
        )

        # Normalize embeddings (optional but recommended for cosine similarity)
        if self.normalize:
            sentence_embeddings = torch.nn.functional.normalize(
                sentence_embeddings,
                p=2,
                dim=1
            )

        # Convert to list
        embeddings = sentence_embeddings.cpu().numpy().tolist()

        # Track metrics
        elapsed_time = time.time() - start_time
        num_tokens = encoded_input['input_ids'].numel()
        self._track_embedding(len(texts), elapsed_time, num_tokens)

        return embeddings

    def _mean_pooling(self, model_output, attention_mask):
        """
        Perform mean pooling on token embeddings to get sentence embedding

        Args:
            model_output: Output from the model (last_hidden_state)
            attention_mask: Attention mask to ignore padding tokens

        Returns:
            Mean-pooled sentence embedding
        """
        token_embeddings = model_output[0]  # First element contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        # Sum embeddings and divide by number of valid tokens
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return sum_embeddings / sum_mask

    def get_embedding_dimension(self) -> int:
        """
        Get embedding vector dimension

        Returns:
            int: 768 (BERT-base dimension)
        """
        return 768

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the SciBERT model

        Returns:
            Dictionary with model information
        """
        return {
            "provider": "scibert",
            "model_name": self.model_name,
            "embedding_dim": self.get_embedding_dimension(),
            "max_length": self.max_length,
            "normalize": self.normalize,
            "device": self.device,
            "description": "Scientific BERT pre-trained on 1.14M research papers",
            "cost": "$0.00 (FREE - local execution)",
            "best_for": "Scientific text, research papers, biomedical literature"
        }
