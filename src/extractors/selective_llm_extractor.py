"""Selective LLM extractor for complex cases (cost-optimized)"""

import json
import uuid
from typing import List, Optional, Dict, Any

from src.models import Entity, EntityType
from src.llm_adapters.factory import get_llm_adapter


class SelectiveLLMExtractor:
    """
    Selective LLM extractor for complex entity extraction

    Only used when pattern/NLP methods have low confidence.
    Optimized for cost by:
    - Short, focused prompts
    - Processing only specific sections
    - Extracting only one entity type at a time
    """

    # Focused prompts for each entity type
    PROMPTS = {
        EntityType.HYPOTHESIS: """Extract ONLY the scientific hypotheses from this text.
A hypothesis is a testable prediction or research question.

Text:
---
{text}
---

Return ONLY valid JSON array:
[
  {{"text": "hypothesis text", "confidence": 0.95}}
]

If no hypotheses found, return: []""",

        EntityType.CONCLUSION: """Extract ONLY the conclusions from this text.
A conclusion is an interpretation or implication drawn from results.

Text:
---
{text}
---

Return ONLY valid JSON array:
[
  {{"text": "conclusion text", "confidence": 0.95}}
]

If no conclusions found, return: []""",

        EntityType.ANALYSIS: """Extract ONLY the analysis methods from this text.
Analysis refers to statistical or computational methods applied to data.

Text:
---
{text}
---

Return ONLY valid JSON array:
[
  {{"text": "analysis method", "confidence": 0.95}}
]

If no analysis methods found, return: []"""
    }

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-5-mini",
        temperature: float = 0.1,
        max_tokens: int = 500,
        max_text_length: int = 3000
    ):
        """
        Initialize selective LLM extractor

        Args:
            provider: LLM provider (openai, ollama)
            model: Model name
            temperature: LLM temperature
            max_tokens: Maximum tokens in response
            max_text_length: Maximum text length (chars) to process
        """
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_text_length = max_text_length

        # Initialize LLM adapter
        self.llm_adapter = get_llm_adapter(provider)

        # Track cost
        self.total_cost = 0.0
        self.total_tokens = 0

    def extract(
        self,
        text: str,
        entity_type: EntityType,
        section_name: Optional[str] = None
    ) -> List[Entity]:
        """
        Extract entities of specific type using LLM

        Args:
            text: Text to extract from
            entity_type: Type of entity to extract
            section_name: Name of section (for metadata)

        Returns:
            List of extracted entities
        """
        # Check if we support this entity type
        if entity_type not in self.PROMPTS:
            raise ValueError(f"Entity type {entity_type} not supported by SelectiveLLMExtractor")

        # Truncate text if too long
        if len(text) > self.max_text_length:
            text = text[:self.max_text_length] + "\n\n[Text truncated]"

        # Build prompt
        prompt = self.PROMPTS[entity_type].format(text=text)

        # Call LLM
        try:
            response = self.llm_adapter.generate(
                prompt=prompt,
                system_prompt="You are a precise scientific information extractor. Return only valid JSON.",
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            content = response["content"]
            tokens_used = response["usage"]["input_tokens"] + response["usage"]["output_tokens"]
            cost = response["cost"]

            # Update metrics
            self.total_tokens += tokens_used
            self.total_cost += cost

        except Exception as e:
            print(f"Warning: LLM extraction failed: {e}")
            return []

        # Parse JSON response
        try:
            extracted_data = self._parse_response(content)
        except Exception as e:
            print(f"Warning: Failed to parse LLM response: {e}")
            return []

        # Convert to Entity objects
        entities = []
        for item in extracted_data:
            entity = Entity(
                id=self._generate_id(entity_type),
                type=entity_type,
                text=item["text"],
                confidence=item.get("confidence", 0.85),
                source_section=section_name,
                metadata={
                    "extraction_method": "selective_llm",
                    "llm_model": self.model,
                    "llm_cost": cost / max(len(extracted_data), 1)  # Cost per entity
                }
            )
            entities.append(entity)

        return entities

    def extract_batch(
        self,
        texts: List[str],
        entity_type: EntityType,
        section_names: Optional[List[str]] = None
    ) -> List[List[Entity]]:
        """
        Extract entities from multiple texts (batch processing)

        Args:
            texts: List of texts to extract from
            entity_type: Type of entity to extract
            section_names: List of section names (optional)

        Returns:
            List of entity lists (one per text)
        """
        if section_names is None:
            section_names = [None] * len(texts)

        results = []
        for text, section_name in zip(texts, section_names):
            entities = self.extract(text, entity_type, section_name)
            results.append(entities)

        return results

    def _parse_response(self, content: str) -> List[Dict[str, Any]]:
        """
        Parse LLM JSON response

        Args:
            content: LLM response text

        Returns:
            Parsed list of dictionaries
        """
        content = content.strip()

        # Remove markdown code blocks
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]

        content = content.strip()

        # Parse JSON
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from text
            import re
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                raise ValueError(f"Could not parse JSON from response")

        # Ensure it's a list
        if not isinstance(data, list):
            raise ValueError("Expected JSON array")

        return data

    def _generate_id(self, entity_type: EntityType) -> str:
        """Generate unique entity ID"""
        short_uuid = str(uuid.uuid4())[:8]
        return f"{entity_type.value}_{short_uuid}"

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get extraction metrics

        Returns:
            Dictionary with cost and token metrics
        """
        return {
            "total_cost_usd": self.total_cost,
            "total_tokens": self.total_tokens,
            "model": self.model,
            "provider": self.provider
        }

    def reset_metrics(self):
        """Reset cost tracking metrics"""
        self.total_cost = 0.0
        self.total_tokens = 0
