"""LLM-based extraction pipeline using gpt-5-mini"""

import json
import time
from typing import Dict, Any, List, Optional

from .base_pipeline import BasePipeline
from src.models import (
    Entity,
    Relationship,
    EntityType,
    RelationshipType,
    ExtractionResult,
    PipelineMetrics
)
from src.llm_adapters.factory import get_llm_adapter


class LLMPipeline(BasePipeline):
    """
    LLM-based extraction pipeline

    Uses gpt-5-mini (or configurable LLM) to extract entities and relationships
    from scientific papers. Designed for high-quality ground truth generation.

    Estimated cost: ~$0.30 per paper with GPT-4, ~$0.03 with gpt-5-mini
    """

    # Extraction prompt template
    EXTRACTION_PROMPT = """You are an expert at extracting structured information from scientific papers.

Extract the following types of entities from the paper text:

1. **Facts**: Established knowledge or background information (from Introduction/Background)
2. **Hypotheses**: Scientific hypotheses or research questions being tested
3. **Experiments**: Experimental procedures or study designs
4. **Techniques/Methods**: Specific techniques, tools, or methodologies used
5. **Results**: Findings, data, observations, or measurements
6. **Datasets**: Data collections used or generated
7. **Analysis**: Statistical or computational analysis methods
8. **Conclusions**: Interpretations, conclusions, or implications

For each entity, extract:
- type: One of [fact, hypothesis, experiment, technique, result, dataset, analysis, conclusion]
- text: The actual text describing the entity (keep it concise but complete)
- confidence: Your confidence in this extraction (0.0-1.0)
- source_section: Which section it came from (abstract, introduction, methods, results, discussion, conclusion)

Also extract relationships between entities:
- source_id: ID of source entity
- target_id: ID of target entity
- relationship_type: Type of relationship (e.g., "fact_to_hypothesis", "hypothesis_to_method", "method_to_result", "result_to_conclusion")
- confidence: Confidence in this relationship (0.0-1.0)

Return ONLY valid JSON in this exact format:
{{
  "entities": [
    {{
      "id": "fact_001",
      "type": "fact",
      "text": "...",
      "confidence": 0.95,
      "source_section": "introduction"
    }}
  ],
  "relationships": [
    {{
      "source_id": "fact_001",
      "target_id": "hyp_001",
      "relationship_type": "fact_to_hypothesis",
      "confidence": 0.90
    }}
  ]
}}

Paper text:
---
{paper_text}
---

Extract ALL relevant entities and relationships. Be thorough but precise."""

    def __init__(
        self,
        config: Dict[str, Any],
        api_key: Optional[str] = None,
        model: str = "gpt-5-mini"
    ):
        """
        Initialize LLM pipeline

        Args:
            config: Configuration dictionary
            api_key: OpenAI API key (if not in config)
            model: Model to use (default: gpt-5-mini)
        """
        super().__init__(config)

        # Get LLM adapter
        self.llm_adapter = get_llm_adapter("openai")
        self.model = model
        self.temperature = config.get("temperature", 0.1)  # Low temp for deterministic output
        self.max_tokens = config.get("max_tokens", 4000)

    def extract(
        self,
        parsed_doc: "ParsedDocument",
        paper_id: str
    ) -> ExtractionResult:
        """
        Extract entities and relationships using LLM

        Args:
            parsed_doc: ParsedDocument with text, sections, and metadata
            paper_id: Unique paper identifier

        Returns:
            ExtractionResult with extracted entities and relationships
        """
        start_time = time.time()

        # Validate input
        self._validate_parsed_doc(parsed_doc)

        # Get text from parsed document
        paper_text = parsed_doc.text

        # Truncate if too long (gpt-5-mini context limit)
        max_chars = 100000  # ~25k tokens
        if len(paper_text) > max_chars:
            paper_text = paper_text[:max_chars] + "\n\n[Text truncated due to length]"

        # Build prompt
        prompt = self.EXTRACTION_PROMPT.format(paper_text=paper_text)

        # Call LLM
        try:
            response = self.llm_adapter.generate(
                prompt=prompt,
                system_prompt="You are a scientific information extraction expert. Return only valid JSON.",
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            content = response["content"]
            tokens_used = response["usage"]["input_tokens"] + response["usage"]["output_tokens"]
            cost = response["cost"]

        except Exception as e:
            raise RuntimeError(f"LLM extraction failed: {str(e)}")

        # Parse JSON response
        try:
            extracted_data = self._parse_llm_response(content)
        except Exception as e:
            raise RuntimeError(f"Failed to parse LLM response: {str(e)}\n\nResponse: {content}")

        # Convert to data models
        entities = self._build_entities(extracted_data["entities"])
        relationships = self._build_relationships(extracted_data["relationships"])

        # Group entities by type
        entities_by_type = {}
        for entity in entities:
            entity_type = entity.type.value
            if entity_type not in entities_by_type:
                entities_by_type[entity_type] = []
            entities_by_type[entity_type].append(entity)

        # Calculate metrics
        processing_time = time.time() - start_time
        metrics = PipelineMetrics(
            processing_time=processing_time,
            tokens_used=tokens_used,
            cost_usd=cost,
            entities_extracted=len(entities),
            relationships_extracted=len(relationships),
            metadata={
                "model": self.model,
                "temperature": self.temperature,
                "pipeline": self.name
            }
        )

        self.last_metrics = metrics

        # Build result
        result = ExtractionResult(
            paper_id=paper_id,
            entities=entities_by_type,
            relationships=relationships,
            metrics=metrics,
            metadata=parsed_doc.metadata
        )

        return result

    def _parse_llm_response(self, content: str) -> Dict[str, Any]:
        """
        Parse LLM JSON response

        Args:
            content: LLM response text

        Returns:
            Parsed dictionary with entities and relationships
        """
        # Try to find JSON in response
        content = content.strip()

        # Remove markdown code blocks if present
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
        except json.JSONDecodeError as e:
            # Try to extract JSON from text
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                raise ValueError(f"Could not parse JSON: {e}")

        # Validate structure
        if "entities" not in data:
            raise ValueError("Response missing 'entities' field")
        if "relationships" not in data:
            data["relationships"] = []  # Optional

        return data

    def _build_entities(self, entities_data: List[Dict]) -> List[Entity]:
        """
        Build Entity objects from parsed data

        Args:
            entities_data: List of entity dictionaries

        Returns:
            List of Entity objects
        """
        entities = []

        for entity_data in entities_data:
            try:
                # Map string type to EntityType enum
                entity_type_str = entity_data["type"].lower()

                # Handle technique/method synonyms
                if entity_type_str in ["method", "technique"]:
                    entity_type = EntityType.TECHNIQUE
                else:
                    entity_type = EntityType(entity_type_str)

                entity = Entity(
                    id=entity_data.get("id", self._generate_entity_id(entity_type_str)),
                    type=entity_type,
                    text=entity_data["text"],
                    confidence=entity_data.get("confidence", 1.0),
                    source_section=entity_data.get("source_section"),
                    metadata={}
                )
                entities.append(entity)

            except (KeyError, ValueError) as e:
                # Skip invalid entities
                print(f"Warning: Skipping invalid entity: {e}")
                continue

        return entities

    def _build_relationships(self, relationships_data: List[Dict]) -> List[Relationship]:
        """
        Build Relationship objects from parsed data

        Args:
            relationships_data: List of relationship dictionaries

        Returns:
            List of Relationship objects
        """
        relationships = []

        for rel_data in relationships_data:
            try:
                # Map string type to RelationshipType enum
                rel_type_str = rel_data["relationship_type"].lower()

                # Try to map to known relationship type
                try:
                    rel_type = RelationshipType(rel_type_str)
                except ValueError:
                    # Default to generic relationship
                    rel_type = RelationshipType.RELATED_TO

                relationship = Relationship(
                    source_id=rel_data["source_id"],
                    target_id=rel_data["target_id"],
                    relationship_type=rel_type,
                    confidence=rel_data.get("confidence", 1.0),
                    metadata={}
                )
                relationships.append(relationship)

            except (KeyError, ValueError) as e:
                # Skip invalid relationships
                print(f"Warning: Skipping invalid relationship: {e}")
                continue

        return relationships

    def get_description(self) -> str:
        """Get pipeline description"""
        return (
            f"LLM Pipeline using {self.model}. "
            f"High-quality extraction for ground truth generation. "
            f"Estimated cost: ${self.get_estimated_cost():.4f}/paper"
        )

    def get_estimated_cost(self) -> float:
        """Get estimated cost per paper"""
        if "gpt-4" in self.model and "mini" not in self.model:
            return 0.30  # GPT-4 full model
        elif "gpt-5-mini" in self.model:
            return 0.03  # gpt-5-mini
        else:
            return 0.05  # Other models
