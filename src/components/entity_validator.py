"""LLM-based entity validator for candidate validation"""

import json
import uuid
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.models import Entity, EntityType, EntitySchema
from src.llm_adapters import BaseLLMAdapter


class EntityValidator:
    """
    Validates entity candidates using lightweight LLM calls

    Uses batch validation (10 candidates at a time) to minimize cost.
    Supports parallel validation across different entity types.

    Cost: ~$0.015 per paper (20 batches × gpt-5-mini)
    """

    def __init__(
        self,
        llm_adapter: BaseLLMAdapter,
        batch_size: int = 10,
        temperature: float = 0.1,
        max_tokens: int = 1000
    ):
        """
        Initialize entity validator

        Args:
            llm_adapter: LLM adapter for validation
            batch_size: Number of candidates to validate per batch
            temperature: LLM temperature (lower = more consistent)
            max_tokens: Max tokens per response
        """
        self.llm = llm_adapter
        self.batch_size = batch_size
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Metrics
        self.total_candidates = 0
        self.total_validated = 0
        self.total_rejected = 0
        self.total_tokens = 0
        self.total_cost = 0.0

    def validate_batch(
        self,
        candidates: List[Dict[str, Any]],
        entity_type: EntityType,
        entity_schema: EntitySchema,
        confidence_threshold: float = 0.7
    ) -> List[Entity]:
        """
        Validate a batch of candidates using LLM

        Args:
            candidates: List of candidate dicts (text, metadata, etc.)
            entity_type: Type of entity being validated
            entity_schema: Schema with description and patterns
            confidence_threshold: Minimum confidence to accept

        Returns:
            List of validated Entity objects (only those above threshold)
        """
        if not candidates:
            return []

        self.total_candidates += len(candidates)

        # Build prompt for batch validation
        prompt = self._build_batch_prompt(
            candidates=candidates,
            entity_type=entity_type,
            entity_schema=entity_schema
        )

        try:
            # Call LLM
            response = self.llm.generate(
                prompt=prompt,
                system_prompt=self._get_system_prompt(),
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            # Track metrics
            self.total_tokens += (
                response["usage"]["input_tokens"] +
                response["usage"]["output_tokens"]
            )
            self.total_cost += response["cost"]

            # Parse validation results
            validated_entities = self._parse_validation_response(
                response["content"],
                candidates=candidates,
                entity_type=entity_type,
                confidence_threshold=confidence_threshold
            )

            self.total_validated += len(validated_entities)
            self.total_rejected += len(candidates) - len(validated_entities)

            return validated_entities

        except Exception as e:
            print(f"Warning: Batch validation failed for {entity_type}: {e}")
            return []

    def validate_parallel(
        self,
        candidates_by_type: Dict[EntityType, List[Dict[str, Any]]],
        entity_schemas: Dict[EntityType, EntitySchema],
        confidence_threshold: float | Dict[str, float] = 0.7,
        max_workers: int = 4
    ) -> Dict[EntityType, List[Entity]]:
        """
        Validate candidates for all entity types in parallel

        Args:
            candidates_by_type: Dict mapping entity types to candidates
            entity_schemas: Dict mapping entity types to schemas
            confidence_threshold: Minimum confidence threshold (float or dict of thresholds by entity type)
            max_workers: Number of parallel threads

        Returns:
            Dict mapping entity types to validated entities
        """
        validated_by_type = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}

            # Submit batch validation tasks for each entity type
            for entity_type, candidates in candidates_by_type.items():
                if not candidates:
                    continue

                schema = entity_schemas.get(entity_type)
                if not schema:
                    print(f"Warning: No schema for {entity_type}, skipping")
                    continue

                # Get threshold for this entity type
                if isinstance(confidence_threshold, dict):
                    threshold = confidence_threshold.get(entity_type.value.upper(), 0.7)
                else:
                    threshold = confidence_threshold

                # Split into batches
                batches = [
                    candidates[i:i + self.batch_size]
                    for i in range(0, len(candidates), self.batch_size)
                ]

                # Submit each batch
                for batch in batches:
                    future = executor.submit(
                        self.validate_batch,
                        batch,
                        entity_type,
                        schema,
                        threshold
                    )
                    futures[future] = entity_type

            # Collect results
            for future in as_completed(futures):
                entity_type = futures[future]
                try:
                    validated = future.result()
                    if entity_type not in validated_by_type:
                        validated_by_type[entity_type] = []
                    validated_by_type[entity_type].extend(validated)
                except Exception as e:
                    print(f"Warning: Parallel validation failed for {entity_type}: {e}")

        return validated_by_type

    def _get_system_prompt(self) -> str:
        """Get system prompt for validation"""
        return (
            "You are a scientific entity validation expert. "
            "Your task is to determine if text fragments are valid instances of "
            "specific scientific entity types. "
            "Return ONLY a JSON array, no explanations."
        )

    def _build_batch_prompt(
        self,
        candidates: List[Dict[str, Any]],
        entity_type: EntityType,
        entity_schema: EntitySchema
    ) -> str:
        """
        Build prompt for batch validation

        Args:
            candidates: List of candidates
            entity_type: Entity type
            entity_schema: Schema with description

        Returns:
            Formatted prompt string
        """
        # Build candidate list
        candidate_texts = []
        for i, candidate in enumerate(candidates):
            text = candidate.get("text", "")
            candidate_texts.append(f"[{i+1}] {text[:300]}")  # Limit to 300 chars

        candidates_str = "\n".join(candidate_texts)

        # Build additional guidelines for specific entity types
        type_specific_guidelines = ""

        if entity_type == EntityType.METHOD:
            type_specific_guidelines = """

⚠️ METHOD VALIDATION CRITERIA (3 levels):

✅ HIGHLY VALID (confidence 0.85+) - Full procedural details:
- "RT-PCR was performed using SuperScript III (Invitrogen) with 200ng RNA at 42°C"
- "Cells were cultured in DMEM with 10% FBS at 37°C for 24 hours"
- Contains: parameters, concentrations, temperatures, durations, versions

✅ VALID (confidence 0.70-0.85) - Method with context:
- "We performed scRNA-seq on sorted CD4+ T cells from intestinal lamina propria"
- "ATAC-seq analysis was performed to assess chromatin accessibility"
- "Flow cytometry was used to analyze cell populations"
- Has: method name + biological context + sample type

❌ INVALID (confidence < 0.65) - Only if too vague:
- "Standard methods were used" (no specifics)
- "Data were processed" (what method?)
- Bare tool names without context: "PCR", "microscopy"
"""

        elif entity_type == EntityType.EXPERIMENT:
            type_specific_guidelines = """

⚠️ EXPERIMENT VALIDATION CRITERIA:

✅ VALID EXPERIMENT includes:
- Experimental manipulation: "CRISPR-Cas9 knockout of OCR369"
- Comparative study: "Control vs Rorc Δ369 mice were compared"
- Intervention + observation: "OCR369 deletion resulted in reduced APCs"
- Experimental setup: "Cells were sorted and subjected to scRNA-seq analysis"

Can be concise if experimental design is clear.
Experiments describe WHAT WAS DONE to test hypothesis, not just observations.
"""

        elif entity_type == EntityType.DATASET:
            type_specific_guidelines = """

⚠️ DATASET VALIDATION CRITERIA:

✅ VALID DATASET includes:
- Public repository IDs: "GSM3638386", "GSE137319 from GEO database"
- Generated datasets: "scRNA-seq data from sorted cells", "ATAC-seq datasets"
- Referenced collections: "published datasets", "integrated with existing data"

Dataset MENTIONS are sufficient - full descriptions not required.
Look for data source identifiers or data collection references.
"""

        prompt = f"""You are validating scientific entities in a research paper.

Entity Type: {entity_type.value.upper()}
Description: {entity_schema.description}

For each text fragment below, determine:
1. is_valid: Is this a valid {entity_type.value}? (true/false)
2. confidence: Confidence score (0.0-1.0)
3. core_text: Extract the core entity statement (1-2 sentences max, or original if already concise)

Text Fragments:
{candidates_str}

Return ONLY a JSON array with this structure:
[
  {{
    "fragment_id": 1,
    "is_valid": true,
    "confidence": 0.92,
    "core_text": "Extracted core entity text here"
  }},
  ...
]

Guidelines:
- is_valid should be true only if the fragment clearly represents a {entity_type.value}
- confidence should reflect how certain you are (0.0-1.0)
- core_text should be the minimal essential text that captures the entity
- If a fragment is not valid, still include it with is_valid=false and low confidence{type_specific_guidelines}
"""

        return prompt

    def _parse_validation_response(
        self,
        llm_response: str,
        candidates: List[Dict[str, Any]],
        entity_type: EntityType,
        confidence_threshold: float
    ) -> List[Entity]:
        """
        Parse LLM validation response into Entity objects

        Args:
            llm_response: Raw LLM response
            candidates: Original candidates
            entity_type: Entity type
            confidence_threshold: Minimum confidence

        Returns:
            List of validated Entity objects
        """
        validated_entities = []

        try:
            # Extract JSON array from response
            content = llm_response.strip()
            start_idx = content.find('[')
            end_idx = content.rfind(']')

            if start_idx == -1 or end_idx == -1:
                print(f"Warning: No JSON array in response: {content[:100]}")
                return []

            json_str = content[start_idx:end_idx+1]
            validation_results = json.loads(json_str)

            # Process each validation result
            for result in validation_results:
                fragment_id = result.get("fragment_id", 0)
                is_valid = result.get("is_valid", False)
                confidence = result.get("confidence", 0.0)
                core_text = result.get("core_text", "")

                # Check validity and confidence
                if not is_valid or confidence < confidence_threshold:
                    continue

                # Get original candidate
                if fragment_id < 1 or fragment_id > len(candidates):
                    continue

                candidate = candidates[fragment_id - 1]

                # Create Entity object
                entity = Entity(
                    id=f"{entity_type.value}_{uuid.uuid4().hex[:8]}",
                    type=entity_type,
                    text=core_text if core_text else candidate.get("text", ""),
                    confidence=confidence,
                    source_section=candidate.get("metadata", {}).get("section"),
                    metadata={
                        "validation_method": "llm_batch",
                        "original_text": candidate.get("text", ""),
                        "position": candidate.get("metadata", {}).get("position"),
                        "char_start": candidate.get("metadata", {}).get("char_start"),
                        "char_end": candidate.get("metadata", {}).get("char_end"),
                        "distance": candidate.get("distance", 0.0)
                    }
                )
                validated_entities.append(entity)

        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse JSON: {e}")
            print(f"Response: {llm_response[:300]}")
        except Exception as e:
            print(f"Warning: Unexpected error parsing validation: {e}")

        return validated_entities

    def get_metrics(self) -> Dict[str, Any]:
        """Get validation metrics"""
        return {
            "total_candidates": self.total_candidates,
            "total_validated": self.total_validated,
            "total_rejected": self.total_rejected,
            "validation_rate": (
                self.total_validated / self.total_candidates
                if self.total_candidates > 0 else 0.0
            ),
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost, 6)
        }

    def reset_metrics(self):
        """Reset metrics counters"""
        self.total_candidates = 0
        self.total_validated = 0
        self.total_rejected = 0
        self.total_tokens = 0
        self.total_cost = 0.0
