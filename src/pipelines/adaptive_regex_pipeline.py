"""Adaptive token pipeline using LLM-guided token generation"""

import time
from typing import Dict, Any, List, Optional
from collections import defaultdict

from .base_pipeline import BasePipeline
from src.models import (
    Entity,
    Relationship,
    EntityType,
    RelationshipType,
    ExtractionResult,
    PipelineMetrics
)
from src.parsers import PDFParser, ParsedDocument
from src.extractors.llm_guided_regex import LLMGuidedTokenExtractor


class AdaptiveRegexPipeline(BasePipeline):
    """
    Adaptive token pipeline with LLM-guided token generation

    Strategy:
    1. Parse PDF with IMRAD sections
    2. Extract facts from Abstract + Introduction using LLM (Stage 1)
    3. Generate context-aware tokens based on facts using LLM (Stage 2)
    4. Apply token matching to extract hypotheses (Stage 3 - FREE)
    5. Build relationships between facts and hypotheses

    Token-based advantages:
    - Simpler than regex (no validation)
    - Faster execution (substring matching)
    - Flexible confidence scoring
    - Easier for LLM to generate

    Target cost: < $0.015/paper
    Target precision: ≥ 85%
    Target recall: ≥ 80%

    Key advantage: Context-aware tokens adapt to each paper's domain and terminology
    """

    def __init__(
        self,
        config: Dict[str, Any],
        llm_provider: str = "openai",
        llm_model: str = "gpt-5-mini",
        temperature: float = 0.2,
        confidence_threshold: float = 0.7
    ):
        """
        Initialize adaptive regex pipeline

        Args:
            config: Configuration dictionary
            llm_provider: LLM provider for pattern generation
            llm_model: LLM model for pattern generation
            temperature: LLM temperature
            confidence_threshold: Minimum confidence for extracted entities
        """
        super().__init__(config)

        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.temperature = temperature
        self.confidence_threshold = confidence_threshold

        # Initialize LLM-guided token extractor
        self.llm_guided_extractor = LLMGuidedTokenExtractor(
            provider=llm_provider,
            model=llm_model,
            temperature=temperature,
            confidence_threshold=confidence_threshold
        )

    def extract(
        self,
        parsed_doc: ParsedDocument,
        paper_id: str
    ) -> ExtractionResult:
        """
        Extract entities and relationships using adaptive regex approach

        Args:
            parsed_doc: ParsedDocument with text, sections, and metadata
            paper_id: Unique paper identifier

        Returns:
            ExtractionResult with extracted entities and relationships
        """
        start_time = time.time()

        # Validate input
        self._validate_parsed_doc(parsed_doc)

        all_entities = []
        extraction_methods = defaultdict(int)

        # Get Abstract and Introduction sections
        abstract_text = parsed_doc.get_section("abstract") or ""
        intro_text = parsed_doc.get_section("introduction") or ""

        if not abstract_text and not intro_text:
            print("Warning: No Abstract or Introduction found. Pipeline requires these sections.")
            # Return empty result
            return self._create_empty_result(paper_id, parsed_doc.metadata, time.time() - start_time)

        # ========== THREE-STAGE EXTRACTION ==========

        print(f"\n{'='*60}")
        print(f"Processing paper: {paper_id}")
        print(f"{'='*60}")

        # Extract facts and hypotheses using LLM-guided token approach
        facts, hypotheses, generated_tokens = self.llm_guided_extractor.extract_facts_and_hypotheses(
            abstract_text=abstract_text,
            intro_text=intro_text,
            paper_id=paper_id
        )

        # Add to entities
        all_entities.extend(facts)
        all_entities.extend(hypotheses)

        extraction_methods["llm_guided_facts"] = len(facts)
        extraction_methods["llm_guided_hypotheses"] = len(hypotheses)

        print(f"\nExtraction summary:")
        print(f"  Facts: {len(facts)}")
        print(f"  Hypotheses: {len(hypotheses)}")
        print(f"  Generated tokens: {len(generated_tokens.get('required_tokens', []))} required + {len(generated_tokens.get('optional_tokens', []))} optional")

        # ========== BUILD RELATIONSHIPS ==========

        relationships = self._build_relationships(all_entities)

        # ========== CALCULATE METRICS ==========

        # Get LLM metrics
        llm_metrics = self.llm_guided_extractor.get_metrics()
        total_tokens = llm_metrics["total_tokens"]
        total_cost = llm_metrics["total_cost_usd"]

        processing_time = time.time() - start_time

        metrics = PipelineMetrics(
            processing_time=processing_time,
            tokens_used=total_tokens,
            cost_usd=total_cost,
            entities_extracted=len(all_entities),
            relationships_extracted=len(relationships),
            metadata={
                "pipeline": "adaptive_tokens",
                "extraction_methods": dict(extraction_methods),
                "generated_tokens": generated_tokens,
                "required_tokens_count": len(generated_tokens.get("required_tokens", [])),
                "optional_tokens_count": len(generated_tokens.get("optional_tokens", [])),
                "llm_model": self.llm_model,
                "sections_processed": list(parsed_doc.imrad_sections.keys()) if parsed_doc.imrad_sections else []
            }
        )

        self.last_metrics = metrics

        print(f"\nPipeline metrics:")
        print(f"  Total cost: ${total_cost:.4f}")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Processing time: {processing_time:.2f}s")
        print(f"  Cost per entity: ${total_cost / max(len(all_entities), 1):.4f}")
        print(f"{'='*60}\n")

        # ========== GROUP ENTITIES BY TYPE ==========

        entities_by_type = defaultdict(list)
        for entity in all_entities:
            entities_by_type[entity.type.value].append(entity)

        # Build result
        result = ExtractionResult(
            paper_id=paper_id,
            entities=dict(entities_by_type),
            relationships=relationships,
            metrics=metrics,
            metadata=parsed_doc.metadata
        )

        return result

    def _build_relationships(self, entities: List[Entity]) -> List[Relationship]:
        """
        Build relationships between entities

        Args:
            entities: List of extracted entities

        Returns:
            List of relationships
        """
        relationships = []

        # Create entity lookup by type
        entities_by_type = defaultdict(list)
        for entity in entities:
            entities_by_type[entity.type].append(entity)

        # Rule 1: Facts → Hypotheses (from same section)
        # High confidence relationship since both come from Abstract+Intro
        for fact in entities_by_type[EntityType.FACT]:
            for hypothesis in entities_by_type[EntityType.HYPOTHESIS]:
                # Both extracted from abstract+intro, so high confidence link
                rel = Relationship(
                    source_id=fact.id,
                    target_id=hypothesis.id,
                    relationship_type=RelationshipType.FACT_TO_HYPOTHESIS,
                    confidence=0.85  # High confidence due to context-aware extraction
                )
                relationships.append(rel)

        return relationships

    def _create_empty_result(
        self,
        paper_id: str,
        metadata: Optional[Dict[str, Any]],
        processing_time: float
    ) -> ExtractionResult:
        """Create empty result when extraction fails"""
        metrics = PipelineMetrics(
            processing_time=processing_time,
            tokens_used=0,
            cost_usd=0.0,
            entities_extracted=0,
            relationships_extracted=0,
            metadata={
                "pipeline": "adaptive_tokens",
                "error": "No Abstract or Introduction found"
            }
        )

        return ExtractionResult(
            paper_id=paper_id,
            entities={},
            relationships=[],
            metrics=metrics,
            metadata=metadata or {}
        )

    def get_description(self) -> str:
        """Get pipeline description"""
        return (
            f"Adaptive Token Pipeline (LLM-guided token generation). "
            f"Three-stage approach: LLM extracts facts + generates context-aware tokens + FREE matching. "
            f"Estimated cost: ${self.get_estimated_cost():.4f}/paper"
        )

    def get_estimated_cost(self) -> float:
        """
        Get estimated cost per paper

        Returns:
            Estimated cost in USD
        """
        # Stage 1: Extract facts (~400 tokens input, 200 tokens output)
        # Stage 2: Generate tokens (~600 tokens input, 300 tokens output)
        #          (simpler output than regex → fewer tokens)
        # Stage 3: Apply token matching (FREE - simple substring search)
        #
        # Total: ~1500 tokens with gpt-5-mini
        # Cost: ~$0.01-0.015/paper (slightly cheaper than regex)
        return 0.012
