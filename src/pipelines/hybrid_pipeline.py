"""Hybrid extraction pipeline combining regex, NLP, and selective LLM"""

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
from src.extractors import (
    HypothesisExtractor,
    MethodExtractor,
    DatasetExtractor,
    ExperimentExtractor,
    ResultExtractor,
    NLPFactExtractor,
    SelectiveLLMExtractor
)


class HybridPipeline(BasePipeline):
    """
    Hybrid extraction pipeline for cost-optimized extraction

    Strategy:
    1. Parse PDF with IMRAD sections
    2. Apply pattern extractors to relevant sections (FREE)
    3. Apply NLP extractor for facts (~$0.001/paper)
    4. Use selective LLM only for low-confidence cases (~$0.01-0.02/paper)

    Target cost: < $0.02/paper
    Target precision: ≥ 85%
    Target recall: ≥ 80%
    """

    def __init__(
        self,
        config: Dict[str, Any],
        confidence_threshold: float = 0.75,
        llm_provider: str = "openai",
        llm_model: str = "gpt-5-mini"
    ):
        """
        Initialize hybrid pipeline

        Args:
            config: Configuration dictionary
            confidence_threshold: Threshold for using LLM fallback
            llm_provider: LLM provider for fallback
            llm_model: LLM model for fallback
        """
        super().__init__(config)

        self.confidence_threshold = confidence_threshold
        self.llm_provider = llm_provider
        self.llm_model = llm_model

        # Initialize extractors
        self.hypothesis_extractor = HypothesisExtractor(
            confidence_threshold=config.get("pattern_confidence_threshold", 0.7)
        )
        self.method_extractor = MethodExtractor(
            confidence_threshold=config.get("pattern_confidence_threshold", 0.7)
        )
        self.dataset_extractor = DatasetExtractor(
            confidence_threshold=config.get("pattern_confidence_threshold", 0.7)
        )
        self.experiment_extractor = ExperimentExtractor(
            confidence_threshold=config.get("pattern_confidence_threshold", 0.7)
        )
        self.result_extractor = ResultExtractor(
            confidence_threshold=config.get("pattern_confidence_threshold", 0.7)
        )

        # NLP extractor
        try:
            self.nlp_extractor = NLPFactExtractor(
                confidence_threshold=config.get("nlp_confidence_threshold", 0.6)
            )
        except ImportError:
            print("Warning: spaCy not available. Facts extraction disabled.")
            self.nlp_extractor = None

        # Selective LLM extractor (lazy initialization)
        self.llm_extractor = None

    def extract(
        self,
        parsed_doc: ParsedDocument,
        paper_id: str
    ) -> ExtractionResult:
        """
        Extract entities and relationships using hybrid approach

        Args:
            parsed_doc: ParsedDocument with text, sections, and metadata
            paper_id: Unique paper identifier

        Returns:
            ExtractionResult with extracted entities and relationships
        """
        start_time = time.time()

        # Validate input
        self._validate_parsed_doc(parsed_doc)

        # Track metrics
        total_tokens = 0
        total_cost = 0.0
        extraction_methods = defaultdict(int)

        all_entities = []

        # ========== PHASE 1: Pattern Extraction (FREE) ==========

        # Extract hypotheses from Introduction
        intro_text = parsed_doc.get_section("introduction") or ""
        if intro_text:
            hyp_entities = self.hypothesis_extractor.extract(
                intro_text,
                section_name="introduction"
            )
            all_entities.extend(hyp_entities)
            extraction_methods["pattern_hypothesis"] += len(hyp_entities)

        # Extract methods from Methods section
        methods_text = parsed_doc.get_section("methods") or ""
        if methods_text:
            method_entities = self.method_extractor.extract(
                methods_text,
                section_name="methods"
            )
            all_entities.extend(method_entities)
            extraction_methods["pattern_methods"] += len(method_entities)

            # Extract datasets from Methods
            dataset_entities = self.dataset_extractor.extract(
                methods_text,
                section_name="methods"
            )
            all_entities.extend(dataset_entities)
            extraction_methods["pattern_datasets"] += len(dataset_entities)

            # Extract experiments from Methods
            exp_entities = self.experiment_extractor.extract(
                methods_text,
                section_name="methods"
            )
            all_entities.extend(exp_entities)
            extraction_methods["pattern_experiments"] += len(exp_entities)

        # Extract results from Results section
        results_text = parsed_doc.get_section("results") or ""
        if results_text:
            result_entities = self.result_extractor.extract(
                results_text,
                section_name="results"
            )
            all_entities.extend(result_entities)
            extraction_methods["pattern_results"] += len(result_entities)

        # ========== PHASE 2: NLP Extraction (~$0.001) ==========

        # Extract facts from Introduction using NLP
        if self.nlp_extractor and intro_text:
            try:
                fact_entities = self.nlp_extractor.extract(
                    intro_text,
                    section_name="introduction"
                )
                all_entities.extend(fact_entities)
                extraction_methods["nlp_facts"] += len(fact_entities)
            except Exception as e:
                print(f"Warning: NLP extraction failed: {e}")

        # ========== PHASE 3: Selective LLM Fallback (~$0.01-0.02) ==========

        relationships = self._build_relationships(all_entities)

        # ========== Group entities by type ==========

        entities_by_type = defaultdict(list)
        for entity in all_entities:
            entities_by_type[entity.type.value].append(entity)

        # Calculate metrics
        processing_time = time.time() - start_time
        metrics = PipelineMetrics(
            processing_time=processing_time,
            tokens_used=total_tokens,
            cost_usd=total_cost,
            entities_extracted=len(all_entities),
            relationships_extracted=len(relationships),
            metadata={
                "pipeline": "hybrid",
                "extraction_methods": dict(extraction_methods),
                "sections_processed": list(parsed_doc.imrad_sections.keys()) if parsed_doc.imrad_sections else []
            }
        )

        self.last_metrics = metrics

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
        Build relationships between entities based on proximity and types

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
        for fact in entities_by_type[EntityType.FACT]:
            for hypothesis in entities_by_type[EntityType.HYPOTHESIS]:
                if fact.source_section == hypothesis.source_section:
                    rel = Relationship(
                        source_id=fact.id,
                        target_id=hypothesis.id,
                        relationship_type=RelationshipType.FACT_TO_HYPOTHESIS,
                        confidence=min(fact.confidence, hypothesis.confidence)
                    )
                    relationships.append(rel)

        # Rule 2: Hypotheses → Experiments
        for hypothesis in entities_by_type[EntityType.HYPOTHESIS]:
            for experiment in entities_by_type[EntityType.EXPERIMENT]:
                rel = Relationship(
                    source_id=hypothesis.id,
                    target_id=experiment.id,
                    relationship_type=RelationshipType.HYPOTHESIS_TO_EXPERIMENT,
                    confidence=0.75
                )
                relationships.append(rel)

        # Rule 3: Experiments → Techniques
        for experiment in entities_by_type[EntityType.EXPERIMENT]:
            for technique in entities_by_type[EntityType.TECHNIQUE]:
                if technique.source_section == "methods":
                    rel = Relationship(
                        source_id=experiment.id,
                        target_id=technique.id,
                        relationship_type=RelationshipType.EXPERIMENT_USES_TECHNIQUE,
                        confidence=0.7
                    )
                    relationships.append(rel)

        # Rule 4: Experiments → Datasets
        for experiment in entities_by_type[EntityType.EXPERIMENT]:
            for dataset in entities_by_type[EntityType.DATASET]:
                rel = Relationship(
                    source_id=experiment.id,
                    target_id=dataset.id,
                    relationship_type=RelationshipType.EXPERIMENT_USES_DATASET,
                    confidence=0.7
                )
                relationships.append(rel)

        # Rule 5: Techniques → Results
        for technique in entities_by_type[EntityType.TECHNIQUE]:
            for result in entities_by_type[EntityType.RESULT]:
                rel = Relationship(
                    source_id=technique.id,
                    target_id=result.id,
                    relationship_type=RelationshipType.METHOD_TO_RESULT,
                    confidence=0.65
                )
                relationships.append(rel)

        # Rule 6: Results → Conclusions
        for result in entities_by_type[EntityType.RESULT]:
            for conclusion in entities_by_type[EntityType.CONCLUSION]:
                rel = Relationship(
                    source_id=result.id,
                    target_id=conclusion.id,
                    relationship_type=RelationshipType.RESULT_TO_CONCLUSION,
                    confidence=0.75
                )
                relationships.append(rel)

        return relationships

    def get_description(self) -> str:
        """Get pipeline description"""
        return (
            f"Hybrid Pipeline (Pattern + NLP + Selective LLM). "
            f"Cost-optimized approach. "
        )

