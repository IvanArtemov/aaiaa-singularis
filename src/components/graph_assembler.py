"""Graph assembler for building entity relationships"""

import re
from typing import List, Dict, Optional, Set
from collections import defaultdict

from src.models import (
    Entity,
    Relationship,
    EntityType,
    RelationshipType,
    Sentence
)


class GraphAssembler:
    """
    Assembles knowledge graph from extracted entities

    Uses heuristics based on:
    - Entity type combinations
    - Section proximity
    - Text co-occurrence
    - Keyword matching

    Cost: FREE (heuristic-based, no LLM)
    Optional: LLM refinement for ambiguous cases (+$0.005-0.01/paper)
    """

    def __init__(
        self,
        use_llm_refinement: bool = False,
        proximity_window: int = 3,
        min_relationship_confidence: float = 0.6
    ):
        """
        Initialize graph assembler

        Args:
            use_llm_refinement: Use LLM to refine relationships (not implemented)
            proximity_window: Sentence proximity window for relationships
            min_relationship_confidence: Minimum confidence for relationships
        """
        self.use_llm_refinement = use_llm_refinement
        self.proximity_window = proximity_window
        self.min_confidence = min_relationship_confidence

        # Metrics
        self.total_relationships = 0
        self.relationships_by_type = defaultdict(int)

    def assemble_graph(
        self,
        entities: List[Entity],
        sentences: Optional[List[Sentence]] = None
    ) -> List[Relationship]:
        """
        Assemble relationships between entities

        Args:
            entities: List of extracted entities
            sentences: Optional list of sentences for proximity detection

        Returns:
            List of relationships
        """
        if not entities:
            return []

        relationships = []

        # Group entities by type
        entities_by_type = self._group_by_type(entities)

        # Apply relationship rules
        relationships.extend(
            self._link_facts_to_hypotheses(entities_by_type)
        )
        relationships.extend(
            self._link_hypotheses_to_experiments(entities_by_type)
        )
        relationships.extend(
            self._link_experiments_to_techniques(entities_by_type)
        )
        relationships.extend(
            self._link_experiments_to_datasets(entities_by_type)
        )
        relationships.extend(
            self._link_techniques_to_results(entities_by_type)
        )
        relationships.extend(
            self._link_results_to_analyses(entities_by_type)
        )
        relationships.extend(
            self._link_results_to_conclusions(entities_by_type)
        )
        relationships.extend(
            self._link_analyses_to_conclusions(entities_by_type)
        )

        # Filter by confidence
        relationships = [
            r for r in relationships
            if r.confidence >= self.min_confidence
        ]

        # Update metrics
        self.total_relationships = len(relationships)
        for rel in relationships:
            self.relationships_by_type[rel.relationship_type.value] += 1

        return relationships

    def _group_by_type(self, entities: List[Entity]) -> Dict[EntityType, List[Entity]]:
        """Group entities by type"""
        grouped = defaultdict(list)
        for entity in entities:
            grouped[entity.type].append(entity)
        return grouped

    def _link_facts_to_hypotheses(
        self,
        entities_by_type: Dict[EntityType, List[Entity]]
    ) -> List[Relationship]:
        """Link facts to hypotheses (same section, likely introduction)"""
        relationships = []

        facts = entities_by_type.get(EntityType.FACT, [])
        hypotheses = entities_by_type.get(EntityType.HYPOTHESIS, [])

        for fact in facts:
            for hypothesis in hypotheses:
                # Rule: same section (typically introduction)
                if fact.source_section == hypothesis.source_section:
                    confidence = min(fact.confidence, hypothesis.confidence) * 0.8

                    rel = Relationship(
                        source_id=fact.id,
                        target_id=hypothesis.id,
                        relationship_type=RelationshipType.FACT_TO_HYPOTHESIS,
                        confidence=confidence,
                        metadata={
                            "rule": "same_section",
                            "section": fact.source_section
                        }
                    )
                    relationships.append(rel)

        return relationships

    def _link_hypotheses_to_experiments(
        self,
        entities_by_type: Dict[EntityType, List[Entity]]
    ) -> List[Relationship]:
        """Link hypotheses to experiments that test them"""
        relationships = []

        hypotheses = entities_by_type.get(EntityType.HYPOTHESIS, [])
        experiments = entities_by_type.get(EntityType.EXPERIMENT, [])

        # Hypotheses are in intro, experiments in methods
        # Link with moderate confidence (assumes experiments test hypotheses)
        for hypothesis in hypotheses:
            for experiment in experiments:
                confidence = min(hypothesis.confidence, experiment.confidence) * 0.75

                rel = Relationship(
                    source_id=hypothesis.id,
                    target_id=experiment.id,
                    relationship_type=RelationshipType.HYPOTHESIS_TO_EXPERIMENT,
                    confidence=confidence,
                    metadata={
                        "rule": "hypothesis_experiment_link"
                    }
                )
                relationships.append(rel)

        return relationships

    def _link_experiments_to_techniques(
        self,
        entities_by_type: Dict[EntityType, List[Entity]]
    ) -> List[Relationship]:
        """Link experiments to techniques used"""
        relationships = []

        experiments = entities_by_type.get(EntityType.EXPERIMENT, [])
        techniques = entities_by_type.get(EntityType.TECHNIQUE, [])

        for experiment in experiments:
            for technique in techniques:
                # Rule: both typically in methods section
                if technique.source_section in ["methods", "materials"]:
                    # Check for keyword overlap
                    overlap = self._text_overlap(experiment.text, technique.text)

                    if overlap > 0.1:  # At least 10% keyword overlap
                        confidence = min(experiment.confidence, technique.confidence) * 0.85
                    else:
                        confidence = min(experiment.confidence, technique.confidence) * 0.7

                    rel = Relationship(
                        source_id=experiment.id,
                        target_id=technique.id,
                        relationship_type=RelationshipType.EXPERIMENT_USES_TECHNIQUE,
                        confidence=confidence,
                        metadata={
                            "rule": "experiment_technique",
                            "text_overlap": round(overlap, 2)
                        }
                    )
                    relationships.append(rel)

        return relationships

    def _link_experiments_to_datasets(
        self,
        entities_by_type: Dict[EntityType, List[Entity]]
    ) -> List[Relationship]:
        """Link experiments to datasets used"""
        relationships = []

        experiments = entities_by_type.get(EntityType.EXPERIMENT, [])
        datasets = entities_by_type.get(EntityType.DATASET, [])

        for experiment in experiments:
            for dataset in datasets:
                # Datasets typically mentioned in methods/materials
                if dataset.source_section in ["methods", "materials"]:
                    confidence = min(experiment.confidence, dataset.confidence) * 0.7

                    rel = Relationship(
                        source_id=experiment.id,
                        target_id=dataset.id,
                        relationship_type=RelationshipType.EXPERIMENT_USES_DATASET,
                        confidence=confidence,
                        metadata={
                            "rule": "experiment_dataset"
                        }
                    )
                    relationships.append(rel)

        return relationships

    def _link_techniques_to_results(
        self,
        entities_by_type: Dict[EntityType, List[Entity]]
    ) -> List[Relationship]:
        """Link techniques (methods) to results"""
        relationships = []

        techniques = entities_by_type.get(EntityType.TECHNIQUE, [])
        results = entities_by_type.get(EntityType.RESULT, [])

        # Methods → Results is a common relationship
        for technique in techniques:
            for result in results:
                # Results section should have results
                if result.source_section in ["results", "discussion"]:
                    # Check for keyword mentions
                    overlap = self._text_overlap(technique.text, result.text)

                    if overlap > 0.15:  # Higher overlap = stronger link
                        confidence = min(technique.confidence, result.confidence) * 0.75
                    else:
                        confidence = min(technique.confidence, result.confidence) * 0.65

                    rel = Relationship(
                        source_id=technique.id,
                        target_id=result.id,
                        relationship_type=RelationshipType.METHOD_TO_RESULT,
                        confidence=confidence,
                        metadata={
                            "rule": "method_result",
                            "text_overlap": round(overlap, 2)
                        }
                    )
                    relationships.append(rel)

        return relationships

    def _link_results_to_analyses(
        self,
        entities_by_type: Dict[EntityType, List[Entity]]
    ) -> List[Relationship]:
        """Link results to analyses performed on them"""
        relationships = []

        results = entities_by_type.get(EntityType.RESULT, [])
        analyses = entities_by_type.get(EntityType.ANALYSIS, [])

        for result in results:
            for analysis in analyses:
                # Same section or adjacent sections
                same_section = result.source_section == analysis.source_section

                if same_section:
                    confidence = min(result.confidence, analysis.confidence) * 0.8
                else:
                    confidence = min(result.confidence, analysis.confidence) * 0.65

                rel = Relationship(
                    source_id=result.id,
                    target_id=analysis.id,
                    relationship_type=RelationshipType.RESULT_TO_ANALYSIS,
                    confidence=confidence,
                    metadata={
                        "rule": "result_analysis",
                        "same_section": same_section
                    }
                )
                relationships.append(rel)

        return relationships

    def _link_results_to_conclusions(
        self,
        entities_by_type: Dict[EntityType, List[Entity]]
    ) -> List[Relationship]:
        """Link results to conclusions drawn from them"""
        relationships = []

        results = entities_by_type.get(EntityType.RESULT, [])
        conclusions = entities_by_type.get(EntityType.CONCLUSION, [])

        for result in results:
            for conclusion in conclusions:
                # Conclusions typically in discussion/conclusion sections
                if conclusion.source_section in ["conclusion", "discussion"]:
                    # Check for keyword overlap
                    overlap = self._text_overlap(result.text, conclusion.text)

                    if overlap > 0.2:  # Strong text overlap
                        confidence = min(result.confidence, conclusion.confidence) * 0.85
                    else:
                        confidence = min(result.confidence, conclusion.confidence) * 0.75

                    rel = Relationship(
                        source_id=result.id,
                        target_id=conclusion.id,
                        relationship_type=RelationshipType.RESULT_TO_CONCLUSION,
                        confidence=confidence,
                        metadata={
                            "rule": "result_conclusion",
                            "text_overlap": round(overlap, 2)
                        }
                    )
                    relationships.append(rel)

        return relationships

    def _link_analyses_to_conclusions(
        self,
        entities_by_type: Dict[EntityType, List[Entity]]
    ) -> List[Relationship]:
        """Link analyses to conclusions"""
        relationships = []

        analyses = entities_by_type.get(EntityType.ANALYSIS, [])
        conclusions = entities_by_type.get(EntityType.CONCLUSION, [])

        for analysis in analyses:
            for conclusion in conclusions:
                # Analysis → Conclusion is a logical flow
                confidence = min(analysis.confidence, conclusion.confidence) * 0.7

                rel = Relationship(
                    source_id=analysis.id,
                    target_id=conclusion.id,
                    relationship_type=RelationshipType.ANALYSIS_TO_CONCLUSION,
                    confidence=confidence,
                    metadata={
                        "rule": "analysis_conclusion"
                    }
                )
                relationships.append(rel)

        return relationships

    def _text_overlap(self, text1: str, text2: str) -> float:
        """
        Calculate keyword overlap between two texts

        Args:
            text1: First text
            text2: Second text

        Returns:
            Overlap ratio (0.0-1.0)
        """
        # Extract keywords (alphanumeric tokens > 3 chars)
        keywords1 = self._extract_keywords(text1)
        keywords2 = self._extract_keywords(text2)

        if not keywords1 or not keywords2:
            return 0.0

        # Calculate Jaccard similarity
        intersection = keywords1 & keywords2
        union = keywords1 | keywords2

        return len(intersection) / len(union) if union else 0.0

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract meaningful keywords from text"""
        # Lowercase and extract words
        words = re.findall(r'\b\w+\b', text.lower())

        # Filter: length > 3, not common stop words
        stop_words = {
            "the", "and", "for", "that", "this", "with", "was", "were",
            "are", "from", "been", "have", "has", "had", "but", "not"
        }

        keywords = {
            word for word in words
            if len(word) > 3 and word not in stop_words
        }

        return keywords

    def get_metrics(self) -> Dict[str, int]:
        """Get graph assembly metrics"""
        return {
            "total_relationships": self.total_relationships,
            "relationships_by_type": dict(self.relationships_by_type)
        }

    def reset_metrics(self):
        """Reset metrics counters"""
        self.total_relationships = 0
        self.relationships_by_type = defaultdict(int)
