"""Data models for extracted entities and relationships"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
from enum import Enum
import json


class EntityType(Enum):
    """Types of extractable entities from scientific papers"""
    FACT = "fact"
    HYPOTHESIS = "hypothesis"
    EXPERIMENT = "experiment"
    METHOD = "method"
    RESULT = "result"
    DATASET = "dataset"
    ANALYSIS = "analysis"
    CONCLUSION = "conclusion"

    def __str__(self):
        return self.value


class RelationshipType(Enum):
    """Types of relationships between entities"""
    # Fact relationships
    FACT_TO_HYPOTHESIS = "fact_to_hypothesis"

    # Hypothesis relationships
    HYPOTHESIS_TO_EXPERIMENT = "hypothesis_to_experiment"
    HYPOTHESIS_TO_METHOD = "hypothesis_to_method"

    # Experiment/Method relationships
    EXPERIMENT_USES_TECHNIQUE = "experiment_uses_technique"
    EXPERIMENT_USES_DATASET = "experiment_uses_dataset"
    METHOD_TO_RESULT = "method_to_result"

    # Result relationships
    RESULT_TO_ANALYSIS = "result_to_analysis"
    RESULT_TO_CONCLUSION = "result_to_conclusion"

    # Analysis relationships
    ANALYSIS_TO_CONCLUSION = "analysis_to_conclusion"

    # Generic relationships
    BASED_ON = "based_on"
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    RELATED_TO = "related_to"

    def __str__(self):
        return self.value


@dataclass
class Entity:
    """
    A single extracted entity from a scientific paper

    Attributes:
        id: Unique identifier for the entity
        type: Type of entity (fact, hypothesis, etc.)
        text: The extracted text content
        confidence: Confidence score (0.0 - 1.0)
        source_section: Section where entity was found (abstract, methods, etc.)
        metadata: Additional metadata (page number, position, etc.)
    """
    id: str
    type: EntityType
    text: str
    confidence: float = 1.0
    source_section: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate entity data"""
        # Convert string type to EntityType if needed
        if isinstance(self.type, str):
            self.type = EntityType(self.type)

        # Validate confidence
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary"""
        return {
            "id": self.id,
            "type": self.type.value,
            "text": self.text,
            "confidence": self.confidence,
            "source_section": self.source_section,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        """Create entity from dictionary"""
        return cls(
            id=data["id"],
            type=EntityType(data["type"]),
            text=data["text"],
            confidence=data.get("confidence", 1.0),
            source_section=data.get("source_section"),
            metadata=data.get("metadata", {})
        )

    def to_json(self) -> str:
        """Convert entity to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    def __str__(self) -> str:
        """String representation"""
        preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return f"Entity({self.type.value}, id={self.id}, text='{preview}')"


@dataclass
class Relationship:
    """
    A relationship between two entities

    Attributes:
        source_id: ID of the source entity
        target_id: ID of the target entity
        relationship_type: Type of relationship
        confidence: Confidence score (0.0 - 1.0)
        metadata: Additional metadata
    """
    source_id: str
    target_id: str
    relationship_type: RelationshipType
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate relationship data"""
        # Convert string type to RelationshipType if needed
        if isinstance(self.relationship_type, str):
            self.relationship_type = RelationshipType(self.relationship_type)

        # Validate confidence
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert relationship to dictionary"""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relationship_type": self.relationship_type.value,
            "confidence": self.confidence,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Relationship":
        """Create relationship from dictionary"""
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            relationship_type=RelationshipType(data["relationship_type"]),
            confidence=data.get("confidence", 1.0),
            metadata=data.get("metadata", {})
        )

    def to_json(self) -> str:
        """Convert relationship to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    def __str__(self) -> str:
        """String representation"""
        return f"Relationship({self.source_id} --[{self.relationship_type.value}]--> {self.target_id})"


@dataclass
class EntitySchema:
    """
    Schema definition for entity types with metadata and patterns

    Used by Entity-Centric Pipeline for semantic retrieval and validation

    Attributes:
        entity_type: Type of entity this schema describes
        description: Human-readable description of the entity type
        typical_sections: IMRAD sections where this entity commonly appears
        signal_patterns: Regex patterns that indicate this entity type
    """
    entity_type: EntityType
    description: str
    typical_sections: List[str]
    signal_patterns: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary"""
        return {
            "entity_type": self.entity_type.value,
            "description": self.description,
            "typical_sections": self.typical_sections,
            "signal_patterns": self.signal_patterns
        }


# Predefined schemas for all entity types
ENTITY_SCHEMAS: Dict[EntityType, EntitySchema] = {
    EntityType.FACT: EntitySchema(
        entity_type=EntityType.FACT,
        description="Established knowledge, prior findings, or background information from literature",
        typical_sections=["introduction", "abstract", "discussion"],
        signal_patterns=[
            r"\b(has\s+been\s+shown|previous\s+stud(?:y|ies)|it\s+is\s+known)",
            r"\b(established|demonstrated|reported|documented)",
            r"\b(evidence\s+suggests|research\s+shows|studies\s+indicate)"
        ]
    ),

    EntityType.HYPOTHESIS: EntitySchema(
        entity_type=EntityType.HYPOTHESIS,
        description="Scientific assumption, prediction, or proposed explanation to be tested",
        typical_sections=["introduction", "abstract"],
        signal_patterns=[
            r"\b(we\s+hypothesi[zs]e|we\s+propose|we\s+predict)",
            r"\b(suggests?\s+that|may\s+explain|could\s+indicate)",
            r"\b(it\s+is\s+likely|we\s+expect|we\s+anticipate)",
            r"\b(research\s+question|aim\s+(?:was|is)\s+to)"
        ]
    ),

    EntityType.EXPERIMENT: EntitySchema(
        entity_type=EntityType.EXPERIMENT,
        description=(
            "Experimental procedures, setups, or manipulations performed to test hypotheses. "
            "Includes: knockout/knockdown experiments (CRISPR, siRNA), comparative studies "
            "(control vs treatment), experimental designs, in vivo/in vitro assays, "
            "cell sorting experiments, intervention studies. "
            "Can be concise (e.g., 'CRISPR knockout mice', 'deletion experiment') "
            "if experimental intent is clear. Focus on WHAT WAS DONE, not just observations."
        ),
        typical_sections=["methods", "materials"],
        signal_patterns=[
            r"\b(experiment|experimental\s+design|trial|study\s+design)",
            r"\b(procedure|protocol|treatment|intervention)",
            r"\b(randomized|controlled|double-blind|placebo)",
            r"\b(cohort|clinical\s+trial|in\s+vivo|in\s+vitro)"
        ]
    ),

    EntityType.METHOD: EntitySchema(
        entity_type=EntityType.METHOD,
        description=(
            "Detailed methodological descriptions including procedures, protocols, parameters, and instruments. "
            "Must describe HOW the method was applied (steps, conditions, settings), not just mention that something was used. "
            "Should contain specifics like: concentrations, temperatures, durations, software versions, instrument models, or procedural steps."
        ),
        typical_sections=["methods", "materials"],
        signal_patterns=[
            r"\b(we\s+used|using|employed|performed|conducted)",
            r"\b(method|technique|approach|assay|analysis)",
            r"\b(protocol|procedure|software|tool|instrument)",
            r"\b(RT-PCR|Western\s+blot|ELISA|microscopy|sequencing)",
            r"\b(trained\s+(?:with|using)|implemented|applied)"
        ]
    ),

    EntityType.RESULT: EntitySchema(
        entity_type=EntityType.RESULT,
        description="Experimental findings, measurements, observations, or data outcomes",
        typical_sections=["results", "discussion"],
        signal_patterns=[
            r"\b(we\s+found|we\s+observed|showed\s+that|revealed)",
            r"\b(result(?:s|ed)|finding(?:s)?|outcome(?:s)?|data\s+showed)",
            r"\b(significant(?:ly)?|increase[ds]?|decrease[ds]?|reduced?)",
            r"\b(p\s*[<>=]|correlation|association|effect)",
            r"\b(measured|detected|identified|quantified)"
        ]
    ),

    EntityType.DATASET: EntitySchema(
        entity_type=EntityType.DATASET,
        description=(
            "Data collections used or generated in research. "
            "Includes: public repository accessions (GSE*, GSM*, SRA*, etc.), "
            "generated datasets (scRNA-seq data, ATAC-seq data, ChIP-seq data), "
            "references to existing datasets ('published datasets', 'integrated with...'), "
            "sample collections, or cohort descriptions. "
            "Brief mentions are acceptable if data source is identifiable "
            "(e.g., 'GSE137319 from GEO database', 'scRNA-seq datasets')."
        ),
        typical_sections=["methods", "materials", "results"],
        signal_patterns=[
            r"\b(dataset|database|repository|data\s+collection)",
            r"\b(samples?|cohort|participant(?:s)?|subject(?:s)?)",
            r"\b(GEO|ArrayExpress|TCGA|dbGaP|GenBank)",
            r"\b(publicly\s+available|deposited\s+in|obtained\s+from)",
            r"\b(training\s+set|test\s+set|validation\s+set)"
        ]
    ),

    EntityType.ANALYSIS: EntitySchema(
        entity_type=EntityType.ANALYSIS,
        description="Statistical tests, computational methods, or data processing approaches",
        typical_sections=["methods", "results"],
        signal_patterns=[
            r"\b(statistical\s+analysis|data\s+analysis|computational)",
            r"\b(t-test|ANOVA|regression|chi-square|Mann-Whitney)",
            r"\b(normalized|adjusted|corrected|transformed)",
            r"\b(calculated|computed|estimated|modeled)",
            r"\b(machine\s+learning|neural\s+network|classification)"
        ]
    ),

    EntityType.CONCLUSION: EntitySchema(
        entity_type=EntityType.CONCLUSION,
        description="Interpretations, implications, or overall findings drawn from results",
        typical_sections=["conclusion", "discussion", "abstract"],
        signal_patterns=[
            r"\b(conclude|conclusion(?:s)?|in\s+summary|overall)",
            r"\b(these\s+findings|our\s+(?:results|study)\s+(?:suggest|demonstrate|indicate))",
            r"\b(implication(?:s)?|significance|impact|contribution)",
            r"\b(future\s+(?:research|studies|work|directions?))",
            r"\b(collectively|taken\s+together|in\s+conclusion)"
        ]
    )
}


def get_entity_schema(entity_type: EntityType) -> EntitySchema:
    """
    Get predefined schema for an entity type

    Args:
        entity_type: Type of entity

    Returns:
        EntitySchema for the given type

    Raises:
        KeyError: If entity type not found in ENTITY_SCHEMAS
    """
    if entity_type not in ENTITY_SCHEMAS:
        raise KeyError(f"No schema defined for entity type: {entity_type}")
    return ENTITY_SCHEMAS[entity_type]
