"""Data models for extracted entities and relationships"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
from enum import Enum
import json


class EntityType(Enum):
    """Types of extractable entities from scientific papers"""
    FACT = "fact"
    HYPOTHESIS = "hypothesis"
    EXPERIMENT = "experiment"
    TECHNIQUE = "technique"
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
