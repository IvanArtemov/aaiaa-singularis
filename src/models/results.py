"""Data models for extraction results and pipeline metrics"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

from .entities import Entity, Relationship


@dataclass
class PipelineMetrics:
    """
    Metrics for pipeline performance and cost

    Attributes:
        processing_time: Time taken to process (seconds)
        tokens_used: Total tokens consumed (input + output)
        cost_usd: Cost in USD
        entities_extracted: Number of entities extracted
        relationships_extracted: Number of relationships extracted
        memory_used_mb: Memory used in MB
        metadata: Additional metrics
    """
    processing_time: float
    tokens_used: int = 0
    cost_usd: float = 0.0
    entities_extracted: int = 0
    relationships_extracted: int = 0
    memory_used_mb: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            "processing_time": self.processing_time,
            "tokens_used": self.tokens_used,
            "cost_usd": self.cost_usd,
            "entities_extracted": self.entities_extracted,
            "relationships_extracted": self.relationships_extracted,
            "memory_used_mb": self.memory_used_mb,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineMetrics":
        """Create metrics from dictionary"""
        return cls(
            processing_time=data["processing_time"],
            tokens_used=data.get("tokens_used", 0),
            cost_usd=data.get("cost_usd", 0.0),
            entities_extracted=data.get("entities_extracted", 0),
            relationships_extracted=data.get("relationships_extracted", 0),
            memory_used_mb=data.get("memory_used_mb", 0.0),
            metadata=data.get("metadata", {})
        )

    def to_json(self) -> str:
        """Convert metrics to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    def __str__(self) -> str:
        """String representation"""
        return (
            f"PipelineMetrics(time={self.processing_time:.2f}s, "
            f"cost=${self.cost_usd:.4f}, "
            f"entities={self.entities_extracted}, "
            f"relationships={self.relationships_extracted})"
        )


@dataclass
class ExtractionResult:
    """
    Result of entity extraction from a paper

    Attributes:
        paper_id: Unique identifier for the paper
        entities: Dictionary mapping entity types to lists of entities
        relationships: List of relationships between entities
        metrics: Pipeline performance metrics
        metadata: Additional metadata (paper title, authors, etc.)
        timestamp: When the extraction was performed
    """
    paper_id: str
    entities: Dict[str, List[Entity]]
    relationships: List[Relationship]
    metrics: PipelineMetrics
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def get_entities_by_type(self, entity_type: str) -> List[Entity]:
        """
        Get all entities of a specific type

        Args:
            entity_type: Type of entity (e.g., "hypothesis", "result")

        Returns:
            List of entities of that type
        """
        return self.entities.get(entity_type, [])

    def get_entity_by_id(self, entity_id: str) -> Optional[Entity]:
        """
        Get entity by ID

        Args:
            entity_id: Entity ID to search for

        Returns:
            Entity if found, None otherwise
        """
        for entities_list in self.entities.values():
            for entity in entities_list:
                if entity.id == entity_id:
                    return entity
        return None

    def get_relationships_for_entity(self, entity_id: str) -> List[Relationship]:
        """
        Get all relationships involving an entity

        Args:
            entity_id: Entity ID

        Returns:
            List of relationships where entity is source or target
        """
        return [
            rel for rel in self.relationships
            if rel.source_id == entity_id or rel.target_id == entity_id
        ]

    def total_entities(self) -> int:
        """Get total number of entities extracted"""
        return sum(len(entities) for entities in self.entities.values())

    def total_relationships(self) -> int:
        """Get total number of relationships extracted"""
        return len(self.relationships)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            "paper_id": self.paper_id,
            "entities": {
                entity_type: [entity.to_dict() for entity in entities_list]
                for entity_type, entities_list in self.entities.items()
            },
            "relationships": [rel.to_dict() for rel in self.relationships],
            "metrics": self.metrics.to_dict(),
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtractionResult":
        """Create result from dictionary"""
        entities = {
            entity_type: [Entity.from_dict(e) for e in entities_list]
            for entity_type, entities_list in data["entities"].items()
        }

        relationships = [
            Relationship.from_dict(rel)
            for rel in data["relationships"]
        ]

        metrics = PipelineMetrics.from_dict(data["metrics"])

        timestamp = datetime.fromisoformat(data["timestamp"])

        return cls(
            paper_id=data["paper_id"],
            entities=entities,
            relationships=relationships,
            metrics=metrics,
            metadata=data.get("metadata", {}),
            timestamp=timestamp
        )

    def to_json(self, filepath: Optional[str] = None) -> str:
        """
        Convert result to JSON string

        Args:
            filepath: If provided, save JSON to file

        Returns:
            JSON string
        """
        json_str = json.dumps(self.to_dict(), indent=2)

        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(json_str)

        return json_str

    @classmethod
    def from_json(cls, json_str: Optional[str] = None, filepath: Optional[str] = None) -> "ExtractionResult":
        """
        Create result from JSON

        Args:
            json_str: JSON string
            filepath: Path to JSON file (alternative to json_str)

        Returns:
            ExtractionResult instance
        """
        if filepath:
            with open(filepath, 'r', encoding='utf-8') as f:
                json_str = f.read()

        if not json_str:
            raise ValueError("Either json_str or filepath must be provided")

        data = json.loads(json_str)
        return cls.from_dict(data)

    def __str__(self) -> str:
        """String representation"""
        return (
            f"ExtractionResult(paper_id={self.paper_id}, "
            f"entities={self.total_entities()}, "
            f"relationships={self.total_relationships()})"
        )
