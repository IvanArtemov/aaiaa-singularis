"""Base pipeline for entity extraction"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import time
import uuid

from src.models import ExtractionResult, PipelineMetrics


class BasePipeline(ABC):
    """
    Abstract base class for all extraction pipelines

    All pipelines must implement the extract() method and provide
    metadata about their performance characteristics.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize pipeline

        Args:
            config: Configuration dictionary with pipeline-specific settings
        """
        self.config = config
        self.name = self.__class__.__name__
        self.last_metrics = None

    @abstractmethod
    def extract(self, paper_text: str, paper_id: str, metadata: Optional[Dict[str, Any]] = None) -> ExtractionResult:
        """
        Extract entities and relationships from paper text

        Args:
            paper_text: Full text of the scientific paper
            paper_id: Unique identifier for the paper
            metadata: Optional metadata about the paper (title, authors, etc.)

        Returns:
            ExtractionResult with extracted entities, relationships, and metrics

        Raises:
            ValueError: If paper_text is empty or invalid
        """
        pass

    @abstractmethod
    def get_description(self) -> str:
        """
        Get human-readable description of the pipeline

        Returns:
            Description string for UI display
        """
        pass

    @abstractmethod
    def get_estimated_cost(self) -> float:
        """
        Get estimated cost per paper in USD

        Returns:
            Estimated cost per paper
        """
        pass

    @property
    def version(self) -> str:
        """Pipeline version"""
        return "1.0.0"

    def _generate_entity_id(self, entity_type: str) -> str:
        """
        Generate unique entity ID

        Args:
            entity_type: Type of entity (fact, hypothesis, etc.)

        Returns:
            Unique ID string
        """
        short_uuid = str(uuid.uuid4())[:8]
        return f"{entity_type}_{short_uuid}"

    def _generate_relationship_id(self) -> str:
        """
        Generate unique relationship ID

        Returns:
            Unique ID string
        """
        short_uuid = str(uuid.uuid4())[:8]
        return f"rel_{short_uuid}"

    def _validate_paper_text(self, paper_text: str) -> None:
        """
        Validate paper text

        Args:
            paper_text: Text to validate

        Raises:
            ValueError: If text is invalid
        """
        if not paper_text or not paper_text.strip():
            raise ValueError("Paper text cannot be empty")

        if len(paper_text) < 100:
            raise ValueError("Paper text too short (minimum 100 characters)")

    def get_last_metrics(self) -> Optional[PipelineMetrics]:
        """
        Get metrics from last extraction

        Returns:
            PipelineMetrics if available, None otherwise
        """
        return self.last_metrics

    def __str__(self) -> str:
        """String representation"""
        return f"{self.name}(version={self.version})"

    def __repr__(self) -> str:
        """Detailed representation"""
        return f"{self.name}(version={self.version}, cost=${self.get_estimated_cost():.4f}/paper)"
