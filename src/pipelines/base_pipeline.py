"""Base pipeline for entity extraction"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import time
import uuid

from src.models import ExtractionResult, PipelineMetrics
from src.parsers.base_parser import ParsedDocument


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
    def extract(self, parsed_doc: ParsedDocument, paper_id: str) -> ExtractionResult:
        """
        Extract entities and relationships from parsed document

        Args:
            parsed_doc: ParsedDocument with text, sections, and metadata
            paper_id: Unique identifier for the paper

        Returns:
            ExtractionResult with extracted entities, relationships, and metrics

        Raises:
            ValueError: If parsed_doc is invalid
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

    def _validate_parsed_doc(self, parsed_doc: ParsedDocument) -> None:
        """
        Validate parsed document

        Args:
            parsed_doc: Document to validate

        Raises:
            ValueError: If document is invalid
        """
        if not parsed_doc:
            raise ValueError("Parsed document cannot be None")

        if not parsed_doc.text or not parsed_doc.text.strip():
            raise ValueError("Parsed document text cannot be empty")

        if len(parsed_doc.text) < 100:
            raise ValueError("Parsed document text too short (minimum 100 characters)")

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
