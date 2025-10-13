"""
Article Registry - Manages downloaded papers metadata and tracks which PDFs have been downloaded
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict


@dataclass
class ArticleRecord:
    """Record of a downloaded article"""
    pmid: Optional[str] = None
    pmc_id: Optional[str] = None
    arxiv_id: Optional[str] = None
    doi: Optional[str] = None
    title: str = ""
    authors: List[str] = None
    journal: str = ""
    publication_date: str = ""
    downloaded_at: str = ""
    pdf_path: Optional[str] = None
    file_size: int = 0
    download_source: str = ""  # e.g., "PMC", "DOI", "ArXiv"

    def __post_init__(self):
        if self.authors is None:
            self.authors = []


class ArticleRegistry:
    """Manages registry of downloaded articles"""

    def __init__(self, registry_path: str = "articles/metadata.json"):
        """
        Initialize article registry

        Args:
            registry_path: Path to the registry JSON file
        """
        self.registry_path = Path(registry_path)
        self.registry_dir = self.registry_path.parent
        self._registry: Dict[str, Dict[str, Any]] = {}
        self._load_registry()

    def _load_registry(self):
        """Load registry from disk"""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r', encoding='utf-8') as f:
                    self._registry = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse {self.registry_path}, starting fresh")
                self._registry = {}
        else:
            # Create directory if it doesn't exist
            self.registry_dir.mkdir(parents=True, exist_ok=True)
            self._registry = {}

    def _save_registry(self):
        """Save registry to disk"""
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, 'w', encoding='utf-8') as f:
            json.dump(self._registry, f, indent=2, ensure_ascii=False)

    def add(self, record: ArticleRecord) -> bool:
        """
        Add article record to registry

        Args:
            record: ArticleRecord to add

        Returns:
            bool: True if added, False if already exists
        """
        # Use PMID as primary key, fall back to arXiv ID, PMC ID or DOI
        key = record.pmid or record.arxiv_id or record.pmc_id or record.doi

        if not key:
            raise ValueError("Record must have at least one of: pmid, arxiv_id, pmc_id, doi")

        if key in self._registry:
            return False  # Already exists

        # Set download timestamp if not set
        if not record.downloaded_at:
            record.downloaded_at = datetime.now().isoformat()

        self._registry[key] = asdict(record)
        self._save_registry()
        return True

    def get(self, identifier: str) -> Optional[ArticleRecord]:
        """
        Get article record by any identifier (PMID, arXiv ID, PMC ID, or DOI)

        Args:
            identifier: PMID, arXiv ID, PMC ID, or DOI

        Returns:
            ArticleRecord if found, None otherwise
        """
        # Direct lookup
        if identifier in self._registry:
            return ArticleRecord(**self._registry[identifier])

        # Search by alternative identifiers
        for key, record_dict in self._registry.items():
            if (record_dict.get('pmid') == identifier or
                record_dict.get('arxiv_id') == identifier or
                record_dict.get('pmc_id') == identifier or
                record_dict.get('doi') == identifier):
                return ArticleRecord(**record_dict)

        return None

    def exists(self, identifier: str) -> bool:
        """
        Check if article exists in registry

        Args:
            identifier: PMID, arXiv ID, PMC ID, or DOI

        Returns:
            bool: True if exists
        """
        return self.get(identifier) is not None

    def update(self, identifier: str, **kwargs):
        """
        Update article record

        Args:
            identifier: PMID, arXiv ID, PMC ID, or DOI
            **kwargs: Fields to update
        """
        record = self.get(identifier)
        if not record:
            raise ValueError(f"Article {identifier} not found in registry")

        # Get the key
        key = record.pmid or record.arxiv_id or record.pmc_id or record.doi

        # Update fields
        for field, value in kwargs.items():
            if hasattr(record, field):
                self._registry[key][field] = value

        self._save_registry()

    def list_all(self) -> List[ArticleRecord]:
        """
        Get all article records

        Returns:
            List of ArticleRecord objects
        """
        return [ArticleRecord(**record_dict) for record_dict in self._registry.values()]

    def count(self) -> int:
        """Get total number of downloaded articles"""
        return len(self._registry)

    def get_by_source(self, source: str) -> List[ArticleRecord]:
        """
        Get articles by download source

        Args:
            source: Download source (e.g., "PMC", "DOI")

        Returns:
            List of matching ArticleRecord objects
        """
        return [
            ArticleRecord(**record_dict)
            for record_dict in self._registry.values()
            if record_dict.get('download_source') == source
        ]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get registry statistics

        Returns:
            Dictionary with statistics
        """
        records = self.list_all()

        total_size = sum(r.file_size for r in records)
        sources = {}
        for r in records:
            source = r.download_source or "unknown"
            sources[source] = sources.get(source, 0) + 1

        return {
            "total_articles": len(records),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "by_source": sources,
            "with_pmc_id": sum(1 for r in records if r.pmc_id),
            "with_arxiv_id": sum(1 for r in records if r.arxiv_id),
            "with_doi": sum(1 for r in records if r.doi),
        }

    def clear(self):
        """Clear all records (use with caution!)"""
        self._registry = {}
        self._save_registry()
