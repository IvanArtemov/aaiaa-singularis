"""Semantic retriever for vector-based entity candidate search"""

import os
from typing import List, Dict, Optional, Any
import numpy as np

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

from src.models import EntityType, Sentence


class SemanticRetriever:
    """
    Vector-based semantic retriever for entity candidates

    Uses ChromaDB for efficient similarity search over document segments.
    Retrieves top-k candidates for each entity type based on keyword embeddings.

    Cost: FREE (local ChromaDB)
    """

    def __init__(
        self,
        collection_name: str = "paper_segments",
        persist_directory: str = "./chroma_db",
        distance_metric: str = "cosine"
    ):
        """
        Initialize semantic retriever

        Args:
            collection_name: Name of ChromaDB collection
            persist_directory: Directory to persist ChromaDB data
            distance_metric: Distance metric ("cosine", "l2", "ip")

        Raises:
            ImportError: If chromadb is not installed
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "chromadb is not installed. Install with: pip install chromadb"
            )

        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.distance_metric = distance_metric

        # Create persist directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": distance_metric}
        )

        # Metrics
        self.total_queries = 0
        self.total_results = 0

    def index_segments(
        self,
        sentences: List[Sentence],
        paper_id: str
    ):
        """
        Index document segments (sentences) in vector database

        Args:
            sentences: List of Sentence objects with embeddings
            paper_id: Unique paper identifier

        Raises:
            ValueError: If sentences don't have embeddings
        """
        if not sentences:
            return

        # Validate that all sentences have embeddings
        if any(s.embedding is None for s in sentences):
            raise ValueError("All sentences must have embeddings before indexing")

        # Prepare data for ChromaDB
        embeddings = []
        documents = []
        metadatas = []
        ids = []

        for i, sentence in enumerate(sentences):
            # Convert numpy array to list if needed
            if isinstance(sentence.embedding, np.ndarray):
                embedding = sentence.embedding.tolist()
            else:
                embedding = sentence.embedding

            embeddings.append(embedding)
            documents.append(sentence.text)
            metadatas.append({
                "paper_id": paper_id,
                "section": sentence.section,
                "position": sentence.position,
                "char_start": sentence.char_start,
                "char_end": sentence.char_end
            })
            ids.append(f"{paper_id}_seg_{i}")

        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def retrieve_candidates(
        self,
        query_embeddings: List[List[float]],
        entity_type: EntityType,
        top_k: int = 20,
        section_filter: Optional[List[str]] = None,
        paper_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve candidate segments for an entity type using semantic search

        Args:
            query_embeddings: List of query embeddings (from keywords)
            entity_type: Type of entity to retrieve candidates for
            top_k: Number of top candidates to retrieve per query
            section_filter: Optional list of sections to filter by
            paper_id: Optional paper ID to restrict search to

        Returns:
            List of candidate dictionaries with text, section, distance, etc.
        """
        if not query_embeddings:
            return []

        self.total_queries += len(query_embeddings)

        all_candidates = []
        seen_ids = set()

        # Query for each embedding
        for query_emb in query_embeddings:
            # Build where filter if needed
            # ChromaDB requires $and operator when combining multiple conditions
            where_filter = None

            if section_filter and paper_id:
                # Both filters: use $and operator
                where_filter = {
                    "$and": [
                        {"section": {"$in": section_filter}},
                        {"paper_id": paper_id}
                    ]
                }
            elif section_filter:
                # Only section filter
                where_filter = {"section": {"$in": section_filter}}
            elif paper_id:
                # Only paper_id filter
                where_filter = {"paper_id": paper_id}

            # Query ChromaDB
            try:
                results = self.collection.query(
                    query_embeddings=[query_emb],
                    n_results=top_k,
                    where=where_filter
                )

                # Process results
                if results and results["ids"] and results["ids"][0]:
                    for i in range(len(results["ids"][0])):
                        result_id = results["ids"][0][i]

                        # Skip duplicates
                        if result_id in seen_ids:
                            continue
                        seen_ids.add(result_id)

                        candidate = {
                            "id": result_id,
                            "text": results["documents"][0][i],
                            "distance": results["distances"][0][i] if results.get("distances") else 0.0,
                            "metadata": results["metadatas"][0][i],
                            "entity_type": entity_type
                        }
                        all_candidates.append(candidate)

            except Exception as e:
                print(f"Warning: Query failed for entity type {entity_type}: {e}")
                continue

        self.total_results += len(all_candidates)

        # Sort by distance (lower is better for cosine)
        all_candidates.sort(key=lambda x: x["distance"])

        # Return top-k unique candidates
        return all_candidates[:top_k]

    def retrieve_by_keywords(
        self,
        keywords: List[str],
        keyword_embeddings: List[List[float]],
        entity_type: EntityType,
        top_k: int = 20,
        section_filter: Optional[List[str]] = None,
        paper_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve candidates using keyword embeddings

        Convenience method that wraps retrieve_candidates

        Args:
            keywords: List of keyword strings (for reference)
            keyword_embeddings: Embeddings for keywords
            entity_type: Entity type
            top_k: Number of results
            section_filter: Optional section filter
            paper_id: Optional paper ID filter

        Returns:
            List of candidate dictionaries
        """
        return self.retrieve_candidates(
            query_embeddings=keyword_embeddings,
            entity_type=entity_type,
            top_k=top_k,
            section_filter=section_filter,
            paper_id=paper_id
        )

    def clear_collection(self):
        """Clear all data from the collection"""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.distance_metric}
            )
        except Exception as e:
            print(f"Warning: Failed to clear collection: {e}")

    def clear_paper(self, paper_id: str):
        """
        Remove all segments for a specific paper

        Args:
            paper_id: Paper ID to remove
        """
        try:
            # Get all IDs for this paper
            results = self.collection.get(
                where={"paper_id": paper_id}
            )

            if results and results["ids"]:
                self.collection.delete(ids=results["ids"])

        except Exception as e:
            print(f"Warning: Failed to clear paper {paper_id}: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get retriever metrics

        Returns:
            Dictionary with metrics
        """
        collection_count = self.collection.count()

        return {
            "total_queries": self.total_queries,
            "total_results": self.total_results,
            "avg_results_per_query": (
                self.total_results / self.total_queries
                if self.total_queries > 0 else 0
            ),
            "collection_size": collection_count,
            "collection_name": self.collection_name
        }

    def reset_metrics(self):
        """Reset metrics counters"""
        self.total_queries = 0
        self.total_results = 0
