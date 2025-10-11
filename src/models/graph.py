"""Knowledge graph representation and visualization"""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
import json

from .entities import Entity, Relationship, EntityType, RelationshipType


@dataclass
class KnowledgeGraph:
    """
    Knowledge graph for a scientific paper

    Attributes:
        paper_id: Unique identifier for the paper
        entities: List of all entities in the graph
        relationships: List of all relationships in the graph
        metadata: Additional metadata about the graph
    """
    paper_id: str
    entities: List[Entity]
    relationships: List[Relationship]
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID"""
        for entity in self.entities:
            if entity.id == entity_id:
                return entity
        return None

    def get_entities_by_type(self, entity_type: EntityType) -> List[Entity]:
        """Get all entities of a specific type"""
        return [e for e in self.entities if e.type == entity_type]

    def get_relationships_for_entity(
        self,
        entity_id: str,
        direction: str = "both"
    ) -> List[Relationship]:
        """
        Get relationships involving an entity

        Args:
            entity_id: Entity ID
            direction: "in" (incoming), "out" (outgoing), or "both"

        Returns:
            List of relationships
        """
        if direction == "in":
            return [r for r in self.relationships if r.target_id == entity_id]
        elif direction == "out":
            return [r for r in self.relationships if r.source_id == entity_id]
        else:  # both
            return [
                r for r in self.relationships
                if r.source_id == entity_id or r.target_id == entity_id
            ]

    def get_neighbors(self, entity_id: str) -> Set[str]:
        """Get IDs of all neighboring entities"""
        neighbors = set()
        for rel in self.relationships:
            if rel.source_id == entity_id:
                neighbors.add(rel.target_id)
            elif rel.target_id == entity_id:
                neighbors.add(rel.source_id)
        return neighbors

    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary"""
        return {
            "paper_id": self.paper_id,
            "entities": [entity.to_dict() for entity in self.entities],
            "relationships": [rel.to_dict() for rel in self.relationships],
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeGraph":
        """Create graph from dictionary"""
        entities = [Entity.from_dict(e) for e in data["entities"]]
        relationships = [Relationship.from_dict(r) for r in data["relationships"]]

        return cls(
            paper_id=data["paper_id"],
            entities=entities,
            relationships=relationships,
            metadata=data.get("metadata", {})
        )

    def to_json(self, filepath: Optional[str] = None) -> str:
        """
        Convert graph to JSON

        Args:
            filepath: If provided, save to file

        Returns:
            JSON string
        """
        json_str = json.dumps(self.to_dict(), indent=2)

        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(json_str)

        return json_str

    @classmethod
    def from_json(
        cls,
        json_str: Optional[str] = None,
        filepath: Optional[str] = None
    ) -> "KnowledgeGraph":
        """
        Create graph from JSON

        Args:
            json_str: JSON string
            filepath: Path to JSON file

        Returns:
            KnowledgeGraph instance
        """
        if filepath:
            with open(filepath, 'r', encoding='utf-8') as f:
                json_str = f.read()

        if not json_str:
            raise ValueError("Either json_str or filepath must be provided")

        data = json.loads(json_str)
        return cls.from_dict(data)

    def to_networkx(self):
        """
        Convert to NetworkX DiGraph

        Returns:
            NetworkX DiGraph with nodes and edges

        Raises:
            ImportError: If networkx is not installed
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("networkx is required. Install with: pip install networkx")

        G = nx.DiGraph()

        # Add nodes with attributes
        for entity in self.entities:
            G.add_node(
                entity.id,
                type=entity.type.value,
                text=entity.text,
                confidence=entity.confidence,
                source_section=entity.source_section,
                metadata=entity.metadata
            )

        # Add edges with attributes
        for rel in self.relationships:
            G.add_edge(
                rel.source_id,
                rel.target_id,
                relationship_type=rel.relationship_type.value,
                confidence=rel.confidence,
                metadata=rel.metadata
            )

        return G

    def to_cytoscape(self) -> Dict[str, Any]:
        """
        Convert to Cytoscape.js format

        Returns:
            Dictionary in Cytoscape.js format
        """
        elements = []

        # Add nodes
        for entity in self.entities:
            elements.append({
                "data": {
                    "id": entity.id,
                    "label": entity.text[:50],  # Truncate for display
                    "type": entity.type.value,
                    "text": entity.text,
                    "confidence": entity.confidence,
                    "source_section": entity.source_section
                }
            })

        # Add edges
        for rel in self.relationships:
            elements.append({
                "data": {
                    "source": rel.source_id,
                    "target": rel.target_id,
                    "label": rel.relationship_type.value,
                    "confidence": rel.confidence
                }
            })

        return {"elements": elements}

    def to_svg(self, output_path: str, layout: str = "hierarchical"):
        """
        Export graph to SVG visualization

        Args:
            output_path: Path to save SVG file
            layout: Layout algorithm ("hierarchical", "spring", "circular")

        Raises:
            ImportError: If required visualization libraries are not installed
        """
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "networkx and matplotlib are required. "
                "Install with: pip install networkx matplotlib"
            )

        G = self.to_networkx()

        # Choose layout
        if layout == "hierarchical":
            # Group by entity type for hierarchical layout
            pos = self._hierarchical_layout(G)
        elif layout == "spring":
            pos = nx.spring_layout(G, k=2, iterations=50)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        else:
            pos = nx.kamada_kawai_layout(G)

        # Create figure
        plt.figure(figsize=(20, 14))

        # Color map for entity types
        color_map = {
            "fact": "#b2ebf2",
            "hypothesis": "#ffe082",
            "experiment": "#f48fb1",
            "technique": "#f48fb1",
            "result": "#a5d6a7",
            "dataset": "#ce93d8",
            "analysis": "#80cbc4",
            "conclusion": "#b0bec5"
        }

        # Get node colors
        node_colors = [
            color_map.get(G.nodes[node].get("type", ""), "#cccccc")
            for node in G.nodes()
        ]

        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=3000,
            alpha=0.9,
            edgecolors='#495057',
            linewidths=1.5
        )

        # Draw edges
        nx.draw_networkx_edges(
            G, pos,
            edge_color='#666666',
            width=2,
            alpha=0.6,
            arrows=True,
            arrowsize=20,
            arrowstyle='->'
        )

        # Draw labels
        labels = {
            node: G.nodes[node].get("text", "")[:30] + "..."
            if len(G.nodes[node].get("text", "")) > 30
            else G.nodes[node].get("text", "")
            for node in G.nodes()
        }

        nx.draw_networkx_labels(
            G, pos,
            labels=labels,
            font_size=8,
            font_weight='bold'
        )

        plt.title(f"Knowledge Graph: {self.paper_id}", fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, format='svg', dpi=300, bbox_inches='tight')
        plt.close()

    def _hierarchical_layout(self, G) -> Dict[str, tuple]:
        """Create hierarchical layout based on entity types"""
        # Define hierarchy levels
        type_levels = {
            "fact": 0,
            "hypothesis": 1,
            "experiment": 2,
            "technique": 2,
            "result": 3,
            "analysis": 4,
            "dataset": 2,
            "conclusion": 5
        }

        pos = {}
        level_counts = {}

        # Group nodes by level
        for node in G.nodes():
            node_type = G.nodes[node].get("type", "")
            level = type_levels.get(node_type, 3)

            if level not in level_counts:
                level_counts[level] = 0

            # Position nodes
            x = level_counts[level]
            y = -level  # Negative for top-to-bottom

            pos[node] = (x, y)
            level_counts[level] += 1

        return pos

    def statistics(self) -> Dict[str, Any]:
        """Get graph statistics"""
        entity_counts = {}
        for entity in self.entities:
            entity_type = entity.type.value
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1

        relationship_counts = {}
        for rel in self.relationships:
            rel_type = rel.relationship_type.value
            relationship_counts[rel_type] = relationship_counts.get(rel_type, 0) + 1

        return {
            "total_entities": len(self.entities),
            "total_relationships": len(self.relationships),
            "entity_counts": entity_counts,
            "relationship_counts": relationship_counts,
            "avg_relationships_per_entity": (
                len(self.relationships) / len(self.entities)
                if self.entities else 0
            )
        }

    def __str__(self) -> str:
        """String representation"""
        return (
            f"KnowledgeGraph(paper_id={self.paper_id}, "
            f"entities={len(self.entities)}, "
            f"relationships={len(self.relationships)})"
        )
