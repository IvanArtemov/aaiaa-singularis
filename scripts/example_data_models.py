"""Example usage of data models"""

from src.models import (
    Entity,
    Relationship,
    EntityType,
    RelationshipType,
    ExtractionResult,
    PipelineMetrics,
    KnowledgeGraph
)
from datetime import datetime


def main():
    print("=" * 80)
    print("DATA MODELS EXAMPLE")
    print("=" * 80)

    # Create some entities
    fact1 = Entity(
        id="fact_001",
        type=EntityType.FACT,
        text="COPD is a chronic inflammatory condition characterized by fixed airflow obstruction",
        confidence=0.95,
        source_section="introduction"
    )

    hypothesis1 = Entity(
        id="hyp_001",
        type=EntityType.HYPOTHESIS,
        text="Airway inflammation and remodeling differ between COPD and healthy controls",
        confidence=0.90,
        source_section="introduction"
    )

    method1 = Entity(
        id="method_001",
        type=EntityType.TECHNIQUE,
        text="Bronchoscopy with BAL and endobronchial biopsies; GMA embedding; IHC quantification",
        confidence=0.98,
        source_section="methods"
    )

    result1 = Entity(
        id="result_001",
        type=EntityType.RESULT,
        text="COPD biopsies show higher macrophages and eosinophils, and greater submucosal microvascular area vs controls",
        confidence=0.92,
        source_section="results"
    )

    conclusion1 = Entity(
        id="conclusion_001",
        type=EntityType.CONCLUSION,
        text="Macrophage/eosinophil increases and microvascularity distinguish COPD from non-COPD in bronchial biopsies",
        confidence=0.93,
        source_section="conclusion"
    )

    entities_list = [fact1, hypothesis1, method1, result1, conclusion1]

    # Create relationships
    rel1 = Relationship(
        source_id="fact_001",
        target_id="hyp_001",
        relationship_type=RelationshipType.FACT_TO_HYPOTHESIS,
        confidence=0.85
    )

    rel2 = Relationship(
        source_id="hyp_001",
        target_id="method_001",
        relationship_type=RelationshipType.HYPOTHESIS_TO_METHOD,
        confidence=0.90
    )

    rel3 = Relationship(
        source_id="method_001",
        target_id="result_001",
        relationship_type=RelationshipType.METHOD_TO_RESULT,
        confidence=0.95
    )

    rel4 = Relationship(
        source_id="result_001",
        target_id="conclusion_001",
        relationship_type=RelationshipType.RESULT_TO_CONCLUSION,
        confidence=0.88
    )

    relationships_list = [rel1, rel2, rel3, rel4]

    # Display entities
    print("\n📦 ENTITIES")
    print("-" * 80)
    for entity in entities_list:
        print(f"\n[{entity.type.value.upper()}]")
        print(f"ID: {entity.id}")
        print(f"Text: {entity.text}")
        print(f"Confidence: {entity.confidence}")
        print(f"Section: {entity.source_section}")

    # Display relationships
    print("\n\n🔗 RELATIONSHIPS")
    print("-" * 80)
    for rel in relationships_list:
        print(f"{rel.source_id} --[{rel.relationship_type.value}]--> {rel.target_id} (conf: {rel.confidence})")

    # Create metrics
    metrics = PipelineMetrics(
        processing_time=15.3,
        tokens_used=5420,
        cost_usd=0.027,
        entities_extracted=5,
        relationships_extracted=4,
        memory_used_mb=128.5
    )

    print("\n\n📊 PIPELINE METRICS")
    print("-" * 80)
    print(metrics)

    # Create extraction result
    entities_by_type = {
        "fact": [fact1],
        "hypothesis": [hypothesis1],
        "technique": [method1],
        "result": [result1],
        "conclusion": [conclusion1]
    }

    result = ExtractionResult(
        paper_id="copd_study_2025",
        entities=entities_by_type,
        relationships=relationships_list,
        metrics=metrics,
        metadata={
            "title": "Inflammatory cells and remodeling in bronchial biopsies from COPD patients",
            "authors": ["Eagan TM", "Nielsen R", "Haaland I"],
            "year": 2025
        }
    )

    print("\n\n📄 EXTRACTION RESULT")
    print("-" * 80)
    print(result)
    print(f"Total entities: {result.total_entities()}")
    print(f"Total relationships: {result.total_relationships()}")

    # Create knowledge graph
    graph = KnowledgeGraph(
        paper_id="copd_study_2025",
        entities=entities_list,
        relationships=relationships_list,
        metadata={"source": "example"}
    )

    print("\n\n🕸️  KNOWLEDGE GRAPH")
    print("-" * 80)
    print(graph)

    # Graph statistics
    stats = graph.statistics()
    print("\nGraph Statistics:")
    print(f"  Total entities: {stats['total_entities']}")
    print(f"  Total relationships: {stats['total_relationships']}")
    print(f"  Avg relationships per entity: {stats['avg_relationships_per_entity']:.2f}")
    print("\nEntity counts:")
    for entity_type, count in stats['entity_counts'].items():
        print(f"  {entity_type}: {count}")

    # Save to JSON
    print("\n\n💾 SAVING TO JSON")
    print("-" * 80)
    result.to_json("example_result.json")
    print("✅ Saved to: example_result.json")

    graph.to_json("example_graph.json")
    print("✅ Saved to: example_graph.json")

    # Load back from JSON
    print("\n\n📥 LOADING FROM JSON")
    print("-" * 80)
    loaded_result = ExtractionResult.from_json(filepath="example_result.json")
    print(f"✅ Loaded result: {loaded_result}")

    loaded_graph = KnowledgeGraph.from_json(filepath="example_graph.json")
    print(f"✅ Loaded graph: {loaded_graph}")

    print("\n" + "=" * 80)
    print("✨ DATA MODELS DEMONSTRATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
