"""Tests for Hybrid Pipeline"""

import pytest
from src.pipelines import HybridPipeline
from src.models import EntityType


# Sample scientific paper text for testing
SAMPLE_PAPER = """
Introduction

Previous studies have shown that aging is associated with cellular senescence.
The role of mitochondrial dysfunction in aging has been well established.

We hypothesize that targeting senescent cells can slow down the aging process.
The aim of this study is to investigate whether senolytics can extend lifespan.

Materials and Methods

We used flow cytometry to measure cellular markers.
Participants were recruited from the local community.
The statistical analysis was performed using R software.
Data were obtained from the GEO database (accession: GSE12345).

We performed an experiment to test the effect of senolytics on lifespan.
The experimental design included treatment and control groups.

Results

We found that senolytic treatment significantly increased lifespan (p < 0.01).
The mean lifespan was 25% higher in the treatment group compared to control.
Significant differences were observed between groups (p < 0.001).

Discussion

These results suggest that targeting senescent cells is a promising strategy.

Conclusion

Further studies are needed to validate these findings in human populations.
"""


@pytest.fixture
def hybrid_pipeline():
    """Create hybrid pipeline for testing"""
    config = {
        "pattern_confidence_threshold": 0.7,
        "nlp_confidence_threshold": 0.6
    }
    return HybridPipeline(
        config=config,
        use_llm_fallback=False,  # Disable LLM to avoid API costs in tests
        confidence_threshold=0.75
    )


@pytest.fixture
def hybrid_pipeline_with_llm():
    """Create hybrid pipeline with LLM fallback for integration tests"""
    config = {
        "pattern_confidence_threshold": 0.7,
        "nlp_confidence_threshold": 0.6
    }
    return HybridPipeline(
        config=config,
        use_llm_fallback=True,
        confidence_threshold=0.75,
        llm_provider="openai",
        llm_model="gpt-4o-mini"
    )


def test_pipeline_initialization(hybrid_pipeline):
    """Test pipeline initialization"""
    assert hybrid_pipeline is not None
    assert hybrid_pipeline.name == "HybridPipeline"
    assert hybrid_pipeline.hypothesis_extractor is not None
    assert hybrid_pipeline.method_extractor is not None


def test_basic_extraction(hybrid_pipeline):
    """Test basic extraction without LLM"""
    result = hybrid_pipeline.extract(
        paper_text=SAMPLE_PAPER,
        paper_id="test_001",
        metadata={"title": "Test Paper"}
    )

    # Check result structure
    assert result is not None
    assert result.paper_id == "test_001"
    assert result.metadata["title"] == "Test Paper"

    # Check that some entities were extracted
    total_entities = result.total_entities()
    assert total_entities > 0, "No entities were extracted"

    # Check metrics
    assert result.metrics.processing_time > 0
    assert result.metrics.entities_extracted == total_entities


def test_pattern_extraction(hybrid_pipeline):
    """Test pattern-based extraction"""
    result = hybrid_pipeline.extract(
        paper_text=SAMPLE_PAPER,
        paper_id="test_002"
    )

    # Check for hypotheses
    hypotheses = result.get_entities_by_type("hypothesis")
    assert len(hypotheses) > 0, "No hypotheses extracted"

    # Check for methods
    techniques = result.get_entities_by_type("technique")
    assert len(techniques) > 0, "No techniques/methods extracted"

    # Check for results
    results = result.get_entities_by_type("result")
    assert len(results) > 0, "No results extracted"


def test_entity_types(hybrid_pipeline):
    """Test that different entity types are extracted"""
    result = hybrid_pipeline.extract(
        paper_text=SAMPLE_PAPER,
        paper_id="test_003"
    )

    # Collect all entity types
    entity_types = set(result.entities.keys())

    # Should have multiple entity types
    assert len(entity_types) >= 2, f"Only extracted {entity_types}"

    # Check for key entity types
    expected_types = ["hypothesis", "technique", "result"]
    found_types = [t for t in expected_types if t in entity_types]
    assert len(found_types) >= 2, f"Expected {expected_types}, found {found_types}"


def test_relationships(hybrid_pipeline):
    """Test relationship building"""
    result = hybrid_pipeline.extract(
        paper_text=SAMPLE_PAPER,
        paper_id="test_004"
    )

    # Check that relationships were created
    assert len(result.relationships) > 0, "No relationships created"

    # Check relationship structure
    for rel in result.relationships:
        assert rel.source_id is not None
        assert rel.target_id is not None
        assert rel.relationship_type is not None
        assert 0.0 <= rel.confidence <= 1.0


def test_confidence_scores(hybrid_pipeline):
    """Test that entities have valid confidence scores"""
    result = hybrid_pipeline.extract(
        paper_text=SAMPLE_PAPER,
        paper_id="test_005"
    )

    for entity_type, entities in result.entities.items():
        for entity in entities:
            assert 0.0 <= entity.confidence <= 1.0, \
                f"Invalid confidence {entity.confidence} for {entity_type}"


def test_cost_tracking(hybrid_pipeline):
    """Test cost tracking (should be $0 without LLM)"""
    result = hybrid_pipeline.extract(
        paper_text=SAMPLE_PAPER,
        paper_id="test_006"
    )

    # Without LLM, cost should be 0
    assert result.metrics.cost_usd == 0.0
    assert result.metrics.tokens_used == 0


def test_extraction_methods_tracking(hybrid_pipeline):
    """Test tracking of which methods extracted entities"""
    result = hybrid_pipeline.extract(
        paper_text=SAMPLE_PAPER,
        paper_id="test_007"
    )

    # Check metadata for extraction methods
    assert "extraction_methods" in result.metrics.metadata
    methods = result.metrics.metadata["extraction_methods"]

    # Should have pattern-based methods
    pattern_methods = [k for k in methods.keys() if "pattern" in k]
    assert len(pattern_methods) > 0, "No pattern methods tracked"


def test_empty_text_handling(hybrid_pipeline):
    """Test handling of empty text"""
    with pytest.raises(ValueError, match="Paper text cannot be empty"):
        hybrid_pipeline.extract(
            paper_text="",
            paper_id="test_008"
        )


def test_short_text_handling(hybrid_pipeline):
    """Test handling of very short text"""
    with pytest.raises(ValueError, match="Paper text too short"):
        hybrid_pipeline.extract(
            paper_text="Short text.",
            paper_id="test_009"
        )


def test_imrad_section_detection(hybrid_pipeline):
    """Test IMRAD section detection"""
    result = hybrid_pipeline.extract(
        paper_text=SAMPLE_PAPER,
        paper_id="test_010"
    )

    # Check that sections were detected
    sections = result.metrics.metadata.get("sections_processed", [])
    assert len(sections) > 0, "No sections detected"

    # Should detect Introduction, Methods, Results
    expected_sections = ["introduction", "methods", "results"]
    found_sections = [s for s in expected_sections if s in sections]
    assert len(found_sections) >= 2, f"Expected {expected_sections}, found {sections}"


def test_pipeline_description(hybrid_pipeline):
    """Test pipeline description"""
    desc = hybrid_pipeline.get_description()
    assert "Hybrid" in desc
    assert "Cost" in desc or "cost" in desc


def test_estimated_cost(hybrid_pipeline):
    """Test estimated cost calculation"""
    cost = hybrid_pipeline.get_estimated_cost()
    assert cost >= 0.0
    assert cost <= 0.05  # Should be under 5 cents


@pytest.mark.integration
@pytest.mark.skipif(
    not pytest.config.getoption("--run-integration"),
    reason="Skipping integration test (requires API keys)"
)
def test_llm_fallback(hybrid_pipeline_with_llm):
    """Integration test with LLM fallback (requires API key)"""
    result = hybrid_pipeline_with_llm.extract(
        paper_text=SAMPLE_PAPER,
        paper_id="test_011"
    )

    # With LLM, should have conclusions
    conclusions = result.get_entities_by_type("conclusion")
    assert len(conclusions) > 0, "No conclusions extracted with LLM"

    # Cost should be > 0 if LLM was used
    assert result.metrics.cost_usd > 0


@pytest.mark.integration
@pytest.mark.skipif(
    not pytest.config.getoption("--run-integration"),
    reason="Skipping integration test (requires API keys)"
)
def test_cost_under_target(hybrid_pipeline_with_llm):
    """Integration test: verify cost is under target"""
    result = hybrid_pipeline_with_llm.extract(
        paper_text=SAMPLE_PAPER,
        paper_id="test_012"
    )

    # Cost should be under $0.05 target
    assert result.metrics.cost_usd < 0.05, \
        f"Cost ${result.metrics.cost_usd:.4f} exceeds target $0.05"

    # Ideally under $0.02
    if result.metrics.cost_usd < 0.02:
        print(f"Excellent! Cost ${result.metrics.cost_usd:.4f} is under $0.02 target")


# Pytest configuration for integration tests
def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run integration tests that require API keys"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: mark test as integration test")
