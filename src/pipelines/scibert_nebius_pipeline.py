"""SciBERT-Nebius Hybrid Extraction Pipeline"""

import time
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict

from .base_pipeline import BasePipeline
from src.models import (
    Entity,
    Relationship,
    EntityType,
    RelationshipType,
    ExtractionResult,
    PipelineMetrics,
    ENTITY_SCHEMAS,
    get_entity_schema
)
from src.parsers import ParsedDocument
from src.extractors import EntityKeywordGenerator, SentenceEmbedder
from src.components import SemanticRetriever, EntityValidator, GraphAssembler
from src.llm_adapters import get_llm_adapter


class SciBertNebiusPipeline(BasePipeline):
    """
    SciBERT-Nebius Hybrid Extraction Pipeline

    Universal system for extracting all entity types using SciBERT embeddings
    and Nebius LLM validation. Combines domain-optimized embeddings with
    cost-efficient LLM processing.

    Architecture:
    0. Parse PDF with GROBID → IMRAD sections
    0.5. Sentence Embeddings via SciBERT (FREE - local, 768 dims) ✨
    1. LLM Keyword Generation via Nebius (~$0.003/paper)
    4. Semantic Retrieval via ChromaDB (FREE - local)
    5. LLM Validation via Nebius in batches (~$0.015/paper)
    6. Graph Assembly with heuristics (FREE)

    Cost: ~$0.018 per paper (embeddings FREE with SciBERT!)
    Embeddings: SciBERT (domain-optimized for scientific papers)
    LLM: Nebius gpt-oss-120b (cost-efficient)
    Target Precision: ≥ 88%
    Target Recall: ≥ 82%
    Target F1-Score: ≥ 85%
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize SciBERT-Nebius Pipeline

        Args:
            config: Configuration dictionary (optional if config_path provided)
            config_path: Path to scibert_nebius_config.yaml (optional)
        """
        # Load config from file if provided
        if config_path:
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                config = file_config.get("scibert_nebius_pipeline", {})
        elif config is None:
            # Load default config
            default_config_path = Path(__file__).parent.parent / "config" / "scibert_nebius_config.yaml"
            if default_config_path.exists():
                with open(default_config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                    config = file_config.get("scibert_nebius_pipeline", {})
            else:
                config = {}

        super().__init__(config)

        self.config = config

        # Initialize Nebius LLM adapter (for text generation: keywords, validation)
        self.llm_adapter = get_llm_adapter("nebius")

        # Phase 0.5: Sentence Embedder (uses SciBERT)
        embedding_config = config.get("embedding", {})
        self.sentence_embedder = SentenceEmbedder(
            embedding_provider="scibert",
            use_embedding_adapter=True,  # Use SciBERT adapter!
            batch_size=embedding_config.get("batch_size", 32),
            cache_size=embedding_config.get("cache_size", 256)
        )

        # Phase 1: Keyword Generator (uses Nebius)
        keyword_config = config.get("keyword_generation", {})
        self.keyword_generator = EntityKeywordGenerator(
            llm_provider="nebius",
            llm_model=keyword_config.get("model", "openai/gpt-oss-120b"),
            cache_size=keyword_config.get("cache_size", 128)
        )

        # Phase 4: Semantic Retriever (ChromaDB)
        retrieval_config = config.get("semantic_retrieval", {})
        self.retriever = SemanticRetriever(
            collection_name=retrieval_config.get("collection_name", "scibert_segments"),
            persist_directory=retrieval_config.get("persist_directory", "./chroma_db_scibert"),
            distance_metric=retrieval_config.get("distance_metric", "cosine")
        )

        # Phase 5: Entity Validator (uses Nebius)
        validation_config = config.get("validation", {})
        self.validator = EntityValidator(
            llm_adapter=self.llm_adapter,
            batch_size=validation_config.get("batch_size", 10),
            temperature=validation_config.get("temperature", 0.1),
            max_tokens=validation_config.get("max_tokens", 1000)
        )

        # Phase 6: Graph Assembler
        graph_config = config.get("graph_assembly", {})
        self.assembler = GraphAssembler(
            use_llm_refinement=graph_config.get("use_llm_refinement", False),
            proximity_window=graph_config.get("proximity_window", 3),
            min_relationship_confidence=graph_config.get("min_relationship_confidence", 0.6)
        )

        # Get embedding adapter for keyword embeddings
        from src.embedding_adapters import get_embedding_adapter
        self.embedding_adapter = get_embedding_adapter("scibert")

    def extract(
        self,
        parsed_doc: ParsedDocument,
        paper_id: str
    ) -> ExtractionResult:
        """
        Extract entities and relationships using SciBERT-Nebius approach

        Args:
            parsed_doc: ParsedDocument with IMRAD sections
            paper_id: Unique paper identifier

        Returns:
            ExtractionResult with extracted entities and relationships
        """
        start_time = time.time()

        # Validate input
        self._validate_parsed_doc(parsed_doc)

        # Track metrics
        total_tokens = 0
        total_cost = 0.0

        # ========== PHASE 0.5: Sentence Embedding (FREE with SciBERT) ==========

        if not parsed_doc.sentences:
            print("📝 Phase 0.5: Creating sentence embeddings with SciBERT...")
            parsed_doc = self.sentence_embedder.process_document(parsed_doc)

            emb_metrics = self.sentence_embedder.get_metrics()
            total_tokens += emb_metrics["total_tokens"]
            total_cost += emb_metrics["total_cost_usd"]

            print(f"   ✓ Embeddings: {emb_metrics['total_sentences']} sentences, "
                  f"${emb_metrics['total_cost_usd']:.6f} (FREE with SciBERT), "
                  f"{emb_metrics['embedding_time_seconds']:.2f}s")

        # ========== PHASE 1: Keyword Generation with Nebius (~$0.003) ==========

        print("🔑 Phase 1: Generating entity-specific keywords with Nebius...")
        keywords_by_type = self.keyword_generator.generate_all_keywords(parsed_doc)

        kg_metrics = self.keyword_generator.get_metrics()
        total_tokens += kg_metrics["total_tokens"]
        total_cost += kg_metrics["total_cost_usd"]

        print(f"   ✓ Keywords: {len(keywords_by_type)} types, "
              f"${kg_metrics['total_cost_usd']:.6f}")

        # ========== PHASE 4: Semantic Retrieval (FREE) ==========

        print("🔍 Phase 4: Semantic retrieval of candidates...")

        # Index document segments
        self.retriever.index_segments(parsed_doc.sentences, paper_id)

        # Get retrieval config
        retrieval_config = self.config.get("semantic_retrieval", {})
        top_k_per_type = retrieval_config.get("top_k_per_type", {})
        sections_by_type = retrieval_config.get("sections_by_type", {})

        # Retrieve candidates for each entity type
        candidates_by_type = {}

        for entity_type in EntityType:
            keywords = keywords_by_type.get(entity_type, [])
            if not keywords:
                continue

            # Create embeddings for keywords using SciBERT
            keyword_embeddings = self.embedding_adapter.embed(keywords)

            # Get top-k and section filter from config
            top_k = top_k_per_type.get(entity_type.value.upper(), 20)
            section_filter = sections_by_type.get(entity_type.value.upper())

            # Retrieve candidates
            candidates = self.retriever.retrieve_candidates(
                query_embeddings=keyword_embeddings,
                entity_type=entity_type,
                top_k=top_k,
                section_filter=section_filter,
                paper_id=paper_id
            )

            if candidates:
                candidates_by_type[entity_type] = candidates

        retrieval_metrics = self.retriever.get_metrics()
        print(f"   ✓ Retrieved: {retrieval_metrics['total_results']} candidates "
              f"from {retrieval_metrics['total_queries']} queries")

        # ========== PHASE 5: LLM Validation with Nebius (~$0.015) ==========

        print("✅ Phase 5: Validating candidates with Nebius LLM...")

        # Get validation config
        validation_config = self.config.get("validation", {})
        confidence_thresholds = validation_config.get("confidence_threshold", {})
        parallel_types = validation_config.get("parallel_types", True)
        max_workers = validation_config.get("max_workers", 4)

        # Validate in parallel
        if parallel_types:
            validated_by_type = self.validator.validate_parallel(
                candidates_by_type=candidates_by_type,
                entity_schemas=ENTITY_SCHEMAS,
                confidence_threshold=confidence_thresholds,
                max_workers=max_workers
            )
        else:
            # Sequential validation
            validated_by_type = {}
            for entity_type, candidates in candidates_by_type.items():
                schema = ENTITY_SCHEMAS.get(entity_type)
                threshold = confidence_thresholds.get(entity_type.value.upper(), 0.7)

                validated = self.validator.validate_batch(
                    candidates=candidates,
                    entity_type=entity_type,
                    entity_schema=schema,
                    confidence_threshold=threshold
                )
                validated_by_type[entity_type] = validated

        # Collect all entities
        all_entities = []
        for entities in validated_by_type.values():
            all_entities.extend(entities)

        validation_metrics = self.validator.get_metrics()
        total_tokens += validation_metrics["total_tokens"]
        total_cost += validation_metrics["total_cost_usd"]

        print(f"   ✓ Validated: {validation_metrics['total_validated']} entities "
              f"(rejected {validation_metrics['total_rejected']}), "
              f"${validation_metrics['total_cost_usd']:.6f}")

        # ========== PHASE 6: Graph Assembly (FREE) ==========

        print("🕸️  Phase 6: Assembling knowledge graph...")

        relationships = self.assembler.assemble_graph(
            entities=all_entities,
            sentences=parsed_doc.sentences
        )

        graph_metrics = self.assembler.get_metrics()
        print(f"   ✓ Graph: {graph_metrics['total_relationships']} relationships")

        # Clean up retriever
        self.retriever.clear_paper(paper_id)

        # ========== Group entities by type ==========

        entities_by_type = defaultdict(list)
        for entity in all_entities:
            entities_by_type[entity.type.value].append(entity)

        # Calculate metrics
        processing_time = time.time() - start_time

        # Prepare metadata
        metadata_dict = {
            "pipeline": "scibert_nebius",
            "embedding_provider": "scibert",
            "llm_provider": "nebius",
            "phases": {
                "embeddings": {
                    "provider": "scibert",
                    "sentences": emb_metrics["total_sentences"] if parsed_doc.sentences else 0,
                    "tokens": emb_metrics["total_tokens"] if parsed_doc.sentences else 0,
                    "cost_usd": emb_metrics["total_cost_usd"] if parsed_doc.sentences else 0
                },
                "keyword_generation": {
                    "entity_types": len(keywords_by_type),
                    "tokens": kg_metrics["total_tokens"],
                    "cost_usd": kg_metrics["total_cost_usd"],
                    "cache_hit_rate": kg_metrics["cache_hit_rate"]
                },
                "semantic_retrieval": {
                    "queries": retrieval_metrics["total_queries"],
                    "candidates": retrieval_metrics["total_results"],
                    "collection_size": retrieval_metrics["collection_size"]
                },
                "validation": {
                    "candidates": validation_metrics["total_candidates"],
                    "validated": validation_metrics["total_validated"],
                    "rejected": validation_metrics["total_rejected"],
                    "tokens": validation_metrics["total_tokens"],
                    "cost_usd": validation_metrics["total_cost_usd"]
                },
                "graph_assembly": {
                    "relationships": graph_metrics["total_relationships"],
                    "by_type": graph_metrics["relationships_by_type"]
                }
            },
            "sections_processed": list(parsed_doc.imrad_sections.keys()) if parsed_doc.imrad_sections else []
        }

        metrics = PipelineMetrics(
            processing_time=processing_time,
            tokens_used=total_tokens,
            cost_usd=total_cost,
            entities_extracted=len(all_entities),
            relationships_extracted=len(relationships),
            metadata=metadata_dict
        )

        self.last_metrics = metrics

        print(f"\n✨ Extraction complete: {len(all_entities)} entities, "
              f"{len(relationships)} relationships, "
              f"${total_cost:.6f}, {processing_time:.2f}s")

        # Build result
        result = ExtractionResult(
            paper_id=paper_id,
            entities=dict(entities_by_type),
            relationships=relationships,
            metrics=metrics,
            metadata=parsed_doc.metadata
        )

        return result

    def get_description(self) -> str:
        """Get pipeline description"""
        return (
            f"SciBERT-Nebius Pipeline (SciBERT Embeddings + Nebius LLM Validation). "
            f"Cost: ~$0.018/paper. Embeddings: FREE (SciBERT), LLM: Nebius gpt-oss-120b."
        )

    def get_estimated_cost(self) -> float:
        """Get estimated cost per paper"""
        return 0.018  # FREE embeddings (SciBERT) + Nebius LLM calls
