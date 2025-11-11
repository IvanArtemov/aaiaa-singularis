# Architecture Quick Reference

## File Structure & Key Components

```
src/
├── pipelines/
│   ├── scibert_nebius_pipeline.py    ← MAIN PIPELINE (6 phases)
│   └── base_pipeline.py
│
├── extractors/
│   ├── sentence_embedder.py           ← PHASE 0.5: Embedding generation
│   ├── keyword_generator.py           ← PHASE 1: Keyword generation
│   └── base_extractor.py
│
├── components/
│   ├── semantic_retriever.py          ← PHASE 4: Candidate retrieval
│   ├── entity_validator.py            ← PHASE 5: Entity validation
│   ├── graph_assembler.py             ← PHASE 6: Graph assembly
│   └── base_component.py
│
├── embedding_adapters/
│   ├── scibert_adapter.py             ← FREE 768-dim embeddings
│   ├── base_embedding_adapter.py
│   └── factory.py
│
├── llm_adapters/
│   ├── nebius_adapter.py              ← Nebius gpt-oss-120b
│   ├── base_adapter.py
│   └── factory.py
│
├── parsers/
│   ├── grobid_parser.py               ← PDF → IMRAD extraction
│   └── base_parser.py
│
├── models/
│   ├── entities.py                    ← Entity types, schemas, relationships
│   ├── graph.py                       ← KnowledgeGraph structure
│   ├── sentence.py                    ← Sentence with embeddings
│   ├── results.py                     ← ExtractionResult, metrics
│   └── base_model.py
│
├── visualization/
│   └── generate_svg.py                ← 8-column SVG layout
│
└── config/
    ├── scibert_nebius_config.yaml     ← Pipeline configuration
    ├── grobid_config.yaml             ← GROBID settings
    ├── llm_config.yaml                ← LLM costs
    └── settings.py
```

## Key Data Structures

```python
# INPUT
ParsedDocument:
  - paper_id: str
  - title: str
  - abstract: str
  - imrad_sections: Dict[str, str]  # "introduction", "methods", "results", "discussion"
  - sentences: List[Sentence]
  - metadata: Dict

# INTERMEDIATE
Sentence:
  - text: str
  - embedding: List[float]  (768 dims from SciBERT)
  - section: str
  - position: int
  - char_start, char_end: int

# OUTPUT
ExtractionResult:
  - paper_id: str
  - entities: Dict[str, List[Entity]]
    {
      "fact": [...],
      "hypothesis": [...],
      "experiment": [...],
      "method": [...],
      "result": [...],
      "dataset": [...],
      "analysis": [...],
      "conclusion": [...]
    }
  - relationships: List[Relationship]
  - metrics: PipelineMetrics
  - metadata: Dict

Entity:
  - id: str (UUID)
  - type: EntityType
  - text: str
  - confidence: float (0.65-1.0)
  - source_section: str
  - metadata: Dict

Relationship:
  - source_id: str
  - target_id: str
  - relationship_type: RelationshipType
  - confidence: float (0.6-1.0)
  - metadata: Dict (rule, overlap, section)
```

## Phase Execution Flow

```
PDF file
  ↓
GrobidParser.parse(file_path)
  ↓
ParsedDocument (IMRAD sections + metadata)
  ↓
SciBertNebiusPipeline.extract(parsed_doc, paper_id)
  │
  ├─ PHASE 0.5: SentenceEmbedder.process_document()
  │  └─ Output: sentences with SciBERT embeddings (768-dim)
  │
  ├─ PHASE 1: EntityKeywordGenerator.generate_all_keywords()
  │  └─ Output: Dict[EntityType → List[keywords]]
  │
  ├─ PHASE 4: SemanticRetriever.retrieve_candidates()
  │  ├─ Embed keywords with SciBERT
  │  ├─ Query ChromaDB with cosine similarity
  │  └─ Output: Dict[EntityType → List[candidates]]
  │
  ├─ PHASE 5: EntityValidator.validate_parallel()
  │  ├─ Batch validation with Nebius
  │  ├─ Apply type-specific thresholds
  │  └─ Output: Dict[EntityType → List[Entity]]
  │
  └─ PHASE 6: GraphAssembler.assemble_graph()
     ├─ Apply 8 linking rules
     ├─ Calculate relationship confidences
     └─ Output: List[Relationship]
  ↓
ExtractionResult
  ↓
Output generation:
  ├─ {paper_id}_entities.json
  ├─ {paper_id}_metrics.json
  └─ {paper_id}_graph.svg
```

## Configuration Parameters

### semantic_retrieval (Phase 4)
- top_k_per_type: Adaptive retrieval count
  - HYPOTHESIS: 35 (rare)
  - EXPERIMENT: 25
  - METHOD: 12 (common)
  - etc.
- section_filtering: Filter by IMRAD section
  - HYPOTHESIS: [intro, abstract]
  - METHOD: [methods, materials]
  - etc.

### validation (Phase 5)
- confidence_threshold: Per-type thresholds
  - METHOD: 0.72 (strict)
  - HYPOTHESIS: 0.68
  - DATASET: 0.65 (loose)
- batch_size: 5-10 candidates per LLM call
- parallel_types: True (4 threads)

### graph_assembly (Phase 6)
- min_relationship_confidence: 0.6
- proximity_window: 3 (sentences)
- relationship_modifiers:
  - same_section: 1.0
  - text_overlap_high (>20%): 0.9
  - text_overlap_medium (10-20%): 0.75
  - text_overlap_low (<10%): 0.6

## Cost Breakdown

| Phase | Cost | Tokens | Notes |
|-------|------|--------|-------|
| 0.5: SciBERT Embeddings | $0.000 | 0 | Local, FREE |
| 1: Keyword Generation | $0.003 | 2,000 | Minimal context |
| 4: Semantic Retrieval | $0.000 | 0 | Local ChromaDB |
| 5: Entity Validation | $0.015 | 10,000-15,000 | Batch processing |
| 6: Graph Assembly | $0.000 | 0 | Heuristics |
| **Total** | **$0.018** | **~12,000-17,000** | **Per paper** |

## Performance Targets

| Metric | Target |
|--------|--------|
| Precision | ≥88% |
| Recall | ≥82% |
| F1-Score | ≥85% |
| Speed | 60-90 seconds |
| Cost | $0.018 per paper |
| Entities | 50-200 per paper |
| Relationships | 100-400 per paper |

## Critical Decision Points

1. **Entity Extraction**
   - Two-stage approach: Broad retrieval (95% recall) → Precise validation (88% precision)
   - SciBERT for free, domain-optimized embeddings
   - Nebius LLM only for validation (expensive), not retrieval

2. **Relationship Assembly**
   - Heuristic-based (no LLM) to keep cost low
   - Section + keyword overlap rules
   - Confidence multiplication instead of LLM scoring

3. **Entity Schemas**
   - Fixed, predefined schemas per type
   - Type-specific validation criteria
   - No learning/adaptation (deterministic)

4. **Visualization**
   - Fixed 8-column layout (hierarchical)
   - Assumes linear flow (facts → conclusions)
   - SVG format (no interactivity)

## Extension Points

### To Improve Recall
1. Increase top-k in Phase 4 (costs nothing)
2. Lower confidence thresholds in Phase 5 (accepts more false positives)
3. Add iterative refinement loop (Phase 1 → Phase 4 feedback)
4. Implement coreference resolution (post-Phase 5)

### To Improve Precision
1. Add context window around candidates (±2 sentences)
2. Implement cross-reference checking (vs paper's abstract)
3. Use embedding distance for semantic validation (Phase 5)
4. Add domain-specific entity linkers (e.g., biomedical)

### To Reduce Cost
1. Skip keyword generation (saves $0.003)
2. Reduce batch size → fewer LLM calls (less accurate)
3. Use smaller LLM model (faster, cheaper)
4. Cache results across papers (same corpus)

### To Improve Speed
1. Parallelize Phase 0.5 (batch embedding)
2. Parallelize Phase 5 (already does 4 threads)
3. Skip SVG generation (saves 2-3 seconds)
4. Use GPU for SciBERT (3-5x faster)

## Testing & Validation

```bash
# Test pipeline with example
python scripts/example_scibert_nebius_pipeline.py

# Process single PDF
python scripts/process_pdf.py --pdf paper.pdf -o results -v

# Check outputs
cat results/paper_id_metrics.json
cat results/paper_id_entities.json
open results/paper_id_graph.svg

# Run Telegram bot
python scripts/run_scibert_telegram_bot.py
```

## Common Issues & Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| "No NEBIUS_API_KEY" | Missing .env | Add NEBIUS_API_KEY to .env |
| "spaCy model not found" | Dependency missing | `python -m spacy download en_core_web_sm` |
| "GROBID service unavailable" | Network issue | Check GROBID_URL in config |
| "ChromaDB not installed" | Missing dependency | `pip install chromadb` |
| "Low entity count" | Thresholds too high | Lower confidence_threshold in config |
| "Many false positives" | Thresholds too low | Raise confidence_threshold |
| "Slow processing" | No GPU | Use GPU or reduce batch_size |
| "Memory error" | Large paper | Reduce batch_size or use smaller model |

