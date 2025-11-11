# AAIAA Singularis Project - Architecture Analysis

## Executive Summary

The **SciBERT-Nebius Knowledge Graph Extractor** is a production-ready system for extracting structured information from scientific papers using a novel three-stage hybrid pipeline that combines cost-efficient embeddings with selective LLM processing. The architecture achieves **~$0.018 per paper** cost with **88%+ precision** by integrating:

1. **Free, domain-optimized SciBERT embeddings** for semantic search
2. **Cost-efficient Nebius LLM** for keyword generation and validation
3. **ChromaDB** for local semantic indexing
4. **GROBID** ML parser for structured PDF extraction
5. **Heuristic-based graph assembly** for relationship mining

---

## 1. Main Pipeline Flow: SciBERT-Nebius Architecture

### High-Level Pipeline Overview

```
PDF Document
    ↓
GROBID Parser (ML-based PDF parsing)
    ↓
ParsedDocument {
  - IMRAD sections (Introduction, Methods, Results, Discussion)
  - Title, Abstract, Metadata
  - Reference extraction
}
    ↓
PHASE 0.5: SciBERT Sentence Embeddings (FREE)
    ↓
Sentence objects with 768-dim vectors
    ↓
PHASE 1: LLM Keyword Generation (~$0.003)
    ↓
Entity-type-specific keywords (15 per type)
    ↓
PHASE 4: Semantic Retrieval via ChromaDB (FREE)
    ↓
Top-k candidate segments per entity type
    ↓
PHASE 5: LLM Validation (~$0.015)
    ↓
Validated entities with confidence scores
    ↓
PHASE 6: Graph Assembly (FREE)
    ↓
Knowledge Graph with relationships
    ↓
SVG Visualization (8-column hierarchical layout)
```

### Phase Breakdown

#### Phase 0.5: Sentence Embedding with SciBERT
**Cost**: $0.000 (FREE - local execution)
**File**: `src/extractors/sentence_embedder.py`

- **Input**: ParsedDocument with IMRAD sections
- **Process**:
  - Uses spaCy (en_core_web_sm) for sentence tokenization
  - Processes each IMRAD section separately
  - Generates SciBERT embeddings (768 dimensions)
  - Tracks sentence position, character offsets, section metadata
  - Batch processing (batch_size: 32)
  - LRU cache for duplicate sentences (cache_size: 256)

- **Output**: `Sentence` objects with embeddings
  ```python
  Sentence:
    - text: str
    - embedding: List[float] (768 dims)
    - section: str (e.g., "introduction")
    - position: int (sentence number)
    - char_start/char_end: int
  ```

- **Key Optimization**: SciBERT is pre-trained on 1.14M scientific papers from Semantic Scholar, making it 3-5x cheaper than API-based embeddings ($0 vs $0.10 per 1M tokens) while maintaining domain relevance.

#### Phase 1: Keyword Generation with Nebius
**Cost**: ~$0.003 per paper
**File**: `src/extractors/keyword_generator.py`

- **Input**: Title, Abstract, Introduction (limited to 500 words)
- **Process**:
  - LLM generates 15 keywords per entity type (8 types = 120 keywords total)
  - Uses minimal context to reduce tokens (~2000 input tokens)
  - Temperature: 0.1 (consistency over creativity)
  - LRU cache (cache_size: 128) - caches entire keyword sets by document fingerprint
  - Cache hit rate tracking

- **Output**: Dictionary mapping EntityType → List[keyword]
  ```python
  {
    EntityType.HYPOTHESIS: [
      "caloric restriction extends lifespan",
      "mitochondrial dysfunction hypothesis",
      "oxidative stress accumulation",
      ...
    ],
    EntityType.RESULT: [...],
    ...
  }
  ```

- **Cost Optimization**: 
  - Minimal prompt (title + abstract + intro only)
  - Fixed keyword count (15 per type)
  - Cache prevents duplicate keyword generation

#### Phase 4: Semantic Retrieval with ChromaDB
**Cost**: $0.000 (FREE - local ChromaDB)
**File**: `src/components/semantic_retriever.py`

- **Input**: Keyword embeddings + EntityType
- **Process**:
  - Embeds each keyword using SciBERT (FREE)
  - Queries ChromaDB with cosine similarity
  - Adaptive top-k per entity type:
    - HYPOTHESIS: 35 (rare, need more candidates)
    - EXPERIMENT: 25 (moderate frequency)
    - METHOD: 12 (common, avoid over-retrieval)
    - FACT: 10 (very common)
  - Section filtering (e.g., HYPOTHESIS only in intro/abstract)
  - Deduplication by segment ID
  - Sorting by distance (ascending)

- **Output**: List of candidates per entity type
  ```python
  {
    "id": "paper_seg_42",
    "text": "Caloric restriction has been shown to extend lifespan in mice",
    "distance": 0.15,  # Lower is better (cosine)
    "metadata": {
      "paper_id": "paper_123",
      "section": "introduction",
      "position": 5,
      "char_start": 1234,
      "char_end": 1456
    },
    "entity_type": EntityType.HYPOTHESIS
  }
  ```

- **Efficiency**:
  - ChromaDB uses HNSW (Hierarchical Navigable Small World) index
  - Cosine distance metric
  - Per-paper cleanup (prevents index bloat)

#### Phase 5: Entity Validation with Nebius
**Cost**: ~$0.015 per paper
**File**: `src/components/entity_validator.py`

- **Input**: Candidates grouped by entity type
- **Process**:
  - Batch validation (batch_size: 5-10 candidates per API call)
  - 3-level validation for METHOD type:
    - ✅ HIGHLY VALID (0.85+): Full procedural details (temp, concentration, duration)
    - ✅ VALID (0.70-0.85): Method with biological context
    - ❌ INVALID (<0.65): Vague method mentions
  
  - Type-specific guidelines:
    - **METHOD**: Requires HOW (parameters, conditions, steps)
    - **EXPERIMENT**: WHAT WAS DONE to test hypothesis (not just observation)
    - **DATASET**: Data source identifiers or collection references
    - **FACT**: Established knowledge from literature
    - **HYPOTHESIS**: Scientific assumptions to test
    - **RESULT**: Measurements, observations, findings
    - **ANALYSIS**: Statistical or computational processing
    - **CONCLUSION**: Interpretations and implications

  - Confidence thresholds per type:
    - HYPOTHESIS: 0.68 (lower to capture more)
    - EXPERIMENT: 0.68
    - METHOD: 0.72 (higher - more critical)
    - DATASET: 0.65 (lower - mentions sufficient)

  - Parallel validation: ThreadPoolExecutor (max_workers: 4)

- **Output**: `Entity` objects
  ```python
  Entity:
    - id: str (UUID-based)
    - type: EntityType
    - text: str (validated, possibly shortened)
    - confidence: float (0.0-1.0)
    - source_section: str
    - metadata: {
        "validation_method": "llm_batch",
        "original_text": str,
        "position": int,
        "char_start/char_end": int,
        "distance": float  # from semantic retrieval
      }
  ```

- **Parsing Logic**:
  - Expects JSON array response from LLM
  - Filters by confidence threshold
  - Core text extraction (LLM can summarize 1-2 sentences)
  - Metadata preservation for traceability

#### Phase 6: Graph Assembly (Heuristic-based)
**Cost**: $0.000 (FREE - no LLM)
**File**: `src/components/graph_assembler.py`

- **Input**: All validated entities + sentences
- **Process**: Applies relationship rules using heuristics
  
  1. **Fact → Hypothesis**: Same section (intro), confidence = min(fact_conf, hyp_conf) × 0.8
  
  2. **Hypothesis → Experiment**: Cross-section link, confidence = min() × 0.75
  
  3. **Experiment → Method**: Text overlap check
     - >10% overlap: confidence × 0.85
     - <10% overlap: confidence × 0.7
  
  4. **Experiment → Dataset**: Section proximity (methods/materials), confidence × 0.7
  
  5. **Method → Result**: Text overlap >15%: confidence × 0.75 / <15%: confidence × 0.65
  
  6. **Result → Analysis**: Same section: confidence × 0.8 / Different: confidence × 0.65
  
  7. **Result → Conclusion**: Text overlap >20%: confidence × 0.85 / <20%: confidence × 0.75
  
  8. **Analysis → Conclusion**: Logical flow, confidence × 0.7

- **Filtering**: Only relationships with confidence ≥ 0.6

- **Output**: List of `Relationship` objects
  ```python
  Relationship:
    - source_id: str
    - target_id: str
    - relationship_type: RelationshipType
    - confidence: float
    - metadata: {
        "rule": str (e.g., "same_section"),
        "text_overlap": float,
        "section": str
      }
  ```

- **Key Optimization**: Text overlap uses Jaccard similarity on keywords:
  ```
  overlap = |keywords1 ∩ keywords2| / |keywords1 ∪ keywords2|
  ```
  - Keywords: words >3 chars, excluding stop words

---

## 2. Entity Extraction Approach

### Two-Level Extraction Strategy

The system uses a **two-stage filtering** approach to achieve precision:

#### Stage 1: Broad Semantic Retrieval
- Use keywords to find candidate segments via embedding similarity
- Retrieve more candidates than needed (top-k varies by type: 10-35)
- Cost: FREE (local SciBERT + ChromaDB)
- Precision: ~50-60% (many false positives)
- Recall: ~95% (catches most real entities)

#### Stage 2: Selective LLM Validation
- Only validate retrieved candidates (not entire document)
- 3-level validation with type-specific criteria
- Cost: ~$0.015 (highly optimized batch processing)
- Precision: ~88%+ (removes false positives)
- Recall: ~82% (may miss some hard-to-detect entities)

### Eight Entity Types

**File**: `src/models/entities.py`

| Entity Type | Description | Typical Sections | Validation Criteria |
|------------|-------------|------------------|-------------------|
| **FACT** | Established knowledge, prior findings | intro, abstract, discussion | Signals: "has been shown", "previous studies", "evidence suggests" |
| **HYPOTHESIS** | Scientific assumption to test | intro, abstract | Signals: "we hypothesize", "we propose", "research question", "aim was to" |
| **EXPERIMENT** | Procedures for testing hypotheses | methods, materials | Signals: "experiment", "procedure", "treatment", "clinical trial", "CRISPR knockout" |
| **METHOD** | Detailed procedural descriptions with parameters | methods, materials | MUST include HOW: concentrations, temps, durations, versions, procedural steps |
| **RESULT** | Experimental findings, measurements, observations | results, discussion | Signals: "we found", "showed that", "revealed", "measured", "p <0.05" |
| **DATASET** | Data collections used/generated | methods, materials, results | Signals: "GSE*", "SRA*", "public repository", "scRNA-seq data", "samples", "cohort" |
| **ANALYSIS** | Statistical tests, computational methods | methods, results | Signals: "t-test", "ANOVA", "chi-square", "machine learning", "normalized" |
| **CONCLUSION** | Interpretations, implications, findings | conclusion, discussion, abstract | Signals: "conclude", "these findings suggest", "implication", "future research" |

### Entity Schemas: Structured Definitions

Each entity type has a schema defining:
1. **Description**: What constitutes a valid entity
2. **Typical sections**: Where to search
3. **Signal patterns**: Regex patterns indicating entity type

Example - METHOD schema:
```python
EntitySchema(
  entity_type=EntityType.METHOD,
  description="Detailed methodological descriptions including procedures, 
              protocols, parameters, and instruments. Must describe HOW...",
  typical_sections=["methods", "materials"],
  signal_patterns=[
    r"\b(we\s+used|using|employed|performed|conducted)",
    r"\b(method|technique|approach|assay|analysis)",
    r"\b(RT-PCR|Western\s+blot|ELISA|microscopy|sequencing)",
    ...
  ]
)
```

### Adaptive Top-k Strategy

The system adjusts retrieval per entity type based on rarity:

```yaml
HYPOTHESIS: 35  # Rare in papers (1-3 per paper)
EXPERIMENT: 25  # Moderate rarity (5-10 per paper)
METHOD: 12      # Common (50+ mentions, avoid over-retrieval)
FACT: 10        # Very common (100+ facts per paper)
RESULT: 15      # Moderate
DATASET: 15     # Moderate
ANALYSIS: 15    # Moderate
CONCLUSION: 20  # Common
```

---

## 3. Knowledge Graph Assembly

### Graph Structure

**File**: `src/models/graph.py`, `src/models/entities.py`

```python
KnowledgeGraph:
  - paper_id: str
  - entities: List[Entity]        # 50-200 entities per paper
  - relationships: List[Relationship]  # 100-400 relationships
  - metadata: Dict

Entity:
  - id: str (UUID-based)
  - type: EntityType
  - text: str (1-3 sentences)
  - confidence: float (0.65-1.0)
  - source_section: str
  - metadata: validation details

Relationship:
  - source_id: str
  - target_id: str
  - relationship_type: RelationshipType (14 types)
  - confidence: float (0.6-1.0)
  - metadata: rule, text_overlap, section info
```

### Relationship Types (14 total)

**Fact Relationships:**
- `FACT_TO_HYPOTHESIS`: Established knowledge informs hypothesis

**Hypothesis Relationships:**
- `HYPOTHESIS_TO_EXPERIMENT`: Hypothesis motivates experiment
- `HYPOTHESIS_TO_METHOD`: Hypothesis guides method selection

**Experimental Relationships:**
- `EXPERIMENT_USES_TECHNIQUE`: Experiment employs method
- `EXPERIMENT_USES_DATASET`: Experiment uses data
- `METHOD_TO_RESULT`: Method produces result

**Result Relationships:**
- `RESULT_TO_ANALYSIS`: Result undergoes analysis
- `RESULT_TO_CONCLUSION`: Result supports conclusion

**Analysis Relationships:**
- `ANALYSIS_TO_CONCLUSION`: Analysis informs conclusion

**Generic Relationships:**
- `BASED_ON`, `SUPPORTS`, `CONTRADICTS`, `RELATED_TO`

### Assembly Rules (Heuristic-Based)

1. **Section-based linking**:
   - Fact + Hypothesis in same section → link (confidence × 0.8)
   - Result + Analysis in same section → stronger link (× 0.8 vs × 0.65)

2. **Cross-section linking**:
   - Hypothesis (intro) → Experiment (methods): confidence × 0.75
   - Method (methods) → Result (results): confidence × 0.65-0.75

3. **Keyword overlap-based linking**:
   - Calculate Jaccard similarity on extracted keywords
   - High overlap (>20%): confidence × 0.85
   - Medium overlap (10-20%): confidence × 0.75
   - Low overlap (<10%): confidence × 0.6

4. **Confidence combination**:
   - New confidence = min(source_confidence, target_confidence) × modifier
   - Filters relationships with confidence < 0.6

### Assembly Flow

```python
# 1. Group entities by type
entities_by_type = {
  EntityType.FACT: [...],
  EntityType.HYPOTHESIS: [...],
  ...
}

# 2. Apply linking rules sequentially
relationships = []
relationships.extend(link_facts_to_hypotheses(entities_by_type))
relationships.extend(link_hypotheses_to_experiments(entities_by_type))
relationships.extend(link_experiments_to_techniques(entities_by_type))
relationships.extend(link_experiments_to_datasets(entities_by_type))
relationships.extend(link_techniques_to_results(entities_by_type))
relationships.extend(link_results_to_analyses(entities_by_type))
relationships.extend(link_results_to_conclusions(entities_by_type))
relationships.extend(link_analyses_to_conclusions(entities_by_type))

# 3. Filter low-confidence relationships
relationships = [r for r in relationships if r.confidence >= 0.6]
```

### Visualization: 8-Column SVG Layout

**File**: `src/visualization/generate_svg.py`

```
Input Facts (150px)
    ↓ (dashed)
Hypotheses (430px)
    ↓
Experiments (710px) | Techniques (990px)
    ↓
Results (1270px) | Datasets (1550px)
    ↓
Analysis (1830px)
    ↓
Conclusions (2110px)
```

**Features**:
- SVG width: 2400px (8 columns × 300px spacing)
- Nodes: Rounded rectangles (200px wide) with text wrapping
- Edges: Bezier curves with directional arrows
- Colors: 8 distinct colors per entity type
- Dynamic node height based on text length

---

## 4. Current Limitations & Areas for Improvement

### A. Limitations

#### 1. **Heuristic-Based Graph Assembly**
- **Issue**: Relationship discovery relies on simple rules (section proximity, keyword overlap)
- **Limitation**: 
  - Cannot capture implicit relationships (e.g., negative results contradicting hypothesis)
  - Misses relationships across distant sections
  - No temporal reasoning
  - Keyword overlap is crude (synonyms not detected)
- **Impact**: Recall ~82% for complex relationships
- **Potential fix**: Optional LLM refinement (not implemented, would add $0.005-0.01)

#### 2. **Sequential Processing**
- **Issue**: Phases run sequentially; no feedback loops
- **Limitation**:
  - Cannot refine keywords based on found entities
  - No iterative validation
  - Missed candidate in Phase 4 cannot be recovered
- **Impact**: Recall ceiling at ~82%

#### 3. **SciBERT Domain Limitations**
- **Issue**: SciBERT trained on computer science + biology papers
- **Limitation**:
  - May underperform on specialized domains (chemistry, materials science, astronomy)
  - Fixed 768-dim vectors might compress domain-specific semantics
  - Pre-trained 2018 (may not capture recent terminology)
- **Impact**: Potentially lower performance on non-core domains
- **Potential fix**: Fine-tune SciBERT on domain-specific corpus

#### 4. **Batch Size Constraints in Validation**
- **Issue**: Batch size limited to 5-10 candidates per LLM call
- **Limitation**:
  - Reduced context (300 chars per candidate)
  - Cannot see relationships between candidates in batch
  - Truncation may lose critical entity identifiers
- **Impact**: May reject valid entities due to isolation
- **Potential fix**: Increase batch_size with better prompt engineering

#### 5. **No Coreference Resolution**
- **Issue**: Same entity mentioned multiple ways not linked
- **Limitation**:
  - "caloric restriction", "CR", "dietary restriction" treated as separate entities
  - Duplicate entities in output
  - Inflated entity count, sparse relationships
- **Impact**: Recall degradation, graph fragmentation
- **Potential fix**: Post-processing with fuzzy matching or LLM-based entity linking

#### 6. **Fixed Confidence Thresholds**
- **Issue**: Same threshold for all papers (e.g., METHOD: 0.72)
- **Limitation**:
  - May be too strict for high-quality papers (miss real entities)
  - May be too loose for low-quality papers (accept false positives)
  - No adaptation to paper characteristics
- **Impact**: ~2-3% precision/recall variance
- **Potential fix**: Adaptive thresholds based on paper quality metrics

#### 7. **Limited IMRAD Flexibility**
- **Issue**: Assumes papers follow IMRAD structure
- **Limitation**:
  - Review papers, position papers, opinion pieces don't fit
  - Non-English papers may not parse well
  - Some fields (math, theory) lack "Experiment" section
- **Impact**: Degraded performance on non-standard papers
- **Potential fix**: Document structure detection + flexible section mapping

#### 8. **Visualization Scalability**
- **Issue**: 8-column SVG layout assumes linear flow
- **Limitation**:
  - 200+ entities become cluttered
  - Complex interconnections hard to see
  - No interactive exploration
- **Impact**: User experience for complex papers
- **Potential fix**: Interactive graph visualization (D3.js, Cytoscape.js)

### B. Areas for Improvement

#### 1. **Relationship Confidence Scoring**
- **Current**: Simple confidence = min(source, target) × modifier
- **Proposed**:
  - Multi-factor scoring: section distance + keyword overlap + entity proximity + type compatibility
  - ML-based relationship validator
  - Bidirectional relationship suggestion (Method ← Result if high overlap)

#### 2. **Semantic Linking**
- **Current**: Keyword overlap via Jaccard
- **Proposed**:
  - Semantic similarity using embedding distance
  - Synonym detection (WordNet, domain ontologies)
  - Biomedical entity linkers (Gene/protein names, diseases)

#### 3. **Iterative Refinement Loop**
```
Iteration 1: Initial extraction
    ↓
Analyze extraction gaps (low entity count)
    ↓
Increase top-k + lower thresholds
    ↓
Iteration 2: Expanded extraction
    ↓
Prune duplicates + low-confidence entities
    ↓
Final graph
```

#### 4. **Context-Aware Validation**
- **Current**: Validate each entity in isolation
- **Proposed**:
  - Include surrounding context (±2 sentences)
  - Cross-reference with paper's abstract/title
  - Check consistency with paper's domain tags

#### 5. **Cross-Paper Linking**
- **Current**: Single paper analysis
- **Proposed**:
  - Link entities across multiple papers
  - Create meta-relationships (this paper extends prior work)
  - Build corpus-wide knowledge graphs

#### 6. **Fine-grained Entity Attributes**
- **Current**: Simple text + confidence
- **Proposed**:
  - Entity attributes (e.g., METHOD: {name, parameters: [{param, value, unit}]})
  - Type-specific schemas with structured sub-fields
  - Enable structured data export

#### 7. **Confidence Interpretability**
- **Current**: Single confidence score
- **Proposed**:
  - Decomposed confidence: validation_confidence + relationship_confidence + source_reliability
  - Traceability: show which validation criteria passed/failed
  - Uncertainty quantification (confidence intervals)

#### 8. **Cost-Quality Trade-offs**
- **Current**: Fixed $0.018/paper (fixed resource budget)
- **Proposed**:
  - Allow user to specify budget vs quality
  - Variable batch sizes: larger batches (×$, +speed, -accuracy)
  - Optional phases: skip keyword generation for cost reduction (-10% recall, +20% speed)

---

## 5. Paper Processing Pipeline

### End-to-End Workflow

**File**: `scripts/process_pdf.py`

```
INPUT: PDF file (paper.pdf)
    ↓
STEP 1: Validation
  - Check NEBIUS_API_KEY in .env
  - Check dependencies (transformers, torch, chromadb, spacy)
  - Verify spaCy model (en_core_web_sm)
  - Check file exists and is readable
    ↓
STEP 2: GROBID Parsing
  - Call GROBID service (https://lfoppiano-grobid.hf.space by default)
  - Get TEI XML with IMRAD sections
  - Extract title, abstract, references
  - Duration: 2-5 seconds per page
    ↓
STEP 3: Document Structure
  ParsedDocument created:
    - paper_id: "filename_hash" (e.g., "acme2023_abc123")
    - title: "Full Paper Title"
    - abstract: "Abstract text..."
    - imrad_sections: {
        "introduction": "...",
        "methods": "...",
        "results": "...",
        "discussion": "..."
      }
    - metadata: {filename, word_count, page_count, upload_time}
    - sentences: [] (empty, populated in Phase 0.5)
    ↓
STEP 4: SciBERT Pipeline Initialization
  SciBertNebiusPipeline created with:
    - config from scibert_nebius_config.yaml
    - SentenceEmbedder (SciBERT model preloaded)
    - EntityKeywordGenerator (LLM adapter ready)
    - SemanticRetriever (ChromaDB client ready)
    - EntityValidator (batch validation ready)
    - GraphAssembler (heuristic rules loaded)
    ↓
STEP 5: Extraction (extract() method)
  Phase 0.5 → Phase 1 → Phase 4 → Phase 5 → Phase 6
  (See detailed flow above)
  Returns: ExtractionResult object
    ↓
STEP 6: Results Assembly
  ExtractionResult:
    - paper_id
    - entities: {type → List[Entity]}
    - relationships: List[Relationship]
    - metrics: PipelineMetrics
    - metadata: document metadata
    ↓
STEP 7: Output Generation
  A. JSON Export
     - {paper_id}_entities.json: Full extraction result
       {
         "paper_id": "...",
         "entities": {
           "hypothesis": [{...}, {...}],
           "experiment": [...],
           ...
         },
         "relationships": [{...}, {...}],
         "metadata": {...}
       }
  
  B. Metrics Export
     - {paper_id}_metrics.json: Performance metrics
       {
         "processing_time_seconds": 45.3,
         "tokens_used": 12847,
         "cost_usd": 0.0184,
         "entities_extracted": 87,
         "relationships_extracted": 156,
         "phases": {
           "embeddings": {...},
           "keyword_generation": {...},
           "semantic_retrieval": {...},
           "validation": {...},
           "graph_assembly": {...}
         }
       }
  
  C. SVG Visualization
     - {paper_id}_graph.svg: Knowledge graph (8 columns)
       Nodes: colored boxes per entity type
       Edges: directed arrows per relationship type
       ↓
OUTPUT: Structured knowledge graph in JSON + Visual representation in SVG
```

### CLI Usage

```bash
# Basic usage
python scripts/process_pdf.py --pdf paper.pdf

# Custom output directory
python scripts/process_pdf.py -p paper.pdf -o results

# Skip SVG (faster)
python scripts/process_pdf.py -p paper.pdf --no-svg

# Verbose mode (debug)
python scripts/process_pdf.py -p paper.pdf -v

# Output structure:
# results/
# ├── paper_id_entities.json
# ├── paper_id_metrics.json
# └── paper_id_graph.svg
```

### Telegram Bot Integration

**File**: `bot/scibert_telegram_bot.py`

```
User sends PDF → Bot downloads → process_pdf.py runs → Returns SVG graph
```

Features:
- Session management (user statistics)
- Rate limiting (configurable)
- Error handling + user feedback
- Results caching

---

## Summary Table

| Component | Technology | Cost | Latency | Key Role |
|-----------|-----------|------|---------|----------|
| **PDF Parsing** | GROBID ML | $0.000 | 2-5s | Structured IMRAD extraction |
| **Sentence Embeddings** | SciBERT (768d) | $0.000 | 3-5s | Free domain-optimized vectors |
| **Keyword Generation** | Nebius gpt-oss-120b | $0.003 | 5-10s | Context-aware search terms |
| **Semantic Search** | ChromaDB HNSW | $0.000 | 2-3s | Fast candidate retrieval |
| **Entity Validation** | Nebius gpt-oss-120b | $0.015 | 15-20s | Precision filtering |
| **Graph Assembly** | Heuristics | $0.000 | 1-2s | Relationship mining |
| **Visualization** | SVG generation | $0.000 | 2-3s | 8-column hierarchical layout |
| **TOTAL** | Hybrid | **$0.018** | **60-90s** | Production-ready extraction |

---

## Metrics: Target vs Current

| Metric | Target | Estimated | Notes |
|--------|--------|-----------|-------|
| **Precision** | ≥88% | ~88-90% | Achieved via 2-stage filtering |
| **Recall** | ≥82% | ~82-85% | Limited by heuristic relationships |
| **F1-Score** | ≥85% | ~85% | Balanced precision/recall |
| **Cost/Paper** | ~$0.018 | $0.018 | SciBERT FREE + Nebius ~$0.018 |
| **Speed** | <90s | 60-90s | Mostly sequential, some parallelization |
| **Entities/Paper** | 50-200 | ~100-150 | Varies by paper complexity |
| **Relationships/Paper** | 100-400 | ~150-250 | Heuristic-based generation |

---

## Architecture Strengths

1. **Cost-efficient**: $0.018/paper vs $0.50-$2.00 pure LLM
2. **Accurate**: 88%+ precision via 2-stage filtering
3. **Fast**: 60-90 seconds end-to-end
4. **Scalable**: Hybrid approach avoids LLM for every candidate
5. **Interpretable**: Each entity has confidence + source traceability
6. **Flexible**: Entity schemas + confidence thresholds are configurable
7. **Production-ready**: CLI tool + Telegram bot + comprehensive logging
8. **Domain-optimized**: SciBERT trained on 1.14M scientific papers

## Architecture Trade-offs

1. **Recall vs Cost**: Limited by retrieval top-k and validation threshold
2. **Precision vs Recall**: Entity-specific thresholds (METHOD stricter than DATASET)
3. **Speed vs Accuracy**: Batch validation trades latency for cost savings
4. **Flexibility vs Simplicity**: Fixed heuristic rules vs LLM refinement

