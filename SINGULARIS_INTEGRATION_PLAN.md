# Singularis Integration Plan
## Анализ интеграции проекта lo-dn/singularis в aaiaa-singularis

**Дата создания:** 2025-11-11
**Статус:** Proposal / Planning Phase

---

## 📊 Comparative Analysis

### Singularis (lo-dn/singularis)
**Approach:** Deterministic rules-based extraction
**Core Technology:** spaCy pattern matching + JSON rule definitions
**Architecture:** 3-stage pipeline (S0→S1→S2)
**Cost:** ~10-100× cheaper than LLM-only
**Quality:** 80-90% of LLM quality
**Strengths:**
- Ultra-low cost and latency
- Deterministic and reproducible
- No API dependencies for core extraction
- Web UI with interactive Cytoscape visualization
- Redis-based job queue for scalability
- Sophisticated dependency-based edge detection

**Weaknesses:**
- Lower precision/recall than hybrid approaches
- Requires manual pattern engineering
- Less flexible for new domains
- Limited semantic understanding

### Current Project (aaiaa-singularis)
**Approach:** Hybrid embeddings + LLM validation
**Core Technology:** SciBERT + ChromaDB + Nebius LLM
**Architecture:** 2-stage pipeline (Retrieve→Validate)
**Cost:** $0.018 per paper
**Quality:** 88%+ precision, 82% recall
**Strengths:**
- High precision/recall balance
- Domain-optimized (SciBERT)
- Cost-efficient hybrid approach
- Production-ready CLI + Telegram bot
- Comprehensive metrics tracking

**Weaknesses:**
- Still dependent on LLM API
- Static SVG visualization
- Sequential processing (no multi-document queue)
- Simple heuristic graph assembly
- No coreference resolution

---

## 🎯 Integration Opportunities

### 1. **Hybrid 3-Stage Pipeline: Rules → Embeddings → LLM**

**Concept:** Combine the best of both approaches

```
Stage S1 (Singularis): spaCy Pattern Matching
   ↓ [High-confidence extractions]
Stage S2 (Current): SciBERT Semantic Retrieval
   ↓ [Medium-confidence candidates]
Stage S3 (Current): Nebius LLM Validation
   ↓ [Final validated entities]
```

**Benefits:**
- **Cost reduction:** 40-60% fewer candidates need LLM validation
- **Speed improvement:** Pattern matching is instant
- **Higher precision:** Clear-cut cases handled deterministically
- **Maintained recall:** Semantic retrieval catches what patterns miss

**Implementation:**
```python
# New module: src/extractors/pattern_extractor.py
class PatternExtractor:
    """Stage 1: Deterministic extraction using spaCy patterns"""

    def extract_high_confidence(self, doc: spacy.Doc) -> List[Entity]:
        """Extract obvious entities using dependency patterns"""
        # Port Singularis patterns from /rules/*.json
        # Example: "We hypothesize that..." → HYPOTHESIS
        # Example: "Experiments were conducted..." → EXPERIMENT
        pass

# Modified: src/pipelines/scibert_nebius_pipeline.py
class EnhancedPipeline:
    def extract(self, parsed_doc: ParsedDocument):
        # Stage 1: Pattern-based (FREE)
        s1_entities = self.pattern_extractor.extract_high_confidence(doc)

        # Stage 2: Semantic retrieval for remaining text (FREE)
        remaining_sentences = exclude_covered_by(s1_entities)
        s2_candidates = self.semantic_retriever.retrieve(remaining_sentences)

        # Stage 3: LLM validation only for s2 candidates ($)
        validated = self.validator.validate(s2_candidates)

        return s1_entities + validated
```

**Expected Impact:**
- Cost: $0.018 → **$0.008-0.012** per paper
- Speed: 60-90s → **40-60s**
- Precision: Maintained at 88%+
- Recall: Potential +2-3% improvement

---

### 2. **Advanced Graph Assembly with Dependency Parsing**

**Current Limitation:** Simple Jaccard similarity + section proximity
**Singularis Solution:** Dependency-based edge detection

**Example Pattern (from Singularis):**
```json
{
  "name": "hypothesis_drives_experiment",
  "pattern": [
    {"DEP": "nsubj", "LABEL": "HYPOTHESIS"},
    {"LEMMA": {"IN": ["test", "validate", "verify"]}},
    {"DEP": "dobj", "LABEL": "EXPERIMENT"}
  ]
}
```

**Implementation:**
```python
# New module: src/components/dependency_linker.py
class DependencyLinker:
    """Stage 2 refinement: Syntax-aware relationship detection"""

    def __init__(self):
        self.patterns = load_patterns("rules/edge_patterns.json")
        self.nlp = spacy.load("en_core_web_sm")

    def detect_edges(self, entities: List[Entity], doc: spacy.Doc) -> List[Relationship]:
        """
        Detect relationships using:
        1. Dependency patterns (Singularis approach)
        2. Semantic similarity (current approach)
        3. Section proximity heuristics
        """
        edges = []

        # Method 1: Pattern-based (high precision)
        edges.extend(self._apply_dependency_patterns(entities, doc))

        # Method 2: Embedding similarity (high recall)
        edges.extend(self._semantic_linking(entities))

        # Method 3: Heuristic fallback
        edges.extend(self._proximity_linking(entities))

        return self._deduplicate_edges(edges)
```

**Expected Impact:**
- Relationship recall: 82% → **88-92%**
- Edge precision: +5-8%
- More explicit causal chains (hypothesis→experiment→result)

---

### 3. **Interactive Web UI + Cytoscape Visualization**

**Current:** Static SVG (8-column layout, limited interactivity)
**Singularis:** React + Cytoscape.js (zoom, pan, filter, layout algorithms)

**Proposed Architecture:**
```
/ui/                         # New directory (from Singularis)
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── GraphViewer.tsx      # Cytoscape wrapper
│   │   │   ├── EntityPanel.tsx      # Details sidebar
│   │   │   ├── UploadForm.tsx       # PDF upload
│   │   │   └── MetricsPanel.tsx     # Cost/time stats
│   │   ├── hooks/
│   │   │   └── useGraphData.ts      # API integration
│   │   └── App.tsx
│   └── package.json
├── backend/
│   ├── api/
│   │   ├── gateway.py               # FastAPI server
│   │   ├── routes/
│   │   │   ├── upload.py            # POST /api/upload
│   │   │   ├── jobs.py              # GET /api/jobs/{id}
│   │   │   └── graph.py             # GET /api/graph/{paper_id}
│   │   └── models.py
│   └── worker/
│       └── processor.py             # Background job execution
└── docker-compose.yml
```

**Features:**
- **Upload interface:** Drag-and-drop PDF upload
- **Real-time progress:** WebSocket updates during processing
- **Interactive graph:** Zoom, pan, filter by entity type
- **Layout algorithms:** Force-directed, hierarchical, circular
- **Export:** JSON, PNG, SVG download
- **Multi-document comparison:** Side-by-side graphs

**Technologies:**
- Frontend: React + TypeScript + Cytoscape.js
- Backend: FastAPI + Redis (RQ for jobs)
- Communication: REST API + WebSocket

---

### 4. **Redis Queue System for Scalability**

**Current:** Synchronous processing (CLI, Telegram bot)
**Singularis:** Async worker pool with Redis

**Architecture:**
```python
# backend/worker/processor.py
from rq import Worker, Queue
from redis import Redis

redis_conn = Redis(host='localhost', port=6379)
queue = Queue('paper_processing', connection=redis_conn)

def process_paper_job(pdf_path: str, paper_id: str):
    """Worker function executed in background"""
    parser = GrobidParser()
    pipeline = EnhancedPipeline()

    # Update job status: "parsing"
    parsed = parser.parse(pdf_path)

    # Update job status: "extracting"
    result = pipeline.extract(parsed, paper_id)

    # Update job status: "completed"
    return result.to_dict()

# backend/api/routes/upload.py
@app.post("/api/upload")
async def upload_pdf(file: UploadFile):
    paper_id = generate_id()

    # Save PDF
    pdf_path = save_upload(file, paper_id)

    # Enqueue job
    job = queue.enqueue(
        process_paper_job,
        pdf_path=pdf_path,
        paper_id=paper_id,
        job_timeout='10m'
    )

    return {"job_id": job.id, "paper_id": paper_id}
```

**Benefits:**
- **Scalability:** Multiple workers process papers in parallel
- **Reliability:** Job persistence, retry on failure
- **User experience:** Non-blocking uploads, progress tracking
- **Resource management:** Rate limiting, priority queues

---

### 5. **Enhanced S2 Refinement Stage**

**Singularis S2 Operations:**
1. **Deduplication:** Merge near-identical entities
2. **Normalization:** Standardize entity types
3. **Edge validation:** Check type compatibility
4. **Retyping:** Fix misclassified entities based on relationships
5. **Connectivity check:** Optional LLM refinement if graph is sparse

**Implementation:**
```python
# src/components/graph_refiner.py
class GraphRefiner:
    """Post-processing and refinement (Singularis-inspired)"""

    def refine(self, graph: KnowledgeGraph) -> KnowledgeGraph:
        """Apply multi-stage refinement"""

        # Step 1: Deduplicate entities (coreference resolution)
        graph = self._deduplicate_entities(graph)

        # Step 2: Validate edge type compatibility
        graph = self._validate_edges(graph)

        # Step 3: Retype entities based on relationships
        graph = self._retype_misclassified(graph)

        # Step 4: Check connectivity
        if self._is_weakly_connected(graph):
            # Optional: LLM refinement for sparse graphs
            graph = self._llm_refinement(graph)

        return graph

    def _deduplicate_entities(self, graph: KnowledgeGraph) -> KnowledgeGraph:
        """Merge entities with high text similarity"""
        # Use sentence-transformers for semantic similarity
        # Merge if similarity > 0.92 and same type
        pass

    def _retype_misclassified(self, graph: KnowledgeGraph) -> KnowledgeGraph:
        """Fix entity types based on relationship context"""
        # Example: "RESULT" connected only to "HYPOTHESIS"
        #          might actually be "EXPERIMENT"
        pass
```

**Expected Impact:**
- Graph coherence: +10-15%
- Duplicate entities: -30-40%
- Relationship accuracy: +5-7%

---

## 🚀 Implementation Roadmap

### Phase 1: Core Integration (2-3 weeks)
**Goal:** Integrate pattern-based extraction (Stage S1)

**Tasks:**
1. Port Singularis spaCy patterns to `/rules/` directory
2. Implement `PatternExtractor` module
3. Modify `SciBertNebiusPipeline` to support 3-stage flow
4. Benchmark cost/quality improvements
5. Update CLI tool to support `--use-patterns` flag

**Deliverables:**
- `src/extractors/pattern_extractor.py`
- `rules/entity_patterns.json`
- Updated pipeline with 40%+ cost reduction

### Phase 2: Graph Refinement (1-2 weeks)
**Goal:** Improve relationship detection

**Tasks:**
1. Implement `DependencyLinker` with edge patterns
2. Port Singularis edge detection rules
3. Implement `GraphRefiner` for S2 refinement
4. Add deduplication and retyping logic
5. Benchmark relationship recall improvements

**Deliverables:**
- `src/components/dependency_linker.py`
- `src/components/graph_refiner.py`
- `rules/edge_patterns.json`
- +6-10% relationship recall

### Phase 3: Web UI (3-4 weeks)
**Goal:** Interactive visualization and job queue

**Tasks:**
1. Set up FastAPI backend with Redis
2. Implement upload and job management endpoints
3. Create React frontend with Cytoscape.js
4. Add WebSocket for real-time updates
5. Implement multi-document upload and comparison
6. Deploy with Docker Compose

**Deliverables:**
- `/ui/frontend/` - React app
- `/ui/backend/` - FastAPI + Redis
- `docker-compose.yml`
- Web interface at `localhost:3000`

### Phase 4: Testing & Optimization (1-2 weeks)
**Goal:** Production readiness

**Tasks:**
1. End-to-end testing with 50+ papers
2. Performance optimization (caching, batching)
3. Documentation updates
4. User testing and feedback
5. Deployment guide

**Deliverables:**
- Test suite with 50+ papers
- Performance benchmarks
- Updated README and CLAUDE.md
- Deployment documentation

---

## 📈 Expected Outcomes

### Cost Efficiency
| Metric | Current | After Integration | Improvement |
|--------|---------|-------------------|-------------|
| Cost per paper | $0.018 | $0.008-0.012 | **40-55%** |
| LLM API calls | ~150 | ~60-80 | **47-60%** |
| Processing time | 60-90s | 40-60s | **30-35%** |

### Quality Metrics
| Metric | Current | After Integration | Improvement |
|--------|---------|-------------------|-------------|
| Entity Precision | 88% | 88-90% | **+0-2%** |
| Entity Recall | 82% | 84-86% | **+2-4%** |
| Relationship Precision | ~75% | 80-83% | **+5-8%** |
| Relationship Recall | ~82% | 88-92% | **+6-10%** |
| Graph Coherence | ~70% | 80-85% | **+10-15%** |

### User Experience
| Feature | Current | After Integration |
|---------|---------|-------------------|
| Visualization | Static SVG | Interactive Cytoscape |
| Upload Interface | CLI/Telegram | Web UI + drag-drop |
| Multi-document | Sequential | Parallel processing |
| Progress Tracking | None | Real-time WebSocket |
| Export Formats | JSON, SVG | JSON, SVG, PNG, CSV |

---

## 🔧 Technical Considerations

### Compatibility
- **spaCy version:** Ensure compatibility between Singularis patterns and current spaCy version
- **Pattern format:** Convert Singularis JSON patterns to match our entity schema
- **API design:** RESTful endpoints compatible with existing CLI/bot

### Infrastructure
- **Redis:** Add to `requirements.txt` and deployment docs
- **Docker:** Multi-container setup (API, Worker, Frontend, Redis)
- **GROBID:** Shared dependency (already used)

### Configuration
- Add to `scibert_nebius_config.yaml`:
```yaml
pattern_extraction:
  enabled: true
  confidence_threshold: 0.85  # High confidence for pattern matches
  patterns_path: "rules/entity_patterns.json"

dependency_linking:
  enabled: true
  patterns_path: "rules/edge_patterns.json"

graph_refinement:
  deduplication_threshold: 0.92
  enable_retyping: true
  llm_refinement_threshold: 0.3  # If connectivity < 30%, use LLM
```

---

## 🎯 Quick Wins (High ROI, Low Effort)

### 1. Pattern-based HYPOTHESIS extraction
**Effort:** 2-3 days
**Impact:** Immediate 30% cost reduction for HYPOTHESIS entities

**Implementation:**
```python
# Simple regex patterns for obvious cases
HYPOTHESIS_PATTERNS = [
    r"we hypothesi[zs]e that",
    r"our hypothesis is",
    r"we propose that",
    r"we predict that",
]

def quick_hypothesis_extraction(text: str) -> List[str]:
    """Extract obvious hypotheses before semantic retrieval"""
    matches = []
    for pattern in HYPOTHESIS_PATTERNS:
        matches.extend(re.finditer(pattern, text, re.IGNORECASE))
    return [extract_sentence(m) for m in matches]
```

### 2. Dependency-based "tests" relationship
**Effort:** 1-2 days
**Impact:** +10% relationship recall for hypothesis→experiment links

**Pattern:**
```json
{
  "name": "hypothesis_tests_experiment",
  "pattern": [
    {"ENT_TYPE": "HYPOTHESIS"},
    {"LEMMA": {"IN": ["test", "evaluate", "verify", "validate", "assess"]}},
    {"ENT_TYPE": "EXPERIMENT"}
  ]
}
```

### 3. Entity deduplication
**Effort:** 2-3 days
**Impact:** -25% duplicate entities, cleaner graphs

**Implementation:**
```python
from sentence_transformers import util

def deduplicate_entities(entities: List[Entity], threshold=0.92) -> List[Entity]:
    """Merge near-duplicate entities"""
    embeddings = [entity.embedding for entity in entities]
    similarity = util.cos_sim(embeddings, embeddings)

    merged = []
    seen = set()

    for i, entity in enumerate(entities):
        if i in seen:
            continue

        duplicates = [j for j in range(i+1, len(entities))
                      if similarity[i][j] > threshold and entities[j].type == entity.type]

        if duplicates:
            # Merge duplicates into primary entity
            merged.append(merge_entities(entity, [entities[j] for j in duplicates]))
            seen.update(duplicates)
        else:
            merged.append(entity)

    return merged
```

---

## 📚 References

### Singularis Resources
- **GitHub:** https://github.com/lo-dn/singularis
- **Key files:**
  - `/pipeline/stages/s1_extract.py` - Pattern-based extraction
  - `/pipeline/stages/s2_refine.py` - Deduplication and validation
  - `/rules/*.json` - spaCy pattern definitions
  - `/ui/` - React frontend with Cytoscape

### Current Project Resources
- **GitHub:** IvanArtemov/aaiaa-singularis
- **Key files:**
  - `src/pipelines/scibert_nebius_pipeline.py` - Main pipeline
  - `src/components/semantic_retriever.py` - Embedding-based retrieval
  - `src/components/entity_validator.py` - LLM validation
  - `src/components/graph_assembler.py` - Heuristic linking

---

## ✅ Decision Points

### Priority Questions
1. **Phase priority:** Start with cost optimization (Phase 1) or user experience (Phase 3)?
2. **Pattern scope:** Port all Singularis patterns or start with high-ROI entities (HYPOTHESIS, EXPERIMENT)?
3. **Web UI timing:** Develop in parallel with core improvements or as final polish?
4. **Breaking changes:** Acceptable to modify pipeline API or maintain backward compatibility?

### Resource Allocation
- **Development time:** 7-11 weeks total (phases 1-4)
- **Infrastructure:** Redis server, Docker hosting
- **Testing:** Dataset of 50-100 papers for validation

---

## 🎓 Learning from Singularis

### Key Takeaways
1. **Rules aren't obsolete:** Pattern matching still provides 80% quality at 100× lower cost
2. **Hybrid is optimal:** Combine rules (precision) + embeddings (recall) + LLM (validation)
3. **UX matters:** Interactive visualization dramatically improves usability
4. **Scalability design:** Job queues enable production deployment
5. **Incremental refinement:** Multi-stage pipelines (S1→S2→S3) allow quality/cost trade-offs

### What NOT to Copy
1. **100% rules-based:** Too brittle for scientific diversity
2. **No embeddings:** Miss semantic relationships
3. **Optional LLM:** In our case, LLM is necessary for 88%+ precision

### What to Adapt
1. **Pattern-based first pass:** FREE obvious entity extraction
2. **Dependency parsing:** Syntax-aware relationship detection
3. **S2 refinement:** Deduplication, validation, retyping
4. **Web infrastructure:** FastAPI + Redis + React
5. **Interactive visualization:** Cytoscape.js

---

## 🚦 Next Steps

### Immediate Actions (This Week)
1. **Clone Singularis repo:** `git clone https://github.com/lo-dn/singularis`
2. **Study patterns:** Review `/rules/*.json` structure
3. **Prototype S1:** Quick implementation of pattern extraction
4. **Benchmark:** Test cost/quality with 10 papers

### Short-term (Next Month)
1. **Implement Phase 1:** Pattern-based extraction
2. **Benchmark Phase 1:** Measure cost reduction
3. **Plan Phase 2:** Design dependency linker architecture

### Long-term (Next Quarter)
1. **Complete Phases 2-3:** Graph refinement + Web UI
2. **Production deployment:** Docker Compose setup
3. **User testing:** Gather feedback, iterate
4. **Documentation:** Update guides for new features

---

**Status:** Ready for review and decision
**Next:** Discuss priorities and begin Phase 1 prototyping

---

*Document created by Claude Code on 2025-11-11*
*Based on analysis of lo-dn/singularis and IvanArtemov/aaiaa-singularis*
