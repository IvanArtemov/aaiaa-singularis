# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## 📋 Project Overview

**Project:** SciBERT-Nebius Knowledge Graph Extractor
**Based on:** Singularis Challenge - Scientific Publishing Reform
**Hackathon:** Agentic AI Against Aging (https://www.hackaging.ai/)

### Mission
Create a **cost-efficient** system for extracting structured information from scientific papers and building knowledge graphs, where papers are represented as graphs of interconnected elements.

### Extracted Entity Types
1. **Input Fact** - Established knowledge entering the research
2. **Hypothesis** - Scientific assumptions to be tested
3. **Experiment** - Procedures for testing hypotheses
4. **Technique/Method** - Methods and tools used
5. **Result** - Data and observations obtained
6. **Dataset** - Data collections used or created
7. **Analysis** - Statistical/computational processing
8. **Conclusion** - Interpretations and findings

### Key Relationships
- Hypothesis → tested by → Experiment
- Result → analyzed using → Analysis
- Conclusion → based on → Result
- Method → applied in → Experiment

---

## 🚀 Quick Start

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Set up environment variables
cp .env.example .env
# Edit .env and add:
# - TELEGRAM_BOT_TOKEN (from @BotFather)
# - NEBIUS_API_KEY (from https://studio.nebius.com/)
```

### Running Commands

#### CLI Tool (Recommended for Single PDF Processing)
```bash
# Process single PDF - produces JSON + SVG output
python scripts/process_pdf.py --pdf paper.pdf

# Custom output directory
python scripts/process_pdf.py -p paper.pdf -o results

# Skip SVG generation (faster)
python scripts/process_pdf.py -p paper.pdf --no-svg

# Verbose mode for debugging
python scripts/process_pdf.py -p paper.pdf -v
```

**Output files:**
- `{paper_id}_entities.json` - Entities and relationships
- `{paper_id}_metrics.json` - Performance metrics (time, cost, tokens)
- `{paper_id}_graph.svg` - Visual knowledge graph (8-column layout)

#### Example Scripts
```bash
# Test SciBERT-Nebius pipeline with pre-parsed XML
python scripts/example_scibert_nebius_pipeline.py

# Run Telegram Bot
python scripts/run_scibert_telegram_bot.py
```

---

## 🏗️ Architecture

### SciBERT-Nebius Pipeline

**Hybrid Architecture:**
- **SciBERT embeddings** (FREE, domain-optimized, 768 dims)
- **Nebius gpt-oss-120b LLM** (cost-efficient, $0.15/$0.60 per 1M tokens)
- **ChromaDB** semantic search (FREE, local)
- **GROBID** ML parser (FREE, structured IMRAD extraction)

**Pipeline Flow:**
```
PDF → GROBID Parser → IMRAD sections
                    ↓
            SciBERT Embeddings (FREE)
                    ↓
            Keyword Generation (Nebius, ~$0.003)
                    ↓
            Semantic Retrieval (ChromaDB, FREE)
                    ↓
            Entity Validation (Nebius, ~$0.015)
                    ↓
            Graph Assembly (Heuristics, FREE)
                    ↓
            SVG Visualization
```

**Cost:** ~$0.018 per paper
**Target Precision:** ≥88%
**Target Recall:** ≥82%

---

## 📦 Project Structure

```
AAIAA/
├── bot/
│   ├── __init__.py
│   ├── exceptions.py
│   ├── scibert_config.py
│   ├── scibert_handlers.py
│   ├── scibert_telegram_bot.py
│   ├── session_manager.py
│   └── utils.py
├── scripts/
│   ├── process_pdf.py                      # CLI tool for single PDF
│   ├── example_scibert_nebius_pipeline.py  # Example with pre-parsed XML
│   └── run_scibert_telegram_bot.py         # Telegram bot
├── src/
│   ├── components/
│   │   ├── entity_validator.py      # LLM-based validation
│   │   ├── graph_assembler.py        # Heuristic relationships
│   │   └── semantic_retriever.py     # ChromaDB search
│   ├── config/
│   │   ├── grobid_config.yaml
│   │   ├── scibert_nebius_config.yaml
│   │   └── settings.py
│   ├── embedding_adapters/
│   │   ├── base_embedding_adapter.py
│   │   ├── factory.py
│   │   └── scibert_adapter.py        # FREE SciBERT embeddings
│   ├── extractors/
│   │   ├── keyword_generator.py      # LLM keyword generation
│   │   └── sentence_embedder.py      # Sentence splitting + embeddings
│   ├── llm_adapters/
│   │   ├── base_adapter.py
│   │   ├── factory.py
│   │   └── nebius_adapter.py         # Nebius AI Studio
│   ├── models/
│   │   ├── entities.py               # Entity, EntityType, etc.
│   │   ├── graph.py                  # KnowledgeGraph
│   │   ├── results.py                # ExtractionResult, Metrics
│   │   └── sentence.py               # Sentence with embeddings
│   ├── parsers/
│   │   ├── base_parser.py
│   │   └── grobid_parser.py          # ML-based IMRAD extraction
│   ├── pipelines/
│   │   ├── base_pipeline.py
│   │   └── scibert_nebius_pipeline.py
│   └── visualization/
│       └── generate_svg.py           # SVG graph generation
├── .env
├── CLAUDE.md
├── README.md
└── requirements.txt
```

---

## 💻 Key Modules

### CLI Tool
**File:** `scripts/process_pdf.py`

Command-line tool for processing single PDFs:
- Parses PDF with GROBID
- Runs SciBERT-Nebius pipeline
- Saves JSON results (entities, metrics)
- Generates SVG visualization

**Usage:**
```bash
python scripts/process_pdf.py --pdf paper.pdf -o results
```

### SciBERT-Nebius Pipeline
**File:** `src/pipelines/scibert_nebius_pipeline.py`

Main extraction pipeline combining:
- SciBERT for FREE domain-optimized embeddings
- Nebius gpt-oss-120b for cost-efficient LLM processing
- ChromaDB for semantic candidate retrieval
- Parallel entity validation (4 threads)

**Usage:**
```python
from src.pipelines import SciBertNebiusPipeline
from src.parsers import GrobidParser

# Parse PDF
parser = GrobidParser()
parsed_doc = parser.parse("paper.pdf")

# Extract entities
pipeline = SciBertNebiusPipeline()
result = pipeline.extract(parsed_doc, paper_id="paper123")

# Access results
print(f"Entities: {result.total_entities()}")
print(f"Relationships: {result.total_relationships()}")
print(f"Cost: ${result.metrics.cost_usd:.4f}")
```

### Telegram Bot
**File:** `bot/scibert_telegram_bot.py`

PDF to Knowledge Graph Telegram Bot:
- Accepts PDF uploads
- Processes with SciBERT-Nebius pipeline
- Returns SVG knowledge graph
- Rate limiting + session management

**Features:**
- `/start` - Welcome message
- `/help` - Instructions
- `/stats` - User statistics
- PDF upload → automatic processing → SVG graph

---

## 🛠️ Tech Stack

### Core
- **Python 3.10+**
- **OpenAI SDK** - Used by Nebius adapter (OpenAI-compatible API)
- **grobid-client-python** - ML-based PDF extraction
- **spacy** - Sentence splitting (en_core_web_sm model)
- **pyyaml** - Configuration
- **python-dotenv** - Environment variables

### Pipeline
- **transformers** - SciBERT model
- **torch** - SciBERT inference
- **chromadb** - Vector database
- **scikit-learn** - Utilities
- **numpy** - Vector operations

### Telegram Bot
- **python-telegram-bot** - Telegram API
- **aiofiles** - Async file ops

---

## 🔑 Configuration

### Environment Variables (`.env`)
```bash
# Required
TELEGRAM_BOT_TOKEN=your_bot_token_from_botfather
NEBIUS_API_KEY=your_nebius_api_key

# Optional
GROBID_URL=https://lfoppiano-grobid.hf.space
```

### Pipeline Config
**File:** `src/config/scibert_nebius_config.yaml`

Configure:
- Embedding batch sizes
- Keyword generation
- Semantic retrieval top-k per entity type
- Validation confidence thresholds
- Graph assembly settings

---

## 📊 Metrics

### Target Metrics
- **Precision:** ≥88%
- **Recall:** ≥82%
- **F1-score:** ≥85%
- **Cost:** ~$0.018 per paper
- **Speed:** 60-90 seconds

### Cost Breakdown
- **Embeddings (SciBERT):** $0.000 (FREE)
- **Keyword Generation:** ~$0.003 (Nebius)
- **Entity Validation:** ~$0.015 (Nebius)
- **Total:** ~$0.018 per paper

---

## 🎯 Development Guidelines

### Code Style
- Use type hints
- Document complex logic
- Keep functions focused
- Follow existing patterns

### Cost Optimization
- Minimize LLM calls (batching)
- Cache repeated operations
- Use SciBERT (FREE) over API embeddings
- Prefer heuristics over LLM

### Testing
- Test pipeline with `scripts/example_scibert_nebius_pipeline.py`
- Use `process_pdf.py --verbose` for debugging
- Check output JSON and SVG files for correctness

---

**Last Updated:** October 22, 2025
**Status:** Production
