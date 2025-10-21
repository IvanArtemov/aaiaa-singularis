# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## 📋 О проекте

**Проект:** SciBERT-Nebius Knowledge Graph Extractor
**Основано на:** Singularis Challenge - Реформа научного публикования
**Хакатон:** Agentic AI Against Aging (https://www.hackaging.ai/)

### Миссия
Создать **cost-efficient** систему для извлечения структурированной информации из научных статей и построения knowledge graph, где статьи представлены как графы взаимосвязанных элементов.

### Структура извлекаемых элементов
1. **Input Fact** - Установленное знание, входящее в исследование
2. **Hypothesis** - Научное предположение для проверки
3. **Experiment** - Процедура тестирования гипотезы
4. **Technique/Method** - Используемые методы и инструменты
5. **Result** - Полученные данные и наблюдения
6. **Dataset** - Использованные или созданные коллекции данных
7. **Analysis** - Статистическая/вычислительная обработка
8. **Conclusion** - Интерпретации и выводы

### Ключевые связи
- Hypothesis → tested by → Experiment
- Result → analyzed using → Analysis
- Conclusion → based on → Result
- Method → applied in → Experiment

---

## 🚀 Быстрый старт

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

```bash
# Test SciBERT-Nebius pipeline directly
python scripts/example_scibert_nebius_pipeline.py

# Run Telegram Bot
python scripts/run_scibert_telegram_bot.py
```

---

## 🏗️ Архитектура

### SciBERT-Nebius Pipeline

**Упрощенная гибридная архитектура:**
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

## 📦 Структура проекта

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
│   ├── example_scibert_nebius_pipeline.py
│   └── run_scibert_telegram_bot.py
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

## 🛠️ Технический стек

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

## 📊 Метрики

### Целевые показатели
- **Precision:** ≥88%
- **Recall:** ≥82%
- **F1-score:** ≥85%
- **Стоимость:** ~$0.018 на статью
- **Скорость:** 60-90 секунд

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

---

**Last Updated:** October 21, 2025  
**Status:** Production (Simplified)
