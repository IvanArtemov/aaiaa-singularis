# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## 📋 О проекте

**Проект:** Singularis Challenge - Реформа научного публикования
**Хакатон:** Agentic AI Against Aging (https://www.hackaging.ai/)
**Дедлайн:** 22 октября 2025, 11:59 PM PT (Code Freeze)
**Призовой фонд:** $20,000

### Миссия Singularis
Изменить способ взаимодействия ученых со знаниями. Минимальная публикуемая единица должна быть **меньше научной статьи** — это может быть одна гипотеза, эксперимент, метод, результат или датасет.

### Задача
Создать **cost-efficient** систему для извлечения структурированной информации из **50 миллионов научных статей** и построения knowledge graph, где статьи представлены как графы взаимосвязанных элементов.

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

## 🔑 Критические требования

### ⚡ ЭКОНОМИЧНОСТЬ (ГЛАВНЫЙ ПРИОРИТЕТ!)
- Минимальная стоимость обработки одной статьи
- Целевая метрика: **< $0.05 на статью**
- Система должна масштабироваться на **миллионы статей**
- ❌ НЕ использовать чисто LLM-подход
- ✅ Использовать гибридный подход: LLM + regex + NLP + heuristics

### 📊 Целевые метрики
- **Precision:** ≥ 85%
- **Recall:** ≥ 80%
- **F1-score:** ≥ 82%
- **Стоимость:** < $0.05 на статью
- **Скорость:** > 100 статей/час
- **Масштаб:** Проектируемость на 50M статей

---

## 💻 Development Commands

### Installation
```bash
# Install all dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Running Tests
```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run only integration tests
pytest tests/integration/ -v

# Run with coverage report
pytest --cov=src --cov-report=term-missing

# Generate HTML coverage report
pytest --cov=src --cov-report=html
# Open htmlcov/index.html in browser
```

### Running Examples

```bash
# Test LLM adapters (OpenAI/Ollama)
python scripts/example_adapters.py

# Test PubMed fetcher
python scripts/example_fetchers.py

# Test arXiv fetcher
python scripts/example_arxiv.py

# Download PDFs from PubMed
python scripts/download_pdfs_demo.py

# Extract PDFs from PMC packages
python scripts/extract_pdfs_from_packages.py

# Batch download by topic (PubMed)
python scripts/batch_download_cross_referenced.py

# Batch download KG papers (arXiv)
python scripts/batch_download_arxiv_kg.py

# Test PDF parser
python scripts/example_pdf_parser.py

# Test LLM extraction pipeline
python scripts/example_llm_pipeline.py

# Generate SVG knowledge graph
python scripts/generate_svg.py results/sample_result.json
```

### Configuration

**Switch LLM Provider (OpenAI ↔ Ollama):**
Edit `src/config/llm_config.yaml`:
```yaml
active_provider: "openai"  # or "ollama"
```

**PubMed API Key (optional but recommended):**
Add to `.env`:
```
NCBI_API_KEY=your_api_key_here
```
This increases rate limit from 3 req/sec to 10 req/sec.

---

## 🏗️ Code Architecture

### High-Level Design Principles

**1. Factory Pattern for Extensibility**
- `get_llm_adapter(provider)` - Create LLM adapters (OpenAI, Ollama, etc.)
- `get_fetcher(type)` - Create paper fetchers (PubMed, PMC, etc.)
- `get_parser(format)` - Create document parsers (PDF, TXT, HTML)

**2. Pipeline Abstraction**
All extraction pipelines inherit from `BasePipeline`:
- `LLMPipeline` - High-quality extraction using GPT (~$0.03-$0.30/paper)
- `RegexPipeline` - Pattern-based extraction (free, lower quality)
- `HybridPipeline` - Optimal balance (~$0.02/paper target)

Each pipeline implements:
```python
def extract(paper_text: str, paper_id: str) -> ExtractionResult
def get_metrics() -> PipelineMetrics
def get_description() -> str
def get_estimated_cost() -> float
```

**3. Type-Safe Data Models**
- `Entity` - Structured entity with `EntityType` enum
- `Relationship` - Typed relationship with `RelationshipType` enum
- `KnowledgeGraph` - Collection of entities and relationships
- `ExtractionResult` - Complete pipeline output with metrics

**4. Configuration Management**
- YAML-based configs: `src/config/llm_config.yaml`, `fetcher_config.yaml`
- Environment variables for API keys
- `Settings` class centralizes configuration access

**5. Modular Component Design**
```
Input → Parser → Pipeline → Extractor → Model → Validator → Output
```

---

## 📦 Key Modules

### Core Data Models (`src/models/`)

**`entities.py`** - Core data structures:
- `EntityType` enum: FACT, HYPOTHESIS, EXPERIMENT, TECHNIQUE, RESULT, DATASET, ANALYSIS, CONCLUSION
- `RelationshipType` enum: HYPOTHESIS_TO_EXPERIMENT, METHOD_TO_RESULT, etc.
- `Entity` class: id, type, text, confidence, source_section, metadata
- `Relationship` class: source_id, target_id, relationship_type, confidence

**`graph.py`** - Knowledge graph structure:
- `KnowledgeGraph` class: paper_id, entities, relationships
- Conversion to NetworkX graphs for visualization

**`results.py`** - Pipeline outputs:
- `ExtractionResult`: paper_id, entities (grouped by type), relationships, metrics
- `PipelineMetrics`: processing_time, tokens_used, cost_usd, entities_extracted

### Extraction Pipelines (`src/pipelines/`)

**`base_pipeline.py`** - Abstract base class:
- Defines contract for all extraction pipelines
- Standardized interface for metrics collection

**`llm_pipeline.py`** - LLM-based extraction:
- Uses GPT-4o-mini (or configurable model) via OpenAI SDK
- Structured JSON output with few-shot prompting
- Cost: ~$0.03/paper (GPT-4o-mini) or ~$0.30/paper (GPT-4)
- Use case: Ground truth generation, high-quality baseline

### LLM Adapters (`src/llm_adapters/`)

**Factory-based LLM abstraction:**
- `base_adapter.py` - Abstract interface
- `openai_adapter.py` - OpenAI/ChatGPT implementation
- `ollama_adapter.py` - Local Ollama implementation
- `factory.py` - `get_llm_adapter(provider)` factory function

**Usage:**
```python
from src.llm_adapters import get_llm_adapter

llm = get_llm_adapter("openai")  # or "ollama"
result = llm.generate(prompt="...", system_prompt="...")
embeddings = llm.embed(["text1", "text2"])
```

### Paper Fetchers (`src/fetchers/`)

**Multi-source paper fetching:**
- `base_fetcher.py` - Abstract fetcher interface
- `pubmed_fetcher.py` - PubMed API implementation (E-utilities)
- `arxiv_fetcher.py` - arXiv API implementation (arxiv.py)
- `factory.py` - `get_fetcher(type)` factory

**Features:**
- Search by query: `fetcher.search("aging research", max_results=10)`
- Fetch metadata: `paper = fetcher.fetch_paper(pmid)` or `fetcher.fetch_paper(arxiv_id)`
- Download PDFs: Full-text PDF download (PubMed PMC, arXiv)
- Category search: `arxiv_fetcher.search_by_category(["cs.CL", "cs.AI"])`
- Article registry: Track downloaded papers in `articles/metadata.json`

**Supported Sources:**
- **PubMed/PMC:** Biomedical literature (NCBI E-utilities API)
- **arXiv:** Preprints in physics, CS, math, biology, etc.

### Document Parsers (`src/parsers/`)

**Multi-format document parsing:**
- `base_parser.py` - Abstract parser interface
- `pdf_parser.py` - PDF parsing using PyMuPDF (fitz)
  - Text extraction with layout preservation
  - Section detection (Abstract, Methods, Results, etc.)
  - Metadata extraction (title, authors, dates)
  - Optional table extraction via pdfplumber

**Section Detection:**
Automatically detects common paper sections using regex patterns:
- Abstract, Introduction, Methods, Results, Discussion, Conclusion, References

### Visualization (`src/visualization/`)

**`generate_svg.py`** - SVG knowledge graph generator:
- Hierarchical layout with entity types in columns
- Color-coded entities and relationships
- Bezier curve edges with arrow markers
- XML-safe text escaping
- Auto-sizing based on content

**Usage:**
```bash
python -m src.visualization.generate_svg results/output.json graph.svg
```

### Utilities (`src/utils/`)

**`article_registry.py`** - Article metadata tracking:
- SQLite-like JSON registry for downloaded papers
- Track PMID, arXiv ID, PMC ID, DOI, PDF path, download source
- Statistics: total articles, size, source breakdown
- Deduplication and lookup by any identifier (PMID, arXiv ID, PMC ID, DOI)

---

## 💡 Cost Optimization Strategy

### Three-Pipeline Approach

**1. LLM Pipeline (Ground Truth)**
- Model: GPT-4 or GPT-4o-mini
- Cost: $0.03-$0.30 per paper
- Precision: ~95% (expected)
- Use: Create 10-15 annotated papers as ground truth

**2. Regex Pipeline (Baseline)**
- Cost: $0.00 (CPU only)
- Precision: ~60-70% (expected)
- Speed: 200-300 papers/hour
- Use: Fast processing, simple pattern matching

**3. Hybrid Pipeline (Production Target)**
- Cost: ~$0.02 per paper
- Precision: ≥85% (target)
- Strategy:
  1. Regex for simple patterns (Methods, Results)
  2. NLP (spaCy) for entity recognition (Facts)
  3. Selective LLM for complex reasoning (Hypotheses, Conclusions)

**Decision Algorithm:**
```
If regex confidence > 0.8:
    Use regex result (FREE)
Elif entity_type in [facts, techniques]:
    Use NLP extractor (~$0.001/paper)
Elif entity_type in [hypotheses, conclusions]:
    Use LLM selectively (~$0.01/paper)
```

### Optimization Techniques
1. **Batch processing** - Combine multiple sections into single API call
2. **Caching** - LRU cache for identical text segments
3. **Model selection** - GPT-4o-mini instead of GPT-4 (20x cheaper)
4. **Chunking** - Process only relevant sections, not full papers

---

## 🛠️ Технический стек

### Core (Existing)
- **Python 3.10+**
- **OpenAI SDK** - GPT-4o-mini for LLM extraction
- **PyMuPDF (fitz)** - PDF text extraction
- **pdfplumber** - PDF table extraction
- **requests** - HTTP client for API calls
- **arxiv** - arXiv API wrapper for paper fetching
- **python-dotenv** - Environment variable management
- **pyyaml** - YAML configuration parsing

### Testing
- **pytest** - Test framework
- **pytest-cov** - Coverage reporting

### Future Dependencies (Planned)
```python
# Will be added as needed:
chromadb>=0.4.0          # Vector database
streamlit>=1.28.0        # Web UI
spacy>=3.7.0             # NLP for hybrid pipeline
scispacy>=0.5.0          # Scientific text processing
networkx>=3.2.0          # Graph processing
plotly>=5.17.0           # Interactive visualizations
```

---

## 🎯 Критерии оценки жюри

### 1. Полнота и Точность (25%)
- **Precision:** % корректно извлеченных элементов
- **Recall:** % найденных элементов от всех существующих
- **F1 Score:** Гармоническое среднее precision и recall

### 2. Робастность (25%)
- **Форматы:** Обработка PDF, HTML, XML
- **Стабильность:** Работа с разными журналами и стилями написания
- **Error Handling:** Восстановление после ошибок

### 3. Стоимостный Анализ (25%)
- **CPU/GPU часы:** Использованные вычислительные ресурсы
- **$/статья:** Общая стоимость обработки
- **Tokens used:** Метрики потребления API
- **Масштабируемость:** Проекция на 50M статей

### 4. Производительность (25%)
- **Throughput:** Статей обработано в час
- **Latency:** Время отклика
- **Parallelization:** Горизонтальное масштабирование

---

## ⭐ BONUS POINTS

1. **Алгоритмические или гибридные решения** с значительным снижением стоимости при сохранении качества

2. **Улучшение концептуального фреймворка:**
   - Новые техники извлечения
   - Улучшенное определение связей
   - Оптимизированные структуры графа
   - Креативные стратегии оптимизации стоимости

---

## 📝 Требования к подаче

✅ **Обязательно:**
- 🎥 **Видео-демо** (3-5 минут)
- 💻 **Открытый репозиторий** с README
- 🌐 **Развернутое решение** (публичный URL)
- 📄 **Описание подхода** - архитектура, методология
- 📊 **Performance metrics** - Precision, Recall, F1, Throughput, Cost
- 💰 **Cost analysis** - Детальная разбивка

⚠️ **Важно:** Жюри НЕ будет запускать код локально!

---

## 🔗 Дополнительная документация

- **Полная спецификация:** `docs/singularis_project_doc.md`
- **Pipeline архитектура:** `docs/pipeline_architecture_plan.md`
- **PubMed API reference:** `docs/pubmed_api_reference.md`

---

## 📞 Поддержка

- **Discord:** #singularis-challenge
- **Менторы:** Доступны через Discord
- **Вопросы:** DM @HackAging.ai

---

**Последнее обновление:** 11 октября 2025
**Статус:** Active Development - Week 1 (MVP)
