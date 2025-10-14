# Pipeline Architecture Plan - Singularis Challenge

**Дата создания:** 10 октября 2025
**Статус:** Approved for Implementation
**Цель:** Cost-efficient извлечение структурированной информации из научных статей

---

## 🎯 Стратегия

### Ключевая идея
Использовать **модульную архитектуру с взаимозаменяемыми пайплайнами**:

1. **LLM Pipeline (v1)** → Создание ground truth датасета (GPT-4, ~$0.30/статья)
2. **Regex Pipeline (v2)** → Паттерн-based извлечение (бесплатно)
3. **Hybrid Pipeline (v3)** → Оптимальное сочетание (Regex + NLP + selective LLM, ~$0.02/статья)

### Обоснование подхода

**Почему сначала LLM?**
- Создание высококачественного ground truth датасета для валидации
- Единоразовые затраты на разметку 10-15 статей
- Объективная метрика для оценки дешевых пайплайнов

**Почему модульность?**
- Возможность A/B тестирования разных подходов
- Итеративное улучшение каждого компонента
- Демонстрация алгоритмической оптимизации (Bonus Points)

---

## 🏗️ Архитектура системы

```
┌──────────────────────────────────────────────────┐
│          Streamlit Web Interface                 │
│   [Upload] [Select Pipeline] [View Results]      │
└────────────────────┬─────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────┐
│         Pipeline Orchestrator                    │
│  • Pipeline Registry                             │
│  • Results Management                            │
│  • Metrics Collection                            │
└────────────────────┬─────────────────────────────┘
                     │
         ┌───────────┼──────────┐
         │           │          │
    ┌────▼────┐ ┌───▼────┐ ┌──▼──────┐
    │  LLM    │ │ Regex  │ │ Hybrid  │
    │Pipeline │ │Pipeline│ │Pipeline │
    └────┬────┘ └───┬────┘ └──┬──────┘
         │          │          │
         └──────────┼──────────┘
                    │
         ┌──────────▼──────────┐
         │  Validation Layer   │
         │  • Ground Truth DB  │
         │  • Metrics Computer │
         └─────────────────────┘
```

---

## 📁 Структура проекта (дополнения)

### Существующие модули ✅
```
src/
├── llm_adapters/          # Адаптеры для LLM (OpenAI, Ollama)
├── fetchers/              # Загрузка статей (PubMed)
├── config/                # Конфигурация
└── utils/                 # Утилиты
```

### Новые модули 🆕
```
src/
├── parsers/               # 🆕 Парсинг документов
│   ├── base_parser.py           # Абстрактный класс
│   ├── pdf_parser.py            # PDF → text (PyMuPDF, pdfplumber)
│   ├── txt_parser.py            # TXT → text
│   ├── html_parser.py           # HTML → text (BeautifulSoup)
│   └── factory.py               # Parser factory
│
├── pipelines/             # 🆕 КЛЮЧЕВОЙ МОДУЛЬ
│   ├── base_pipeline.py         # Абстрактный класс Pipeline
│   ├── llm_pipeline.py          # v1: GPT-4 для ground truth
│   ├── regex_pipeline.py        # v2: Regex-based
│   ├── hybrid_pipeline.py       # v3: Hybrid (Regex + NLP + LLM)
│   ├── registry.py              # Регистрация пайплайнов
│   └── orchestrator.py          # Управление пайплайнами
│
├── extractors/            # 🆕 Компоненты извлечения
│   ├── regex_extractor.py       # Regex паттерны
│   ├── nlp_extractor.py         # spaCy/scispaCy NER
│   ├── section_detector.py      # Детекция секций статьи
│   └── entity_linker.py         # Связи между сущностями
│
├── models/                # 🆕 Модели данных
│   ├── entities.py              # Fact, Hypothesis, Experiment, etc.
│   ├── graph.py                 # Knowledge graph structure
│   └── results.py               # ExtractionResult, PipelineMetrics
│
├── validation/            # 🆕 Валидация и метрики
│   ├── ground_truth.py          # Ground truth database
│   ├── metrics.py               # Precision, Recall, F1
│   └── comparator.py            # Сравнение пайплайнов
│
└── storage/               # 🆕 Хранение результатов
    ├── results_db.py            # SQLite для результатов
    └── chroma_store.py          # ChromaDB wrapper

ui/                        # 🆕 Streamlit UI
├── app.py                       # Main application
├── pages/
│   ├── 1_Upload_Papers.py
│   ├── 2_Run_Pipeline.py
│   ├── 3_View_Results.py
│   └── 4_Compare_Pipelines.py
└── components/
    ├── graph_viewer.py          # Knowledge graph visualization
    └── metrics_dashboard.py     # Metrics display

ground_truth/              # 🆕 Эталонные данные
├── papers/                      # Размеченные статьи
│   ├── paper_001.json
│   ├── paper_002.json
│   └── ...
└── annotations.json             # Ground truth разметка
```

---

## 🔧 Детальная спецификация модулей

### 1. Base Pipeline Interface

**Файл:** `src/pipelines/base_pipeline.py`

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ExtractionResult:
    """Результат работы пайплайна"""
    paper_id: str
    entities: Dict[str, List[Any]]  # {"facts": [...], "hypotheses": [...], ...}
    relationships: List[Dict]        # [{"source": ..., "target": ..., "type": ...}]
    metadata: Dict[str, Any]         # Время, стоимость, токены
    timestamp: datetime

@dataclass
class PipelineMetrics:
    """Метрики производительности"""
    processing_time: float
    tokens_used: int
    cost_usd: float
    entities_extracted: int
    memory_used_mb: float


class BasePipeline(ABC):
    """Абстрактный класс для всех пайплайнов"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = self.__class__.__name__
        self.last_metrics = None

    @abstractmethod
    def extract(self, paper_text: str, paper_id: str) -> ExtractionResult:
        """
        Извлечение сущностей из статьи

        Args:
            paper_text: Полный текст статьи
            paper_id: Уникальный идентификатор статьи

        Returns:
            ExtractionResult с извлеченными сущностями
        """
        pass

    @abstractmethod
    def get_metrics(self) -> PipelineMetrics:
        """Получить метрики последнего прогона"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Описание пайплайна для UI"""
        pass

    @property
    @abstractmethod
    def estimated_cost_per_paper(self) -> float:
        """Ориентировочная стоимость на статью в USD"""
        pass

    @property
    def version(self) -> str:
        """Версия пайплайна"""
        return "1.0.0"
```

---

### 2. LLM Pipeline (Ground Truth Generation)

**Файл:** `src/pipelines/llm_pipeline.py`

**Цель:** Создание высококачественного ground truth датасета

**Характеристики:**
- Модель: GPT-4 или GPT-4o (высокая точность)
- Стоимость: ~$0.30 на статью
- Использование: Разовая разметка 10-15 статей
- Precision: ~95% (ожидаемое)

**Ключевые особенности:**
```python
class LLMPipeline(BasePipeline):
    def __init__(self, config):
        super().__init__(config)
        self.llm = get_llm_adapter("openai")
        self.model = "gpt-4o"  # Более точная модель
        self.temperature = 0.1  # Низкая температура для детерминизма

    def extract(self, paper_text: str, paper_id: str):
        # Структурированный промпт с примерами
        prompt = self._build_extraction_prompt(paper_text)

        # JSON-режим для структурированного вывода
        response = self.llm.generate(
            prompt=prompt,
            response_format={"type": "json_object"}
        )

        # Валидация и парсинг
        entities = self._parse_and_validate(response)

        return ExtractionResult(...)
```

**Промпт-стратегия:**
- Few-shot examples (3-5 примеров)
- Структурированный JSON output
- Явное определение каждого типа сущности
- Примеры связей между сущностями

---

### 3. Regex Pipeline (Cost-Free Baseline)

**Файл:** `src/pipelines/regex_pipeline.py`

**Цель:** Максимально дешевое извлечение простых паттернов

**Характеристики:**
- Стоимость: $0 (только CPU)
- Скорость: ~200-300 статей/час
- Precision: ~60-70% (ожидаемое)
- Recall: ~50-60% (ожидаемое)

**Извлекаемые паттерны:**
```python
# Примеры regex паттернов
PATTERNS = {
    "hypothesis": [
        r"we hypothesize(?:d)? that (.+?)[\.\;]",
        r"our hypothesis (?:is|was) (.+?)[\.\;]",
        r"we propose(?:d)? that (.+?)[\.\;]"
    ],
    "methods": [
        r"we (?:used|employed|utilized) (.+?) to",
        r"performed using (.+?)[\.\;]",
        r"measured (?:by|with|using) (.+?)[\.\;]"
    ],
    "results": [
        r"(?:showed|demonstrated|found) that (.+?)[\.\;]",
        r"p\s*[<=]\s*0\.0\d+",  # p-values
        r"\d+\.?\d*\s*±\s*\d+\.?\d*"  # measurements
    ]
}
```

**Детекция секций:**
```python
SECTION_HEADERS = {
    "abstract": r"^abstract$",
    "introduction": r"^introduction$",
    "methods": r"^(?:methods|materials and methods|methodology)$",
    "results": r"^results$",
    "discussion": r"^discussion$",
    "conclusion": r"^(?:conclusion|conclusions)$"
}
```

---

### 4. Hybrid Pipeline (Production Target)

**Файл:** `src/pipelines/hybrid_pipeline.py`

**Цель:** Оптимальный баланс стоимости и качества

**Характеристики:**
- Стоимость: ~$0.01-0.02 на статью
- Precision: ~85% (целевое)
- Recall: ~80% (целевое)
- F1-score: ~82% (целевое)

**Стратегия:**
```
┌──────────────────┐
│  Input Paper     │
└────────┬─────────┘
         │
    ┌────▼─────┐
    │  Regex   │ ──► Simple patterns (Methods, Results) [FREE]
    │ Extractor│
    └────┬─────┘
         │
    ┌────▼─────┐
    │   NLP    │ ──► Entities, dependencies (Facts) [CHEAP]
    │ Extractor│     spaCy: ~$0.001/paper
    └────┬─────┘
         │
    ┌────▼─────┐
    │   LLM    │ ──► Complex reasoning (Hypotheses, Conclusions) [SELECTIVE]
    │(selective)│     gpt-5-mini: Only 20% of content
    └────┬─────┘     ~$0.01/paper
         │
    ┌────▼─────┐
    │  Output  │
    └──────────┘
```

**Алгоритм принятия решений:**
```python
def extract_entity(self, text: str, entity_type: str):
    # Шаг 1: Попробовать regex
    regex_result, confidence = self.regex_extractor.extract(text, entity_type)

    if confidence > 0.8:
        return regex_result  # Высокая уверенность → используем regex

    # Шаг 2: Попробовать NLP
    if entity_type in ["facts", "techniques"]:
        nlp_result = self.nlp_extractor.extract(text, entity_type)
        return nlp_result

    # Шаг 3: LLM только для сложных случаев
    if entity_type in ["hypotheses", "conclusions", "analysis"]:
        llm_result = self.llm_extractor.extract(text, entity_type)
        return llm_result
```

**Оптимизация LLM вызовов:**
- Batch processing (5-10 запросов → 1 API call)
- Кэширование похожих запросов
- Использование gpt-5-mini вместо GPT-4

---

### 5. Data Models

**Файл:** `src/models/entities.py`

```python
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class EntityType(Enum):
    """Типы извлекаемых сущностей"""
    FACT = "fact"
    HYPOTHESIS = "hypothesis"
    EXPERIMENT = "experiment"
    TECHNIQUE = "technique"
    RESULT = "result"
    DATASET = "dataset"
    ANALYSIS = "analysis"
    CONCLUSION = "conclusion"

@dataclass
class Entity:
    """Базовая сущность"""
    id: str
    type: EntityType
    text: str
    confidence: float  # 0.0 - 1.0
    source_section: str  # Abstract, Methods, Results, etc.
    metadata: dict

@dataclass
class Relationship:
    """Связь между сущностями"""
    source_id: str
    target_id: str
    relationship_type: str  # "tested_by", "based_on", "uses", etc.
    confidence: float

@dataclass
class KnowledgeGraph:
    """Knowledge graph для статьи"""
    paper_id: str
    entities: List[Entity]
    relationships: List[Relationship]

    def to_networkx(self):
        """Конвертация в NetworkX граф"""
        import networkx as nx
        G = nx.DiGraph()

        for entity in self.entities:
            G.add_node(entity.id, **entity.__dict__)

        for rel in self.relationships:
            G.add_edge(rel.source_id, rel.target_id,
                      type=rel.relationship_type,
                      confidence=rel.confidence)

        return G
```

---

### 6. Validation & Metrics

**Файл:** `src/validation/metrics.py`

```python
from dataclasses import dataclass
from typing import List, Dict
from src.models.entities import Entity

@dataclass
class ValidationMetrics:
    """Метрики валидации пайплайна"""
    precision: float
    recall: float
    f1_score: float

    # По типам сущностей
    per_entity_metrics: Dict[str, Dict[str, float]]

    # Aggregate
    total_tp: int  # True Positives
    total_fp: int  # False Positives
    total_fn: int  # False Negatives

class MetricsCalculator:
    """Расчет метрик качества"""

    def compute_metrics(
        self,
        predicted: List[Entity],
        ground_truth: List[Entity]
    ) -> ValidationMetrics:
        """
        Сравнение предсказаний с ground truth

        Args:
            predicted: Сущности, извлеченные пайплайном
            ground_truth: Эталонные сущности

        Returns:
            ValidationMetrics
        """
        # Matching logic (с учетом частичного совпадения текста)
        tp, fp, fn = self._calculate_matches(predicted, ground_truth)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Per-entity breakdown
        per_entity = self._calculate_per_entity_metrics(predicted, ground_truth)

        return ValidationMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1,
            per_entity_metrics=per_entity,
            total_tp=tp,
            total_fp=fp,
            total_fn=fn
        )

    def _calculate_matches(self, predicted, ground_truth):
        """Fuzzy matching с порогом similarity"""
        from difflib import SequenceMatcher

        tp = fp = fn = 0
        matched_gt = set()

        for pred in predicted:
            best_match = None
            best_score = 0

            for idx, gt in enumerate(ground_truth):
                if idx in matched_gt:
                    continue

                if pred.type != gt.type:
                    continue

                # Text similarity
                similarity = SequenceMatcher(None, pred.text, gt.text).ratio()

                if similarity > best_score:
                    best_score = similarity
                    best_match = idx

            if best_score > 0.8:  # Threshold для совпадения
                tp += 1
                matched_gt.add(best_match)
            else:
                fp += 1

        fn = len(ground_truth) - len(matched_gt)

        return tp, fp, fn
```

---

### 7. Storage Layer

**Файл:** `src/storage/results_db.py`

```python
import sqlite3
import json
from typing import List, Optional
from src.models.results import ExtractionResult

class ResultsDatabase:
    """SQLite database для хранения результатов"""

    def __init__(self, db_path: str = "data/results.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Создание таблиц"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS extraction_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id TEXT NOT NULL,
                pipeline_name TEXT NOT NULL,
                entities TEXT,  -- JSON
                relationships TEXT,  -- JSON
                metadata TEXT,  -- JSON
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(paper_id, pipeline_name)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ground_truth (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id TEXT UNIQUE NOT NULL,
                entities TEXT,  -- JSON
                relationships TEXT,  -- JSON
                annotator TEXT,
                annotation_date DATETIME
            )
        """)

        conn.commit()
        conn.close()

    def save_result(self, result: ExtractionResult, pipeline_name: str):
        """Сохранить результат извлечения"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO extraction_results
            (paper_id, pipeline_name, entities, relationships, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, (
            result.paper_id,
            pipeline_name,
            json.dumps(result.entities),
            json.dumps(result.relationships),
            json.dumps(result.metadata)
        ))

        conn.commit()
        conn.close()

    def get_result(self, paper_id: str, pipeline_name: str) -> Optional[ExtractionResult]:
        """Получить результат для статьи и пайплайна"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT entities, relationships, metadata
            FROM extraction_results
            WHERE paper_id = ? AND pipeline_name = ?
        """, (paper_id, pipeline_name))

        row = cursor.fetchone()
        conn.close()

        if row:
            return ExtractionResult(
                paper_id=paper_id,
                entities=json.loads(row[0]),
                relationships=json.loads(row[1]),
                metadata=json.loads(row[2])
            )
        return None
```

---

### 8. Streamlit UI

**Файл:** `ui/app.py`

```python
import streamlit as st
from src.pipelines.orchestrator import PipelineOrchestrator
from src.parsers import get_parser
from ui.components.graph_viewer import display_knowledge_graph
from ui.components.metrics_dashboard import display_metrics

st.set_page_config(
    page_title="Singularis Knowledge Extractor",
    page_icon="🧬",
    layout="wide"
)

# Sidebar
st.sidebar.title("⚙️ Configuration")
pipeline_choice = st.sidebar.selectbox(
    "Select Pipeline",
    ["LLM (Ground Truth)", "Regex-based", "Hybrid (Recommended)"],
    index=2
)

# Cost indicator
costs = {
    "LLM (Ground Truth)": "$0.30",
    "Regex-based": "$0.00",
    "Hybrid (Recommended)": "$0.02"
}
st.sidebar.metric("Estimated Cost/Paper", costs[pipeline_choice])

# Main area
st.title("🧬 Singularis Knowledge Extractor")
st.markdown("Extract structured knowledge from scientific papers")

# File upload
uploaded_file = st.file_uploader(
    "Upload Scientific Paper",
    type=["pdf", "txt", "html"],
    help="Supported formats: PDF, TXT, HTML"
)

if uploaded_file:
    # Parse document
    parser = get_parser(uploaded_file.type)
    paper_text = parser.parse(uploaded_file)

    # Display paper info
    st.info(f"📄 Paper loaded: {len(paper_text)} characters")

    # Extract button
    if st.button("🚀 Extract Knowledge", type="primary"):
        with st.spinner("Processing paper..."):
            orchestrator = PipelineOrchestrator()
            result = orchestrator.run_pipeline(
                pipeline_choice,
                paper_text,
                uploaded_file.name
            )

        # Display results in tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Entities",
            "🔗 Knowledge Graph",
            "📈 Metrics",
            "🔬 Raw JSON"
        ])

        with tab1:
            st.subheader("Extracted Entities")
            for entity_type, entities in result.entities.items():
                with st.expander(f"{entity_type.title()} ({len(entities)})"):
                    for entity in entities:
                        st.markdown(f"- {entity['text']}")

        with tab2:
            st.subheader("Knowledge Graph")
            display_knowledge_graph(result.relationships, result.entities)

        with tab3:
            st.subheader("Pipeline Metrics")
            display_metrics(result.metadata)

        with tab4:
            st.json(result.entities)
```

---

## 📅 Implementation Timeline

### Week 1: Infrastructure & Ground Truth (7-13 октября)

**Приоритет:** Создание foundation + LLM pipeline

**Задачи:**
1. ✅ **Модели данных** (`src/models/`)
   - `entities.py` - Entity, Relationship, KnowledgeGraph
   - `results.py` - ExtractionResult, PipelineMetrics

2. ✅ **Base Pipeline** (`src/pipelines/base_pipeline.py`)
   - Абстрактный класс BasePipeline
   - Интерфейсы для всех пайплайнов

3. ✅ **LLM Pipeline** (`src/pipelines/llm_pipeline.py`)
   - Интеграция с существующими LLM адаптерами
   - Промпт-инжиниринг для извлечения
   - JSON-режим для структурированного вывода

4. ✅ **Парсеры** (`src/parsers/`)
   - `pdf_parser.py` (PyMuPDF + pdfplumber)
   - `txt_parser.py`
   - `factory.py`

5. ✅ **Storage** (`src/storage/`)
   - `results_db.py` - SQLite для результатов
   - `ground_truth.py` - Ground truth database

6. ✅ **Создание Ground Truth**
   - Обработать 10-15 статей через LLM Pipeline
   - Ручная валидация результатов (с экспертом)
   - Сохранение в `ground_truth/`

**Deliverable:** Ground truth датасет (10-15 размеченных статей)

---

### Week 2: Cost-Efficient Pipelines (14-20 октября)

**Приоритет:** Regex + Hybrid pipelines + валидация

**Задачи:**
1. ✅ **Extractors** (`src/extractors/`)
   - `regex_extractor.py` - Паттерны для 8 типов сущностей
   - `nlp_extractor.py` - spaCy/scispaCy интеграция
   - `section_detector.py` - Детекция секций статьи

2. ✅ **Regex Pipeline** (`src/pipelines/regex_pipeline.py`)
   - Реализация чисто паттерн-based извлечения
   - Тестирование на ground truth

3. ✅ **Hybrid Pipeline** (`src/pipelines/hybrid_pipeline.py`)
   - Комбинация Regex + NLP + selective LLM
   - Оптимизация стоимости (batch processing, caching)

4. ✅ **Validation** (`src/validation/`)
   - `metrics.py` - Расчет Precision, Recall, F1
   - `comparator.py` - Сравнение пайплайнов

5. ✅ **Pipeline Orchestrator** (`src/pipelines/orchestrator.py`)
   - Управление пайплайнами
   - Сбор метрик

6. ✅ **Тестирование на ground truth**
   - Прогнать все 3 пайплайна на 10-15 статьях
   - Собрать метрики качества и стоимости

**Deliverable:** Рабочие Regex и Hybrid pipelines с метриками

---

### Week 3: UI, Optimization & Deployment (21-22 октября)

**Приоритет:** Демонстрация + финализация

**Задачи:**
1. ✅ **Streamlit UI** (`ui/`)
   - `app.py` - Main application
   - `components/graph_viewer.py` - Визуализация графа
   - `components/metrics_dashboard.py` - Дашборд метрик

2. ✅ **Оптимизация**
   - Тюнинг regex паттернов на основе ошибок
   - Оптимизация промптов для LLM части
   - Batch processing для API вызовов

3. ✅ **Документация**
   - README с инструкциями
   - API documentation
   - Performance metrics report

4. ✅ **Deployment**
   - Деплой на Streamlit Cloud / Hugging Face Spaces
   - Публичный URL для жюри

5. ✅ **Видео-демо** (3-5 минут)
   - Загрузка статьи
   - Выбор пайплайна
   - Визуализация результатов
   - Сравнение метрик

**Deliverable:** Production-ready приложение + видео-демо

---

## 🎯 Success Metrics

### Технические метрики
- ✅ **Precision:** ≥ 85%
- ✅ **Recall:** ≥ 80%
- ✅ **F1-score:** ≥ 82%
- ✅ **Cost:** < $0.05 на статью (целевое: $0.02)
- ✅ **Speed:** > 100 статей/час

### Критерии жюри
- ✅ **Полнота и Точность (25%):** Метрики выше
- ✅ **Робастность (25%):** Работа с PDF/TXT/HTML
- ✅ **Стоимость (25%):** $0.02/статья vs $0.30 baseline
- ✅ **Производительность (25%):** 100+ статей/час

### Bonus Points
- ✅ **Алгоритмический подход:** Hybrid pipeline показывает снижение стоимости в 15x при сохранении качества
- ✅ **Инновации:** Модульная архитектура, визуализация графов

---

## 🔧 Technical Stack (Updated)

### Core (Existing)
- Python 3.10+
- OpenAI API (GPT-4, gpt-5-mini)
- Ollama (опционально, для локальных экспериментов)

### New Dependencies
```txt
# Parsing
PyMuPDF>=1.23.0        # PDF extraction
pdfplumber>=0.10.0     # Tables from PDF
beautifulsoup4>=4.12.0 # HTML parsing

# NLP
spacy>=3.7.0
scispacy>=0.5.0        # Scientific NLP
en-core-sci-sm         # SciBERT model

# Vector DB
chromadb>=0.4.0

# Storage
sqlite3 (built-in)

# UI
streamlit>=1.28.0
plotly>=5.17.0         # Interactive graphs
networkx>=3.2.0        # Graph processing

# Utils
python-dotenv>=1.0.0
pyyaml>=6.0.1
```

---

## 💡 Key Optimizations

### 1. LLM Cost Reduction
```python
# Плохо: Каждую сущность отдельно
for section in sections:
    llm.generate(f"Extract hypothesis from: {section}")

# Хорошо: Batch processing
combined_prompt = "\n---\n".join([
    f"Section {i}: {section}"
    for i, section in enumerate(sections)
])
result = llm.generate(f"Extract hypotheses from all sections:\n{combined_prompt}")
```

### 2. Regex Patterns Optimization
```python
# Используем compiled regex для скорости
import re

class RegexExtractor:
    def __init__(self):
        # Compile patterns once
        self.hypothesis_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in HYPOTHESIS_PATTERNS
        ]

    def extract(self, text):
        # Reuse compiled patterns
        for pattern in self.hypothesis_patterns:
            matches = pattern.findall(text)
            # ...
```

### 3. Caching
```python
from functools import lru_cache

class HybridPipeline:
    @lru_cache(maxsize=100)
    def _extract_with_llm(self, text_hash: str, entity_type: str):
        """Cache LLM results for identical texts"""
        # LLM call here
```

---

## 📊 Expected Results

### Pipeline Comparison (Projected)

| Pipeline | Cost/Paper | Precision | Recall | F1 | Speed |
|----------|-----------|-----------|--------|-------|-------|
| LLM (GPT-4) | $0.30 | 95% | 92% | 93.5% | 30/hour |
| Regex | $0.00 | 65% | 55% | 59.6% | 300/hour |
| **Hybrid** | **$0.02** | **85%** | **80%** | **82.4%** | **120/hour** |

### Bonus Points Justification
- **15x cost reduction** (from $0.30 to $0.02)
- **Quality retention** (93.5% F1 → 82.4% F1, only 11% drop)
- **Algorithmic innovation** (модульная архитектура)

---

## 🚀 Next Steps

1. ✅ **Approve this plan** ← You are here
2. 📝 **Week 1:** Implement infrastructure + LLM pipeline
3. 📝 **Week 2:** Implement Regex + Hybrid pipelines
4. 📝 **Week 3:** UI + deployment + video
5. 🎉 **Submit before October 22, 11:59 PM PT**

---

**Last Updated:** October 10, 2025
**Status:** ✅ Ready for Implementation
**Estimated Total Dev Time:** 60-80 hours over 3 weeks
