# Singularis Challenge - AAIAA Project

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
```

Edit `.env`:
```bash
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxx
```

### 3. Configure LLM Provider

Edit `src/config/llm_config.yaml` to choose your provider:

```yaml
# Switch between "openai" or "ollama"
active_provider: "openai"
```

### 4. Run Examples

```bash
# LLM adapters example
python examples/example_adapters.py

# Paper fetcher example
python examples/example_fetchers.py
```

### 5. Run Tests

```bash
# Run all tests
pytest

# Run only integration tests
pytest tests/integration/ -v

# Run with coverage report
pytest --cov=src --cov-report=html
```

---

## 🔌 LLM Adapters

### Switching Providers

**Option 1: In config file**
```yaml
# src/config/llm_config.yaml
active_provider: "ollama"  # Change here
```

**Option 2: In code**

```python
from src.llm_adapters import get_llm_adapter

# Use specific provider
llm = get_llm_adapter("openai")
# or
llm = get_llm_adapter("ollama")
```

### Usage Examples

**Text Generation:**

```python
from src.llm_adapters import get_llm_adapter

llm = get_llm_adapter()

result = llm.generate(
    prompt="Extract facts from this paper...",
    system_prompt="You are a scientific data extractor."
)

print(result["content"])
print(f"Cost: ${result['cost']:.6f}")
```

**Embeddings:**
```python
texts = ["text 1", "text 2", "text 3"]
embeddings = llm.embed(texts)
```

**Streaming:**
```python
for chunk in llm.stream_generate("Tell me about aging"):
    print(chunk, end="", flush=True)
```

---

## 🛠️ Supported Providers

### OpenAI (ChatGPT)
- **Model:** gpt-4o-mini
- **Embeddings:** text-embedding-3-small
- **Cost:** ~$0.15 input, ~$0.60 output per 1M tokens

### Ollama (Local)
- **Model:** llama3.1:8b (configurable)
- **Embeddings:** nomic-embed-text
- **Cost:** $0 (runs locally)

**Install Ollama models:**
```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

---

## 📚 Paper Fetchers

### Fetching Papers from PubMed

```python
from src.fetchers import get_fetcher

# Get PubMed fetcher
fetcher = get_fetcher("pubmed")

# Search for papers
pmids = fetcher.search("caloric restriction aging", max_results=10)

# Fetch paper metadata
paper = fetcher.fetch_paper(pmids[0])
print(f"Title: {paper.title}")
print(f"Authors: {', '.join(paper.authors)}")
print(f"Abstract: {paper.abstract}")

# Or search and fetch in one call
papers = fetcher.search_and_fetch("rapamycin longevity", max_results=5)
```

**Run example:**
```bash
python examples/example_fetchers.py
```

---

## 🧪 Testing

### Running Tests

```bash
# Install dependencies (including pytest)
pip install -r requirements.txt

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

### Test Structure

```
tests/
├── conftest.py              # Pytest fixtures
└── integration/
    └── test_pubmed_fetcher.py  # PubMed API tests
```

**Integration tests:**
- Test real API calls to PubMed
- Require internet connection
- Can use NCBI_API_KEY for faster rate limits (10 req/sec vs 3)

**Tips:**
- Integration tests are fast (~5-10 seconds)
- No API key needed (but recommended for speed)
- Tests use real PubMed data

---

## 📁 Project Structure

```
AAIAA/
├── src/
│   ├── config/
│   │   ├── llm_config.yaml       # LLM provider configuration
│   │   ├── fetcher_config.yaml   # Paper fetcher configuration
│   │   └── settings.py           # Config loader
│   ├── adapters/
│   │   ├── base_adapter.py       # Abstract LLM adapter
│   │   ├── openai_adapter.py     # OpenAI implementation
│   │   ├── ollama_adapter.py     # Ollama implementation
│   │   └── factory.py            # Adapter factory
│   └── fetchers/
│       ├── base_fetcher.py       # Abstract fetcher
│       ├── pubmed_fetcher.py     # PubMed E-utilities
│       └── factory.py            # Fetcher factory
├── tests/
│   ├── conftest.py               # Pytest fixtures
│   └── integration/
│       └── test_pubmed_fetcher.py # PubMed integration tests
├── examples/
│   ├── example_adapters.py       # LLM adapter examples
│   └── example_fetchers.py       # Paper fetcher examples
├── docs/
│   ├── Claude.md                 # Project context
│   └── singularis_project_doc.md # Full documentation
├── .env.example                  # Environment template
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

---

## 📝 Next Steps

1. ✅ LLM adapters created
2. ✅ Paper fetchers created (PubMed)
3. 🔄 Create document loader module
4. 🔄 Create text processor module
5. 🔄 Create RAG pipeline
6. 🔄 Create UI with Streamlit

---

## 📚 Documentation

See [`docs/singularis_project_doc.md`](docs/singularis_project_doc.md) for full project documentation.
