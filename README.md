# SciBERT-Nebius Knowledge Graph Extractor

Cost-efficient scientific paper analysis using SciBERT embeddings and Nebius LLM.

## Features

- **FREE SciBERT embeddings** (domain-optimized for scientific papers, 768 dims)
- **Cost-efficient Nebius LLM** ($0.018/paper)
- **ChromaDB semantic search** (local, FREE)
- **GROBID ML parser** (structured IMRAD extraction)
- **Telegram Bot** for PDF → Knowledge Graph conversion

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Set up environment
cp .env.example .env
# Edit .env and add:
# TELEGRAM_BOT_TOKEN=...
# NEBIUS_API_KEY=...

# Test pipeline
python scripts/example_scibert_nebius_pipeline.py

# Run Telegram Bot
python scripts/run_scibert_telegram_bot.py
```

## Architecture

```
PDF → GROBID → SciBERT → Keywords → ChromaDB → Validation → Graph → SVG
       (ML)     (FREE)    (Nebius)   (LOCAL)    (Nebius)   (Heuristics)
```

## Cost

- Embeddings: $0.000 (SciBERT, local)
- Keywords: ~$0.003 (Nebius)
- Validation: ~$0.015 (Nebius)
- **Total: ~$0.018/paper**

## Metrics

- Precision: ≥88%
- Recall: ≥82%
- Processing: 60-90 seconds/paper

## Documentation

See [CLAUDE.md](CLAUDE.md) for detailed architecture and development guide.

## License

MIT
