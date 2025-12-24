# Financial RAG Evaluation with Promptfoo

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![RAG](https://img.shields.io/badge/RAG-ChromaDB%20%7C%20Qdrant-green)](https://www.trychroma.com/)
[![LLM](https://img.shields.io/badge/LLM-OpenAI%20%7C%20Anthropic%20%7C%20Google-orange)](https://openai.com/)
[![Promptfoo](https://img.shields.io/badge/eval-promptfoo-5D3FD3)](https://www.promptfoo.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/achamorrofdz14/promptfoo-llm-quality-gate/actions/workflows/ci.yml/badge.svg)](https://github.com/achamorrofdz14/promptfoo-llm-quality-gate/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

> **Read the full analysis**: [5 Experiments Later: Is Promptfoo the LLM Quality Gate MLOps Has Been Missing?](BLOG_URL)

A comprehensive LLM evaluation framework for financial RAG systems using [Promptfoo](https://www.promptfoo.dev/). This project demonstrates systematic evaluation across 5 dimensions using real 10-Q SEC filings from AAPL, MSFT, NVDA, and INTC.

## Architecture

![Project Architecture](/assets/architecture.png)

## Evaluation Dimensions

| # | Evaluation | Purpose |
|---|------------|---------|
| 01 | Model Comparison | Compare cost, latency, and quality across OpenAI/Anthropic/Google |
| 02 | RAG Retriever | ChromaDB vs Qdrant retrieval quality and latency |
| 03 | RAG Context | Factuality verification and hallucination detection |
| 04 | Prompt Engineering | Template comparison across providers |
| 05 | Red Team | Security testing (prompt injection, PII, policy violations) |

## Tech Stack

| Component | Technology |
|-----------|------------|
| Evaluation | [Promptfoo](https://www.promptfoo.dev/) |
| Vector DBs | ChromaDB, Qdrant |
| LLM Providers | OpenAI, Anthropic, Google |
| PDF Processing | [Docling](https://github.com/DS4SD/docling) |
| Framework | LangChain |
| Package Manager | [uv](https://github.com/astral-sh/uv) |

## Setup

### Prerequisites

- Python 3.12+
- Node.js (for Promptfoo CLI)
- Docker (for Qdrant)

### 1. Install Dependencies

```bash
npm install -g promptfoo
uv sync
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```bash
OPENAI_API_KEY=sk-...          # Required
ANTHROPIC_API_KEY=sk-ant-...   # Optional - for model comparison
GOOGLE_CLOUD_PROJECT=...       # Optional - for Vertex AI
```

### 3. Set Up Vector Databases

**ChromaDB** - No setup needed, runs embedded with local persistence.

**Qdrant** - Requires Docker:

```bash
mkdir -p data/qdrant
docker run -d --name qdrant -p 6333:6333 -v $(pwd)/data/qdrant:/qdrant/storage qdrant/qdrant
```

### 4. Ingest Documents

```bash
python scripts/ingest_documents.py
```

## Running Evaluations

```bash
./scripts/run_evaluations.sh 1    # Model comparison
./scripts/run_evaluations.sh 2    # RAG retriever
./scripts/run_evaluations.sh 3    # RAG context
./scripts/run_evaluations.sh 4    # Prompt engineering
./scripts/run_evaluations.sh 5    # Red team security
./scripts/run_evaluations.sh all  # All evaluations

promptfoo view                    # View results in browser
```

Or run directly with Promptfoo:

```bash
promptfoo eval -c config/promptfoo/evaluations/01_model_comparison.yaml
promptfoo redteam run -c config/promptfoo/evaluations/05_redteam.yaml
```

## Project Structure

```
├── config/promptfoo/evaluations/   # 5 evaluation configs
├── src/financial_rag/
│   ├── providers/                  # Custom Promptfoo providers
│   ├── vectorstores/               # ChromaDB + Qdrant implementations
│   └── processing/                 # Docling PDF processing
├── data/documents/                 # 10-Q PDF reports
├── prompts/                        # System and RAG prompts
└── tests/                          # Pytest test suite
```

## Development

```bash
uv run pytest tests/ -v        # Run tests
uv run ruff check src/ tests/  # Lint
pre-commit install             # Setup hooks
```

## License

[MIT](LICENSE)
