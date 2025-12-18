# Financial RAG Evaluation with Promptfoo

A production-grade demo for evaluating LLM and RAG systems on financial quarterly earnings reports using [promptfoo](https://www.promptfoo.dev/).

## Overview

This project demonstrates comprehensive LLM evaluation capabilities:

- **Model Comparison**: Compare costs, latency, and quality across OpenAI, Anthropic, and Google models
- **RAG Retriever Evaluation**: Compare ChromaDB vs Qdrant retrieval quality
- **RAG Context Quality**: Evaluate retrieval relevance and recall
- **RAG Generation Quality**: Test response accuracy with different LLM providers
- **Prompt Engineering**: Compare different prompt templates
- **Red Team Testing**: Security testing including prompt injection, hallucination detection, and policy compliance

## Data

The project uses quarterly earnings reports (10-Q filings) from major technology companies:
- Apple (AAPL)
- Microsoft (MSFT)
- NVIDIA (NVDA)
- Intel (INTC)

PDF documents are processed using [Docling](https://github.com/DS4SD/docling) for accurate table extraction (97.9% accuracy).

## Project Structure

```
promptfoo-llm-quality-gate/
├── config/
│   ├── app_config.yaml              # Application settings
│   └── promptfoo/
│       ├── providers.yaml           # Provider definitions
│       ├── defaults.yaml            # Default settings
│       └── evaluations/             # Evaluation configs
│           ├── 01_model_comparison.yaml
│           ├── 02_rag_retriever.yaml
│           ├── 03_rag_context.yaml
│           ├── 04_rag_generation.yaml
│           ├── 05_prompt_evaluation.yaml
│           └── 06_redteam.yaml
├── data/
│   ├── documents/                   # PDF earnings reports
│   ├── gold_qn/                     # Ground truth Q&A
│   └── processed/                   # Docling cache
├── prompts/
│   ├── system/                      # System prompts
│   └── rag/                         # RAG prompts
├── scripts/
│   ├── ingest_documents.py          # Document ingestion
│   └── run_evaluations.sh           # Run evaluations
├── src/financial_rag/
│   ├── config.py                    # Configuration management
│   ├── processing/                  # Document processing
│   ├── providers/                   # Promptfoo providers
│   ├── vectorstores/                # Vector DB implementations
│   ├── embeddings/                  # Embedding utilities
│   └── utils/                       # Test loader, etc.
└── results/                         # Evaluation results
```

## Setup

### Prerequisites

- Python 3.12+
- Node.js (for promptfoo CLI)
- Docker (for Qdrant vector database)

### Step 1: Install Dependencies

```bash
# Install promptfoo CLI
npm install -g promptfoo

# Install Python dependencies (using uv - recommended)
uv sync

# Or using pip
pip install -e .
```

### Step 2: Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```bash
# Required - for embeddings and LLM
OPENAI_API_KEY=sk-your-openai-key-here

# Optional - for model comparison
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
```

### Step 3: Set Up Vector Databases

This project uses two vector databases for comparison. Both run locally at no cost.

#### ChromaDB (No Setup Required)

ChromaDB runs embedded in Python with local file persistence. It's automatically configured:

```
CHROMA_PERSIST_DIRECTORY=./data/chroma_db
CHROMA_COLLECTION_NAME=financial_documents
```

No additional setup needed - it creates the database automatically during ingestion.

#### Qdrant (Requires Docker)

Qdrant runs as a Docker container with persistent storage.

**Start Qdrant:**

```bash
# Create data directory and start Qdrant with persistent storage
mkdir -p data/qdrant
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v $(pwd)/data/qdrant:/qdrant/storage \
  qdrant/qdrant
```

**Verify Qdrant is running:**

```bash
# Check container status
docker ps --filter name=qdrant

# Check health endpoint
curl http://localhost:6333/healthz
# Expected output: healthz check passed
```

**Manage Qdrant container:**

```bash
# Stop Qdrant
docker stop qdrant

# Start Qdrant (after stopping)
docker start qdrant

# Remove Qdrant container (data persists in data/qdrant/)
docker rm qdrant

# View Qdrant logs
docker logs qdrant
```

**Qdrant Web UI:** Open http://localhost:6333/dashboard in your browser to explore collections.

### Step 4: Document Ingestion

Process PDF documents and populate vector stores:

```bash
python scripts/ingest_documents.py
```

This will:
- Parse PDFs using Docling with table-aware chunking
- Populate ChromaDB and Qdrant with document chunks
- Cache processed documents for faster re-runs

## Running Evaluations

### Using the script

```bash
# Show help
./scripts/run_evaluations.sh

# Run specific evaluation
./scripts/run_evaluations.sh 1  # Model comparison
./scripts/run_evaluations.sh 2  # RAG retriever
./scripts/run_evaluations.sh 3  # RAG context
./scripts/run_evaluations.sh 4  # RAG generation
./scripts/run_evaluations.sh 5  # Prompt evaluation
./scripts/run_evaluations.sh 6  # Red team

# Run all evaluations
./scripts/run_evaluations.sh all
```

### Using promptfoo directly

```bash
# Run specific evaluation
promptfoo eval -c config/promptfoo/evaluations/01_model_comparison.yaml

# Run red team
promptfoo redteam run -c config/promptfoo/evaluations/06_redteam.yaml

# View results
promptfoo view
```

## Configuration

### Application Settings (`config/app_config.yaml`)

```yaml
documents:
  selected:
    - "2023 Q3 AAPL.pdf"
    - "2023 Q3 MSFT.pdf"
    # ...

test_data:
  question_types:
    - "Single-Doc Single-Chunk RAG"
  chunk_types:
    - "Table"
    - "Text"

rag:
  top_k: 5
  chunk_size: 1000
  chunk_overlap: 200
```

### Environment Variables (`.env`)

- `OPENAI_API_KEY`: OpenAI API key
- `ANTHROPIC_API_KEY`: Anthropic API key
- `GOOGLE_CLOUD_PROJECT`: GCP project for Vertex AI
- `CHROMA_PERSIST_DIRECTORY`: ChromaDB storage path
- `QDRANT_HOST`: Qdrant server host

## Custom Providers

The project includes custom promptfoo providers:

- **chromadb_rag.py**: Full RAG pipeline with ChromaDB
- **qdrant_rag.py**: Full RAG pipeline with Qdrant
- **retriever_only.py**: Retrieval evaluation only
- **generation_only.py**: Generation evaluation with pre-provided context

## Evaluation Metrics

### Standard Metrics
- **Factuality**: Compare response accuracy against ground truth
- **Context Relevance**: Evaluate retrieved document relevance
- **Context Recall**: Measure information retrieval completeness
- **Cost**: Track API costs per evaluation
- **Latency**: Measure response times

### Red Team Tests
- Prompt injection detection
- Hallucination testing (critical for financial data)
- Policy compliance (no investment advice)
- PII leakage prevention
- Jailbreak attempts

## Troubleshooting

### Qdrant Connection Issues

```bash
# Check if Qdrant is running
docker ps --filter name=qdrant

# If not running, start it
docker start qdrant

# If container doesn't exist, create it
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v $(pwd)/data/qdrant:/qdrant/storage qdrant/qdrant
```

### ChromaDB Issues

ChromaDB stores data in `./data/chroma_db/`. To reset:

```bash
rm -rf data/chroma_db
python scripts/ingest_documents.py
```

### API Key Issues

```bash
# Verify your API key is set
echo $OPENAI_API_KEY

# Or check .env file
cat .env | grep OPENAI_API_KEY
```

### Port Conflicts

If port 6333 is in use:

```bash
# Find what's using the port
lsof -i :6333

# Use a different port for Qdrant
docker run -d --name qdrant -p 6334:6333 qdrant/qdrant
# Update QDRANT_PORT=6334 in .env
```

## License

MIT
