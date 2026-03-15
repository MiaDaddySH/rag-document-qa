# RAG MVP (PDF QA Assistant)

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)
![Azure OpenAI](https://img.shields.io/badge/Azure-OpenAI-blue)
![Qdrant](https://img.shields.io/badge/Qdrant-VectorDB-orange)

This project is a PDF question-answering system built with **FastAPI + Azure OpenAI + Qdrant**.  
Typical flow: upload PDF → index document → ask questions → return grounded answers with sources and citations.

## Current Capabilities (V0.2 Level)
- PDF upload, extraction, chunking, embedding, retrieval, and QA
- Retrieval reranking (vector score + keyword overlap score)
- Confidence-based refusal strategy (avoid low-evidence over-answering)
- Sentence-level evidence citations (`citations`)
- Query embedding cache (lower latency and cost for repeated questions)
- Offline evaluation and parameter tuning (English + Chinese)

## Project Structure

```text
backend/
  app/
    main.py              # FastAPI entry and APIs
    config.py            # config and validation
    document_loader.py   # PDF text extraction
    chunker.py           # text chunking
    embedding.py         # embedding calls + cache
    vector_store.py      # Qdrant read/write
    rag_pipeline.py      # retrieve, rerank, refusal, evidence alignment
    evaluation.py        # offline evaluation
    tune_rag.py          # grid search tuning
    eval_cases.sample.json
    eval_cases.zh.json
  requirements.txt

web-admin/
  index.html
  style.css
  app.js
```

## Quick Start

### 1) Start Qdrant

```bash
docker run -d --name rag-qdrant -p 6333:6333 qdrant/qdrant
```

### 2) Start Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `backend/.env` (minimal example):

```bash
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com
AZURE_OPENAI_DEPLOYMENT=your_chat_deployment
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=your_embedding_deployment

QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=rag_documents
```

Run API:

```bash
uvicorn app.main:app --reload
```

### 3) Start Web Admin

```bash
cd web-admin
python3 -m http.server 5500
```

Open: `http://127.0.0.1:5500`

## Common APIs

### Health Check

```http
GET /health
GET /health?check_dependencies=true
```

### Upload PDF

```http
POST /upload
```

### Index Document

```http
POST /index/{filename}
```

### Ask Question

```http
POST /ask
Content-Type: application/json
```

Request example:

```json
{
  "question": "What implementation recommendations does the guide provide?",
  "top_k": 6,
  "filename": "ai-for-enterprise-rag-eguide.pdf"
}
```

Important response fields:
- `answer`
- `sources`
- `citations`
- `is_refused`
- `refusal_reason`
- `answer_confidence`

## Offline Evaluation

English evaluation:

```bash
cd backend
python -m app.evaluation --cases-file app/eval_cases.sample.json --min-pass-rate 0.6 --min-avg-keyword-hit 0.3
```

Chinese evaluation:

```bash
cd backend
python -m app.evaluation --cases-file app/eval_cases.zh.json --min-pass-rate 0.6 --min-avg-keyword-hit 0.3
```

## Parameter Tuning (Grid Search)

```bash
cd backend
python -m app.tune_rag --cases-file app/eval_cases.sample.json --top-n 5
python -m app.tune_rag --cases-file app/eval_cases.zh.json --top-n 5
```

## Key Configuration

RAG quality:
- `RAG_MIN_SCORE`
- `RAG_RETRIEVAL_MULTIPLIER`
- `RAG_RERANK_VECTOR_WEIGHT`
- `RAG_RERANK_KEYWORD_WEIGHT`
- `RAG_CONFIDENCE_THRESHOLD`
- `RAG_MIN_TOP_RERANK_SCORE`
- `RAG_MAX_CITATIONS`
- `RAG_CITATION_MIN_OVERLAP`

Performance:
- `RAG_QUERY_EMBEDDING_CACHE_ENABLED`
- `RAG_QUERY_EMBEDDING_CACHE_SIZE`

## Architecture

```mermaid
flowchart TD

User[User] --> WebUI[Web Admin UI]
WebUI --> FastAPI[FastAPI Backend]

FastAPI --> Upload[POST /upload]
FastAPI --> Index[POST /index/{filename}]
FastAPI --> Ask[POST /ask]

Upload --> LocalStorage[(Local PDF Storage)]

Index --> Extract[PDF Text Extraction]
Extract --> Chunk[Chunking]
Chunk --> Embed[Embedding Generation]
Embed --> AzureOpenAI[(Azure OpenAI Embedding)]
Embed --> Qdrant[(Qdrant Vector DB)]

Ask --> QueryEmbed[Question Embedding]
QueryEmbed --> AzureOpenAI
QueryEmbed --> Retrieve[Vector Retrieval]
Retrieve --> Qdrant
Retrieve --> Rerank[Rerank: vector + keyword]
Rerank --> Confidence[Confidence & Refusal Gate]
Confidence --> LLM[Answer Generation]
LLM --> AzureOpenAI
LLM --> Citations[Evidence Citation Alignment]
Citations --> Response[Answer + Sources + Citations + Refusal Signals]
Response --> WebUI
```

Simplified flow:

```text
PDF -> Text -> Chunk -> Embedding -> Qdrant
Question -> Query Embedding -> Retrieve -> Rerank -> Confidence Gate -> LLM -> Citations -> Response
```

## Known Limits
- Current default usage is single-document QA (via `filename` filter).
- Citation alignment uses lightweight token overlap and may be suboptimal on complex tables or cross-paragraph reasoning.
- For production, add auth, rate limiting, audit trails, and monitoring/alerting.

## Suggested Next Steps
- Query rewrite for multi-turn QA
- Hybrid retrieval (vector + keyword)
- Structure-aware chunking (titles/paragraphs/lists)
- Metrics dashboarding (latency, refusal rate, citation coverage)

## License

MIT
