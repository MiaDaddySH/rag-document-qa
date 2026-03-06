# RAG MVP

A monorepo project for building a document-based RAG application.

## Structure

- `backend/`: FastAPI backend for document ingestion and question answering
- `web-admin/`: lightweight web UI for PDF upload and admin testing
- `mobile-app/`: Flutter client for the Q&A experience

## Backend setup

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Backend Endpoints

- GET / -> Service info
- GET /health -> Health check
- GET /docs -> Swagger API documentation