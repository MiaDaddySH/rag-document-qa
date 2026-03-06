from fastapi import FastAPI
from app.config import settings

app = FastAPI(
    title="RAG MVP Backend",
    version="0.1.0",
    description="Backend service for PDF upload, retrieval, and question answering."
)


@app.get("/")
def read_root():
    return {
        "message": "RAG MVP backend is running",
        "service": "backend",
        "version": "0.1.0",
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "azure_endpoint_configured": bool(settings.azure_openai_endpoint),
    }