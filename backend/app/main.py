from pathlib import Path
import shutil

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.config import settings
from app.document_loader import extract_text_from_pdf

app = FastAPI(
    title="RAG MVP Backend",
    version="0.1.0",
    description="Backend service for PDF upload, retrieval, and question answering."
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


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


@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is missing.")

    file_path = UPLOAD_DIR / file.filename

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "message": "File uploaded successfully",
        "filename": file.filename,
        "saved_path": str(file_path),
    }


@app.get("/extract-text/{filename}")
def extract_text(filename: str):
    file_path = UPLOAD_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found.")

    try:
        result = extract_text_from_pdf(file_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract text: {str(e)}")