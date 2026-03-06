from pathlib import Path
import shutil

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from app.chunker import chunk_text
from app.config import settings
from app.document_loader import extract_text_from_pdf
from app.embedding import embed_text

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


@app.get("/chunk/{filename}")
def chunk_document(
    filename: str,
    chunk_size: int = Query(500, gt=0),
    chunk_overlap: int = Query(100, ge=0),
):
    file_path = UPLOAD_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found.")

    if chunk_overlap >= chunk_size:
        raise HTTPException(
            status_code=400,
            detail="chunk_overlap must be smaller than chunk_size.",
        )

    try:
        extraction_result = extract_text_from_pdf(file_path)
        text = extraction_result["text"]

        chunks = chunk_text(
            text=text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        return {
            "filename": filename,
            "page_count": extraction_result["page_count"],
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "chunk_count": len(chunks),
            "chunks": chunks,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to chunk document: {str(e)}")
    
    
@app.get("/embed/{filename}")
def embed_document(
    filename: str,
    chunk_size: int = Query(500, gt=0),
    chunk_overlap: int = Query(100, ge=0),
):
    file_path = UPLOAD_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found.")

    if chunk_overlap >= chunk_size:
        raise HTTPException(
            status_code=400,
            detail="chunk_overlap must be smaller than chunk_size.",
        )

    try:
        extraction_result = extract_text_from_pdf(file_path)
        text = extraction_result["text"]

        chunks = chunk_text(
            text=text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        embedded_chunks = []
        for chunk in chunks:
            vector = embed_text(chunk["content"])
            embedded_chunks.append(
                {
                    "chunk_index": chunk["chunk_index"],
                    "content_preview": chunk["content"][:120],
                    "start_char": chunk["start_char"],
                    "end_char": chunk["end_char"],
                    "embedding_dimension": len(vector),
                }
            )

        return {
            "filename": filename,
            "page_count": extraction_result["page_count"],
            "chunk_count": len(chunks),
            "embedded_chunk_count": len(embedded_chunks),
            "chunks": embedded_chunks,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate embeddings: {str(e)}")