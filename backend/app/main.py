import logging
import platform
from pathlib import Path
import random
import re
import shutil
import time
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.chunker import chunk_text
from app.config import get_settings_health_report, settings
from app.document_loader import extract_text_from_pdf
from app.embedding import embed_texts
from app.llm_client import probe_azure_openai_dependency
from pydantic import BaseModel
from app.rag_pipeline import answer_question
from app.vector_store import delete_chunks_by_filename, probe_qdrant_dependency, upsert_chunks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_mvp_backend")
SENSITIVE_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key\s*[:=]\s*)([^\s,;]+)"),
    re.compile(r"(?i)(authorization\s*[:=]\s*)([^\s,;]+)"),
    re.compile(r"(?i)(token\s*[:=]\s*)([^\s,;]+)"),
    re.compile(r"(?i)(bearer\s+)([a-z0-9\-._~+/]+=*)"),
]

app = FastAPI(
    title="RAG MVP Backend",
    version="0.1.0",
    description="Backend service for PDF upload, retrieval, and question answering."
)

# 添加 CORS 中间件，允许所有来源的请求（在生产环境中应更严格地配置）。
# 这使得前端应用可以从不同的域名访问这个后端 API。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


def error_payload(code: str, message: str, request_id: str | None) -> dict:
    return {
        "success": False,
        "error": {
            "code": code,
            "message": message,
            "request_id": request_id,
        },
    }


def raise_api_error(status_code: int, code: str, message: str) -> None:
    raise HTTPException(
        status_code=status_code,
        detail={
            "code": code,
            "message": message,
        },
    )


def status_code_to_error_code(status_code: int) -> str:
    mapping = {
        400: "bad_request",
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        409: "conflict",
        413: "payload_too_large",
        422: "validation_error",
        429: "rate_limited",
    }
    return mapping.get(status_code, "request_error")


def redact_text(value: str) -> str:
    text = value
    for pattern in SENSITIVE_PATTERNS:
        text = pattern.sub(lambda match: f"{match.group(1)}***", text)
    return text


def should_log_success(status_code: int, elapsed_ms: float) -> bool:
    if status_code >= 400:
        return True
    if elapsed_ms >= settings.log_slow_request_ms:
        return True
    return random.random() < settings.log_success_sample_rate


@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-Id") or str(uuid4())
    request.state.request_id = request_id
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.error(
            "request_failed request_id=%s method=%s path=%s duration_ms=%s error=%s",
            request_id,
            request.method,
            request.url.path,
            elapsed_ms,
            redact_text(str(exc)),
        )
        raise
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    response.headers["X-Request-Id"] = request_id
    response.headers["X-Process-Time-Ms"] = str(elapsed_ms)
    if should_log_success(response.status_code, elapsed_ms):
        logger.info(
            "request_ok request_id=%s method=%s path=%s status=%s duration_ms=%s sampled=%s",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            response.status_code < 400 and elapsed_ms < settings.log_slow_request_ms,
        )
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    request_id = getattr(request.state, "request_id", None)
    detail = exc.detail
    if isinstance(detail, dict):
        code = str(detail.get("code") or status_code_to_error_code(exc.status_code))
        message = str(detail.get("message") or "Request failed.")
    else:
        code = status_code_to_error_code(exc.status_code)
        message = str(detail) if detail else "Request failed."
    return JSONResponse(
        status_code=exc.status_code,
        content=error_payload(code, message, request_id),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", None)
    logger.error(
        "unhandled_exception request_id=%s path=%s error=%s",
        request_id,
        request.url.path,
        redact_text(str(exc)),
    )
    return JSONResponse(
        status_code=500,
        content=error_payload("internal_error", "Internal server error.", request_id),
    )


def normalize_filename(filename: str) -> str:
    safe_name = Path(filename).name.strip()
    if not safe_name:
        raise_api_error(400, "invalid_filename", "Filename is invalid.")
    return safe_name

def ensure_upload_size(file: UploadFile) -> None:
    try:
        file.file.seek(0, 2)
        size = file.file.tell()
        file.file.seek(0)
    except Exception:
        raise_api_error(400, "file_size_read_failed", "Failed to read file size.")

    max_bytes = settings.upload_max_mb * 1024 * 1024
    if size > max_bytes:
        raise_api_error(413, "file_too_large", "File is too large.")

def batch_embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    batch_size = settings.embedding_batch_size if settings.embedding_batch_size > 0 else len(texts)
    embeddings: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        embeddings.extend(embed_texts(texts[start : start + batch_size]))
    return embeddings

@app.get("/")
def read_root():
    return {
        "message": "RAG MVP backend is running",
        "service": "backend",
        "version": "0.1.0",
    }

@app.get("/health")
def health_check(check_dependencies: bool = Query(False)):
    config_report = get_settings_health_report()
    dependency_report = {
        "checked": check_dependencies,
        "ready": True,
        "azure_openai": {"ready": None, "detail": "skipped"},
        "qdrant": {"ready": None, "detail": "skipped"},
    }
    if check_dependencies:
        azure_ready, azure_detail = probe_azure_openai_dependency()
        qdrant_ready, qdrant_detail = probe_qdrant_dependency()
        dependency_report = {
            "checked": True,
            "ready": azure_ready and qdrant_ready,
            "azure_openai": {"ready": azure_ready, "detail": redact_text(azure_detail)},
            "qdrant": {"ready": qdrant_ready, "detail": redact_text(qdrant_detail)},
        }
    return {
        "status": "ok",
        "service_ready": config_report["config_validated"] and dependency_report["ready"],
        "runtime": {
            "service": "backend",
            "version": app.version,
            "python_version": platform.python_version(),
        },
        "config": config_report,
        "dependencies": dependency_report,
    }

@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise_api_error(400, "invalid_file_type", "Only PDF files are allowed.")

    if not file.filename:
        raise_api_error(400, "missing_filename", "Filename is missing.")

    ensure_upload_size(file)
    safe_filename = normalize_filename(file.filename)
    if not safe_filename.lower().endswith(".pdf"):
        raise_api_error(400, "invalid_file_extension", "Only PDF files are allowed.")

    file_path = UPLOAD_DIR / safe_filename

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "message": "File uploaded successfully",
        "filename": safe_filename,
        "saved_path": str(file_path),
    }

@app.get("/extract-text/{filename}")
def extract_text(filename: str):
    safe_filename = normalize_filename(filename)
    file_path = UPLOAD_DIR / safe_filename

    if not file_path.exists():
        raise_api_error(404, "file_not_found", "File not found.")

    try:
        result = extract_text_from_pdf(file_path)
        return result
    except Exception:
        raise_api_error(500, "extract_text_failed", "Failed to extract text.")


@app.get("/chunk/{filename}")
def chunk_document(
    filename: str,
    chunk_size: int = Query(500, gt=0),
    chunk_overlap: int = Query(100, ge=0),
):
    safe_filename = normalize_filename(filename)
    file_path = UPLOAD_DIR / safe_filename

    if not file_path.exists():
        raise_api_error(404, "file_not_found", "File not found.")

    if chunk_overlap >= chunk_size:
        raise_api_error(400, "invalid_chunk_overlap", "chunk_overlap must be smaller than chunk_size.")

    try:
        extraction_result = extract_text_from_pdf(file_path)
        text = extraction_result["text"]

        chunks = chunk_text(
            text=text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        return {
            "filename": safe_filename,
            "page_count": extraction_result["page_count"],
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "chunk_count": len(chunks),
            "chunks": chunks,
        }
    except Exception:
        raise_api_error(500, "chunk_document_failed", "Failed to chunk document.")
    

@app.get("/embed/{filename}")
def embed_document(
    filename: str,
    chunk_size: int = Query(500, gt=0),
    chunk_overlap: int = Query(100, ge=0),
):
    safe_filename = normalize_filename(filename)
    file_path = UPLOAD_DIR / safe_filename

    if not file_path.exists():
        raise_api_error(404, "file_not_found", "File not found.")

    if chunk_overlap >= chunk_size:
        raise_api_error(400, "invalid_chunk_overlap", "chunk_overlap must be smaller than chunk_size.")

    try:
        extraction_result = extract_text_from_pdf(file_path)
        text = extraction_result["text"]

        chunks = chunk_text(
            text=text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        contents = [chunk["content"] for chunk in chunks]
        vectors = batch_embed_texts(contents)
        embedded_chunks = []
        for chunk, vector in zip(chunks, vectors):
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
            "filename": safe_filename,
            "page_count": extraction_result["page_count"],
            "chunk_count": len(chunks),
            "embedded_chunk_count": len(embedded_chunks),
            "chunks": embedded_chunks,
        }
    except Exception:
        raise_api_error(500, "embed_document_failed", "Failed to generate embeddings.")
    
@app.post("/index/{filename}")
def index_document(
    filename: str,
    chunk_size: int = Query(500, gt=0),
    chunk_overlap: int = Query(100, ge=0),
):
    safe_filename = normalize_filename(filename)
    file_path = UPLOAD_DIR / safe_filename

    if not file_path.exists():
        raise_api_error(404, "file_not_found", "File not found.")

    if chunk_overlap >= chunk_size:
        raise_api_error(400, "invalid_chunk_overlap", "chunk_overlap must be smaller than chunk_size.")

    try:
        extraction_result = extract_text_from_pdf(file_path)
        text = extraction_result["text"]

        chunks = chunk_text(
            text=text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        contents = [chunk["content"] for chunk in chunks]
        vectors = batch_embed_texts(contents)
        chunks_with_embeddings = []
        for chunk, embedding in zip(chunks, vectors):
            chunks_with_embeddings.append(
                {
                    "filename": safe_filename,
                    "page_count": extraction_result["page_count"],
                    "chunk_index": chunk["chunk_index"],
                    "content": chunk["content"],
                    "start_char": chunk["start_char"],
                    "end_char": chunk["end_char"],
                    "embedding": embedding,
                }
            )

        delete_chunks_by_filename(safe_filename)
        inserted_count = upsert_chunks(chunks_with_embeddings)

        return {
            "message": "Document indexed successfully",
            "filename": safe_filename,
            "chunk_count": len(chunks),
            "inserted_count": inserted_count,
            "collection_name": settings.qdrant_collection_name,
        }
    except Exception:
        raise_api_error(500, "index_document_failed", "Failed to index document.")
    
class AskRequest(BaseModel):
    question: str
    top_k: int = 3
    filename: str | None = None

@app.post("/ask")
def ask_question(request: AskRequest):
    try:
        safe_filename = normalize_filename(request.filename) if request.filename else None
        result = answer_question(
            question=request.question,
            top_k=request.top_k,
            filename=safe_filename,
        )
        return result
    except Exception:
        raise_api_error(500, "ask_question_failed", "Failed to answer question.")
