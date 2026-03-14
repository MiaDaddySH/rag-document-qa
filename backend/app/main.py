from pathlib import Path
import shutil

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.chunker import chunk_text
from app.config import get_settings_health_report, settings
from app.document_loader import extract_text_from_pdf
from app.embedding import embed_texts
from pydantic import BaseModel
from app.rag_pipeline import answer_question
from app.vector_store import delete_chunks_by_filename, upsert_chunks

# 主应用实例
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

# 统一处理上传文件名，避免路径穿越。
def normalize_filename(filename: str) -> str:
    safe_name = Path(filename).name.strip()
    if not safe_name:
        raise HTTPException(status_code=400, detail="Filename is invalid.")
    return safe_name

# 校验上传大小，避免过大文件占用资源。
def ensure_upload_size(file: UploadFile) -> None:
    try:
        file.file.seek(0, 2)
        size = file.file.tell()
        file.file.seek(0)
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to read file size.")

    max_bytes = settings.upload_max_mb * 1024 * 1024
    if size > max_bytes:
        raise HTTPException(status_code=400, detail="File is too large.")

# 批量生成 embeddings，降低调用开销。
def batch_embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    batch_size = settings.embedding_batch_size if settings.embedding_batch_size > 0 else len(texts)
    embeddings: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        embeddings.extend(embed_texts(texts[start : start + batch_size]))
    return embeddings

# 根路径，简单的欢迎信息。
@app.get("/")
def read_root():
    return {
        "message": "RAG MVP backend is running",
        "service": "backend",
        "version": "0.1.0",
    }

# 健康检查端点，返回服务状态和 Azure OpenAI 配置状态。
@app.get("/health")
def health_check():
    config_report = get_settings_health_report()
    return {
        "status": "ok",
        "service_ready": config_report["config_validated"],
        "config": config_report,
    }

# 文件上传端点，接受 PDF 文件并保存到服务器。
@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is missing.")

    ensure_upload_size(file)
    safe_filename = normalize_filename(file.filename)
    if not safe_filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    file_path = UPLOAD_DIR / safe_filename

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "message": "File uploaded successfully",
        "filename": safe_filename,
        "saved_path": str(file_path),
    }

# 定义一个 GET 端点，接受文件名参数，调用 extract_text_from_pdf 函数，并返回提取的文本内容和结构化信息。
@app.get("/extract-text/{filename}")
def extract_text(filename: str):
    safe_filename = normalize_filename(filename)
    file_path = UPLOAD_DIR / safe_filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found.")

    try:
        result = extract_text_from_pdf(file_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract text: {str(e)}")


# 定义一个 GET 端点，接受文件名、chunk_size 和 chunk_overlap 参数，调用 chunk_text 函数，并返回切分后的 chunks。
@app.get("/chunk/{filename}")
def chunk_document(
    filename: str,
    chunk_size: int = Query(500, gt=0),
    chunk_overlap: int = Query(100, ge=0),
):
    safe_filename = normalize_filename(filename)
    file_path = UPLOAD_DIR / safe_filename

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
            "filename": safe_filename,
            "page_count": extraction_result["page_count"],
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "chunk_count": len(chunks),
            "chunks": chunks,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to chunk document: {str(e)}")
    

# 定义一个 GET 端点，接受文件名、chunk_size 和 chunk_overlap 参数，执行切分和嵌入，并返回嵌入结果。   
@app.get("/embed/{filename}")
def embed_document(
    filename: str,
    chunk_size: int = Query(500, gt=0),
    chunk_overlap: int = Query(100, ge=0),
):
    safe_filename = normalize_filename(filename)
    file_path = UPLOAD_DIR / safe_filename

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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate embeddings: {str(e)}")
    
# 定义一个 POST 端点，接受文件名、chunk_size 和 chunk_overlap 参数，执行切分、嵌入和写入向量数据库，并返回操作结果。
@app.post("/index/{filename}")
def index_document(
    filename: str,
    chunk_size: int = Query(500, gt=0),
    chunk_overlap: int = Query(100, ge=0),
):
    safe_filename = normalize_filename(filename)
    file_path = UPLOAD_DIR / safe_filename

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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to index document: {str(e)}")
    
# 定义请求模型，用于接收用户问题和可选的 top_k 参数。
class AskRequest(BaseModel):
    question: str
    top_k: int = 3
    filename: str | None = None

# 定义一个POST 端点，接受用户问题，执行 RAG 流程，并返回答案和相关信息。
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to answer question: {str(e)}")
