from pathlib import Path
import shutil

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from app.chunker import chunk_text
from app.config import settings
from app.document_loader import extract_text_from_pdf
from app.embedding import embed_text
from app.vector_store import upsert_chunks
from pydantic import BaseModel
from app.rag_pipeline import answer_question

# 主应用实例
app = FastAPI(
    title="RAG MVP Backend",
    version="0.1.0",
    description="Backend service for PDF upload, retrieval, and question answering."
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

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
    return {
        "status": "ok",
        "azure_endpoint_configured": bool(settings.azure_openai_endpoint),
    }

# 文件上传端点，接受 PDF 文件并保存到服务器。
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

# 定义一个 GET 端点，接受文件名参数，调用 extract_text_from_pdf 函数，并返回提取的文本内容和结构化信息。
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


# 定义一个 GET 端点，接受文件名、chunk_size 和 chunk_overlap 参数，调用 chunk_text 函数，并返回切分后的 chunks。
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
    

# 定义一个 GET 端点，接受文件名、chunk_size 和 chunk_overlap 参数，执行切分和嵌入，并返回嵌入结果。   
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
    
# 定义一个 POST 端点，接受文件名、chunk_size 和 chunk_overlap 参数，执行切分、嵌入和向量存储，并返回处理结果。
@app.post("/index/{filename}")
def index_document(
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

        chunks_with_embeddings = []
        for chunk in chunks:
            embedding = embed_text(chunk["content"])
            chunks_with_embeddings.append(
                {
                    "filename": filename,
                    "page_count": extraction_result["page_count"],
                    "chunk_index": chunk["chunk_index"],
                    "content": chunk["content"],
                    "start_char": chunk["start_char"],
                    "end_char": chunk["end_char"],
                    "embedding": embedding,
                }
            )

        inserted_count = upsert_chunks(chunks_with_embeddings)

        return {
            "message": "Document indexed successfully",
            "filename": filename,
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

# 定义一个POST 端点，接受用户问题，执行 RAG 流程，并返回答案和相关信息。
@app.post("/ask")
def ask_question(request: AskRequest):
    try:
        result = answer_question(
            question=request.question,
            top_k=request.top_k,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to answer question: {str(e)}")