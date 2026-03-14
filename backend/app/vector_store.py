import random
import time
from collections.abc import Callable
from typing import TypeVar
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.config import settings

T = TypeVar("T")

# 负责创建 Qdrant 客户端。
def get_qdrant_client() -> QdrantClient:
    if settings.qdrant_api_key:
        return QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=settings.qdrant_timeout_seconds,
        )

    return QdrantClient(
        url=settings.qdrant_url,
        timeout=settings.qdrant_timeout_seconds,
    )


def is_retryable_qdrant_error(error: Exception) -> bool:
    message = str(error).lower()
    retry_signals = [
        "timeout",
        "tempor",
        "connection",
        "unavailable",
        "deadline",
        "reset",
        "429",
        "502",
        "503",
        "504",
    ]
    return any(signal in message for signal in retry_signals)


def run_qdrant_with_retry(operation: Callable[[], T], operation_name: str) -> T:
    max_retries = settings.qdrant_max_retries
    base_delay = settings.qdrant_retry_base_delay_seconds
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return operation()
        except Exception as error:
            last_error = error
            if attempt >= max_retries or not is_retryable_qdrant_error(error):
                raise RuntimeError(f"{operation_name} failed: {error}") from error
            sleep_seconds = base_delay * (2 ** attempt) + random.uniform(0, max(base_delay * 0.25, 0.01))
            time.sleep(sleep_seconds)

    raise RuntimeError(f"{operation_name} failed: {last_error}")

# 确保 collection 存在。
# 如果不存在，就按给定的向量维度创建。
def ensure_collection(vector_size: int) -> None:
    client = get_qdrant_client()
    collection_name = settings.qdrant_collection_name

    collections = run_qdrant_with_retry(
        operation=lambda: client.get_collections().collections,
        operation_name="get_collections",
    )
    exists = any(collection.name == collection_name for collection in collections)

    if not exists:
        run_qdrant_with_retry(
            operation=lambda: client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            ),
            operation_name="create_collection",
        )

# 删除 Qdrant 中所有 filename 为指定值的 chunks。
def delete_chunks_by_filename(filename: str) -> None:
    client = get_qdrant_client()

    run_qdrant_with_retry(
        operation=lambda: client.delete(
            collection_name=settings.qdrant_collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="filename",
                        match=MatchValue(value=filename),
                    )
                ]
            ),
        ),
        operation_name="delete_points_by_filename",
    )

# 把 chunk + embedding + metadata 一起写入 Qdrant。
def upsert_chunks(chunks_with_embeddings: list[dict]) -> int:
    if not chunks_with_embeddings:
        return 0

    client = get_qdrant_client()
    collection_name = settings.qdrant_collection_name

    vector_size = len(chunks_with_embeddings[0]["embedding"])
    ensure_collection(vector_size=vector_size)

    points = []
    for item in chunks_with_embeddings:
        point = PointStruct(
            id=str(uuid4()),
            vector=item["embedding"],
            payload={
                "filename": item["filename"],
                "chunk_index": item["chunk_index"],
                "text": item["content"],
                "start_char": item["start_char"],
                "end_char": item["end_char"],
                "page_count": item["page_count"],
            },
        )
        points.append(point)

    run_qdrant_with_retry(
        operation=lambda: client.upsert(
            collection_name=collection_name,
            points=points,
        ),
        operation_name="upsert_points",
    )

    return len(points)

# 根据查询向量，在 Qdrant 中检索相似的 chunks，并返回它们的文本内容和相关信息。可以选择只检索特定文件的 chunks。
def search_similar_chunks(
    query_vector: list[float],
    limit: int = 3,
    filename: str | None = None,
) -> list[dict]:
    client = get_qdrant_client()

    query_filter = None
    if filename:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="filename",
                    match=MatchValue(value=filename),
                )
            ]
        )

    result = run_qdrant_with_retry(
        operation=lambda: client.query_points(
            collection_name=settings.qdrant_collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
            query_filter=query_filter,
        ),
        operation_name="query_points",
    )

    points = result.points if hasattr(result, "points") else []

    matches = []
    for point in points:
        payload = point.payload or {}
        matches.append(
            {
                "score": point.score,
                "filename": payload.get("filename"),
                "chunk_index": payload.get("chunk_index"),
                "text": payload.get("text"),
                "start_char": payload.get("start_char"),
                "end_char": payload.get("end_char"),
                "page_count": payload.get("page_count"),
            }
        )

    return matches


def probe_qdrant_dependency() -> tuple[bool, str]:
    client = get_qdrant_client()
    try:
        run_qdrant_with_retry(
            operation=lambda: client.get_collections(),
            operation_name="probe_qdrant_collections",
        )
        return True, "ok"
    except Exception as error:
        return False, str(error)
