from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.config import settings

# 负责创建 Qdrant 客户端。
def get_qdrant_client() -> QdrantClient:
    if settings.qdrant_api_key:
        return QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )

    return QdrantClient(url=settings.qdrant_url)

# 确保 collection 存在。
# 如果不存在，就按给定的向量维度创建。
def ensure_collection(vector_size: int) -> None:
    client = get_qdrant_client()
    collection_name = settings.qdrant_collection_name

    collections = client.get_collections().collections
    exists = any(collection.name == collection_name for collection in collections)

    if not exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
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

    client.upsert(
        collection_name=collection_name,
        points=points,
    )

    return len(points)

# 给定一个查询向量，返回最相似的 chunks。
def search_similar_chunks(query_vector: list[float], limit: int = 3) -> list[dict]:
    client = get_qdrant_client()

    result = client.query_points(
        collection_name=settings.qdrant_collection_name,
        query=query_vector,
        limit=limit,
        with_payload=True,
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