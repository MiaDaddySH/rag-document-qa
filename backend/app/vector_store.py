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