from collections import OrderedDict
import threading

from app.config import settings
from app.llm_client import get_llm_client, run_openai_with_retry

_QUERY_EMBEDDING_CACHE: OrderedDict[str, list[float]] = OrderedDict()
_QUERY_EMBEDDING_CACHE_LOCK = threading.Lock()

# 批量调用 Azure OpenAI 的 embedding API，减少网络开销。
def embed_texts(texts: list[str]) -> list[list[float]]:
    normalized_texts = [text.strip() for text in texts if text and text.strip()]
    if not normalized_texts:
        raise ValueError("Input texts for embedding are empty.")

    if not settings.azure_openai_embedding_deployment:
        raise ValueError("AZURE_OPENAI_EMBEDDING_DEPLOYMENT is not configured.")

    client = get_llm_client()

    response = run_openai_with_retry(
        operation=lambda: client.embeddings.create(
            model=settings.azure_openai_embedding_deployment,
            input=normalized_texts,
        ),
        operation_name="create_embeddings",
    )

    return [item.embedding for item in response.data]

def get_cached_query_embedding(text: str) -> list[float] | None:
    if not settings.rag_query_embedding_cache_enabled:
        return None
    key = text.strip()
    if not key:
        return None
    with _QUERY_EMBEDDING_CACHE_LOCK:
        cached = _QUERY_EMBEDDING_CACHE.get(key)
        if cached is None:
            return None
        _QUERY_EMBEDDING_CACHE.move_to_end(key)
        return cached


def set_cached_query_embedding(text: str, embedding: list[float]) -> None:
    if not settings.rag_query_embedding_cache_enabled:
        return
    key = text.strip()
    if not key:
        return
    with _QUERY_EMBEDDING_CACHE_LOCK:
        _QUERY_EMBEDDING_CACHE[key] = embedding
        _QUERY_EMBEDDING_CACHE.move_to_end(key)
        while len(_QUERY_EMBEDDING_CACHE) > settings.rag_query_embedding_cache_size:
            _QUERY_EMBEDDING_CACHE.popitem(last=False)


def embed_text(text: str) -> list[float]:
    cached = get_cached_query_embedding(text)
    if cached is not None:
        return cached
    embedding = embed_texts([text])[0]
    set_cached_query_embedding(text, embedding)
    return embedding
