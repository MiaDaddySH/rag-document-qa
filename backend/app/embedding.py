from app.config import settings
from app.llm_client import get_llm_client, run_openai_with_retry

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

# 兼容单条输入的便捷方法。
def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0]
