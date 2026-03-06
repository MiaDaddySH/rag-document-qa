from app.config import settings
from app.llm_client import get_llm_client


def embed_text(text: str) -> list[float]:
    if not text.strip():
        raise ValueError("Input text for embedding is empty.")

    if not settings.azure_openai_embedding_deployment:
        raise ValueError("AZURE_OPENAI_EMBEDDING_DEPLOYMENT is not configured.")

    client = get_llm_client()

    response = client.embeddings.create(
        model=settings.azure_openai_embedding_deployment,
        input=text,
    )

    return response.data[0].embedding