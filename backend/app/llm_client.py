from openai import OpenAI

from app.config import settings

# 负责创建和配置 OpenAI 客户端，以便在其他模块中调用 Azure OpenAI 的 API。
def get_llm_client() -> OpenAI:
    if not settings.azure_openai_api_key:
        raise ValueError("AZURE_OPENAI_API_KEY is not configured.")

    if not settings.azure_openai_endpoint:
        raise ValueError("AZURE_OPENAI_ENDPOINT is not configured.")

    base_url = f"{settings.azure_openai_endpoint.rstrip('/')}/openai/v1/"

    return OpenAI(
        api_key=settings.azure_openai_api_key,
        base_url=base_url,
    )