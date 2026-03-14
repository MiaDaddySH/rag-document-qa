import random
import time
from collections.abc import Callable
from typing import TypeVar

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError

from app.config import settings

T = TypeVar("T")

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
        timeout=settings.llm_timeout_seconds,
        max_retries=0,
    )


def is_retryable_openai_error(error: Exception) -> bool:
    if isinstance(error, (APITimeoutError, APIConnectionError, RateLimitError)):
        return True
    if isinstance(error, APIStatusError):
        return error.status_code in {408, 409, 429} or error.status_code >= 500
    return False


def run_openai_with_retry(operation: Callable[[], T], operation_name: str) -> T:
    max_retries = settings.llm_max_retries
    base_delay = settings.llm_retry_base_delay_seconds
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return operation()
        except Exception as error:
            last_error = error
            if attempt >= max_retries or not is_retryable_openai_error(error):
                raise RuntimeError(f"{operation_name} failed: {error}") from error
            sleep_seconds = base_delay * (2 ** attempt) + random.uniform(0, max(base_delay * 0.25, 0.01))
            time.sleep(sleep_seconds)

    raise RuntimeError(f"{operation_name} failed: {last_error}")


def probe_azure_openai_dependency() -> tuple[bool, str]:
    endpoint = settings.azure_openai_endpoint.rstrip("/")
    url = f"{endpoint}/openai/deployments"
    params = {"api-version": "2024-10-21"}
    headers = {"api-key": settings.azure_openai_api_key}
    try:
        response = httpx.get(
            url,
            params=params,
            headers=headers,
            timeout=settings.dependency_probe_timeout_seconds,
        )
        if response.status_code == 200:
            return True, "ok"
        return False, f"http_{response.status_code}"
    except Exception as error:
        return False, str(error)
