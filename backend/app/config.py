from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


REQUIRED_ENV_ALIAS = {
    "azure_openai_api_key": "AZURE_OPENAI_API_KEY",
    "azure_openai_endpoint": "AZURE_OPENAI_ENDPOINT",
    "azure_openai_deployment": "AZURE_OPENAI_DEPLOYMENT",
    "azure_openai_embedding_deployment": "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
}


class Settings(BaseSettings):
    azure_openai_api_key: str = Field(..., alias="AZURE_OPENAI_API_KEY")
    azure_openai_endpoint: str = Field(..., alias="AZURE_OPENAI_ENDPOINT")
    azure_openai_deployment: str = Field(..., alias="AZURE_OPENAI_DEPLOYMENT")
    azure_openai_embedding_deployment: str = Field(
        ...,
        alias="AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
    )

    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL", min_length=1)
    qdrant_api_key: str = Field(default="", alias="QDRANT_API_KEY")
    qdrant_collection_name: str = Field(
        default="rag_documents",
        alias="QDRANT_COLLECTION_NAME",
        min_length=1,
    )

    rag_min_score: float = Field(default=0.2, alias="RAG_MIN_SCORE", ge=0.0, le=1.0)
    rag_context_max_chars: int = Field(default=6000, alias="RAG_CONTEXT_MAX_CHARS", ge=1)
    rag_retrieval_multiplier: int = Field(default=3, alias="RAG_RETRIEVAL_MULTIPLIER", ge=1, le=20)
    rag_rerank_vector_weight: float = Field(default=0.7, alias="RAG_RERANK_VECTOR_WEIGHT", ge=0.0, le=1.0)
    rag_rerank_keyword_weight: float = Field(default=0.3, alias="RAG_RERANK_KEYWORD_WEIGHT", ge=0.0, le=1.0)
    embedding_batch_size: int = Field(default=32, alias="EMBEDDING_BATCH_SIZE", ge=1, le=2048)
    upload_max_mb: int = Field(default=20, alias="UPLOAD_MAX_MB", ge=1, le=1024)
    llm_timeout_seconds: float = Field(default=30.0, alias="LLM_TIMEOUT_SECONDS", ge=0.1, le=300.0)
    llm_max_retries: int = Field(default=2, alias="LLM_MAX_RETRIES", ge=0, le=10)
    llm_retry_base_delay_seconds: float = Field(
        default=1.0,
        alias="LLM_RETRY_BASE_DELAY_SECONDS",
        ge=0.0,
        le=30.0,
    )
    qdrant_timeout_seconds: float = Field(default=10.0, alias="QDRANT_TIMEOUT_SECONDS", ge=0.1, le=120.0)
    qdrant_max_retries: int = Field(default=2, alias="QDRANT_MAX_RETRIES", ge=0, le=10)
    qdrant_retry_base_delay_seconds: float = Field(
        default=0.5,
        alias="QDRANT_RETRY_BASE_DELAY_SECONDS",
        ge=0.0,
        le=30.0,
    )
    dependency_probe_timeout_seconds: float = Field(
        default=5.0,
        alias="DEPENDENCY_PROBE_TIMEOUT_SECONDS",
        ge=0.1,
        le=60.0,
    )
    log_success_sample_rate: float = Field(default=0.2, alias="LOG_SUCCESS_SAMPLE_RATE", ge=0.0, le=1.0)
    log_slow_request_ms: int = Field(default=1200, alias="LOG_SLOW_REQUEST_MS", ge=1, le=60000)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator(
        "azure_openai_api_key",
        "azure_openai_endpoint",
        "azure_openai_deployment",
        "azure_openai_embedding_deployment",
        "qdrant_url",
        "qdrant_collection_name",
        mode="before",
    )
    @classmethod
    def strip_string(cls, value: str) -> str:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator(
        "azure_openai_api_key",
        "azure_openai_endpoint",
        "azure_openai_deployment",
        "azure_openai_embedding_deployment",
    )
    @classmethod
    def validate_required_non_empty(cls, value: str, info) -> str:
        if not value:
            alias = REQUIRED_ENV_ALIAS.get(info.field_name, info.field_name)
            raise ValueError(f"{alias} 不能为空，请在 backend/.env 中配置。")
        return value

    @field_validator("azure_openai_endpoint")
    @classmethod
    def validate_endpoint(cls, value: str) -> str:
        if not (value.startswith("http://") or value.startswith("https://")):
            raise ValueError("AZURE_OPENAI_ENDPOINT 必须以 http:// 或 https:// 开头。")
        return value.rstrip("/")


def load_settings() -> Settings:
    try:
        return Settings()
    except ValidationError as exc:
        details = []
        for error in exc.errors():
            location = ".".join(str(item) for item in error.get("loc", []))
            message = error.get("msg", "Invalid value")
            details.append(f"- {location}: {message}")
        detail_text = "\n".join(details)
        raise RuntimeError(
            "配置校验失败，请检查 backend/.env：\n"
            f"{detail_text}"
        ) from exc


settings = load_settings()


def get_settings_health_report() -> dict:
    azure_endpoint = settings.azure_openai_endpoint
    azure_host = ""
    if "://" in azure_endpoint:
        azure_host = azure_endpoint.split("://", 1)[1]

    return {
        "config_validated": True,
        "required_config": {
            "azure_openai_api_key_configured": bool(settings.azure_openai_api_key),
            "azure_openai_endpoint_configured": bool(settings.azure_openai_endpoint),
            "azure_openai_deployment_configured": bool(settings.azure_openai_deployment),
            "azure_openai_embedding_deployment_configured": bool(
                settings.azure_openai_embedding_deployment
            ),
        },
        "security": {
            "qdrant_api_key_configured": bool(settings.qdrant_api_key),
        },
        "effective_limits": {
            "rag_min_score": settings.rag_min_score,
            "rag_context_max_chars": settings.rag_context_max_chars,
            "rag_retrieval_multiplier": settings.rag_retrieval_multiplier,
            "rag_rerank_vector_weight": settings.rag_rerank_vector_weight,
            "rag_rerank_keyword_weight": settings.rag_rerank_keyword_weight,
            "embedding_batch_size": settings.embedding_batch_size,
            "upload_max_mb": settings.upload_max_mb,
            "llm_timeout_seconds": settings.llm_timeout_seconds,
            "llm_max_retries": settings.llm_max_retries,
            "qdrant_timeout_seconds": settings.qdrant_timeout_seconds,
            "qdrant_max_retries": settings.qdrant_max_retries,
            "dependency_probe_timeout_seconds": settings.dependency_probe_timeout_seconds,
            "log_success_sample_rate": settings.log_success_sample_rate,
            "log_slow_request_ms": settings.log_slow_request_ms,
        },
        "targets": {
            "azure_openai_endpoint_host": azure_host,
            "qdrant_url": settings.qdrant_url,
            "qdrant_collection_name": settings.qdrant_collection_name,
        },
    }
