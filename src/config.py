from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "Label Guardian API"
    app_description: str = "QA orchestration API for 2D perception annotations"
    app_version: str = "0.1.0"
    app_env: Literal["development", "production", "test"] = "development"
    app_port: int = Field(default=8000, ge=1, le=65535)
    app_host: str = "0.0.0.0"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # Authentication. Supabase Auth owns credentials and sessions; this API
    # verifies its access tokens and stores only application profiles/roles.
    auth_enabled: bool = False
    supabase_url: str | None = None
    supabase_jwt_audience: str = "authenticated"
    supabase_jwt_secret: SecretStr | None = None
    supabase_jwks_url: str | None = None
    supabase_service_role_key: SecretStr | None = None
    auth_bootstrap_admin_emails: str = ""
    auth_dev_user_id: str = "local-admin"
    auth_dev_user_email: str = "admin@label-guardian.local"
    auth_dev_user_name: str = "Local Administrator"

    # LLM
    openai_api_key: str = ""
    model_name: str = "gpt-4o-mini"
    llm_temperature: float = Field(default=0.7, ge=0.0, le=2.0)

    # Label QA Agent (optional runtime)
    # ``auto`` prefers OpenAI when both provider keys are present.
    label_qa_llm_provider: Literal["auto", "openai", "gemini"] = "auto"
    google_api_key: SecretStr | None = None
    google_model_name: str = "gemini-flash-latest"
    # Production can keep QA orchestration in this API while delegating detector
    # execution to a separate GPU service.
    inference_mode: Literal["local", "remote"] = "local"
    inference_service_url: str | None = None
    inference_service_token: SecretStr | None = None
    inference_request_timeout_seconds: float = Field(default=30.0, gt=0.0, le=300.0)
    # CPU-safe production default. Use a larger checkpoint only on a dedicated
    # inference service with enough memory/compute.
    yolo_model_name: str = "yolo26n.pt"
    yolo_confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    agent_evaluation_cache_entries: int = Field(default=128, ge=1, le=10_000)

    # Local real-data MVP. The backend serves only files resolved below this root.
    dataset_backend: Literal["filesystem", "database"] = "filesystem"
    dataset_root: Path = Path("data/class.txt")
    dataset_id: str = "local-yolo"
    dataset_version: str = "workspace"
    dataset_default_split: str = "val"

    # Database
    database_url: str = "postgresql+asyncpg://label_guardian:label_guardian_dev@localhost:5432/label_guardian"
    database_pool_size: int = Field(default=5, ge=1, le=50)
    database_max_overflow: int = Field(default=10, ge=0, le=50)

    # Vector Store
    chroma_persist_dir: str = "./data/chroma"

    @field_validator("google_api_key", mode="before")
    @classmethod
    def normalize_empty_token(cls, value: object) -> object | None:
        if isinstance(value, str):
            normalized = value.strip()
            if not normalized:
                return None
            return normalized
        return value

    @field_validator("supabase_jwt_secret", mode="before")
    @classmethod
    def normalize_empty_jwt_secret(cls, value: object) -> object | None:
        if isinstance(value, str):
            return value.strip() or None
        return value

    @model_validator(mode="after")
    def validate_production_auth(self) -> "Settings":
        if self.app_env == "production" and not self.auth_enabled:
            raise ValueError("AUTH_ENABLED must be true in production.")
        if self.auth_enabled and not self.supabase_url:
            raise ValueError("SUPABASE_URL is required when AUTH_ENABLED is true.")
        if self.inference_mode == "remote" and not self.inference_service_url:
            raise ValueError("INFERENCE_SERVICE_URL is required when INFERENCE_MODE=remote.")
        if self.app_env == "production":
            if self.dataset_backend != "database":
                raise ValueError("DATASET_BACKEND must be database in production.")
            if self.dataset_id.startswith("<") or self.dataset_version.startswith("<"):
                raise ValueError("DATASET_ID and DATASET_VERSION must be configured in production.")
            if "localhost" in self.database_url or "127.0.0.1" in self.database_url:
                raise ValueError("DATABASE_URL must point to a reachable production PostgreSQL service.")
            origins = self.cors_origin_values
            if not origins or any("*" in origin for origin in origins):
                raise ValueError("CORS_ORIGINS must contain explicit frontend origins in production.")
            if any("localhost" in origin.lower() or "127.0.0.1" in origin for origin in origins):
                raise ValueError("CORS_ORIGINS must not contain local origins in production.")
            invalid_origins = []
            for origin in origins:
                parsed_origin = urlparse(origin)
                if (
                    parsed_origin.scheme != "https"
                    or not parsed_origin.netloc
                    or parsed_origin.path
                    or parsed_origin.params
                    or parsed_origin.query
                    or parsed_origin.fragment
                ):
                    invalid_origins.append(origin)
            if invalid_origins:
                raise ValueError(
                    "CORS_ORIGINS must contain HTTPS origins without paths, queries or trailing slashes in production."
                )
            parsed_supabase_url = urlparse(self.supabase_url or "")
            if parsed_supabase_url.scheme != "https" or not parsed_supabase_url.netloc:
                raise ValueError("SUPABASE_URL must be an HTTPS URL in production.")
            if not self.auth_bootstrap_admin_email_values:
                raise ValueError(
                    "AUTH_BOOTSTRAP_ADMIN_EMAILS must contain at least one recovery administrator in production."
                )
        return self

    @property
    def cors_origin_values(self) -> list[str]:
        """Return normalized origins for FastAPI's CORS middleware."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def auth_bootstrap_admin_email_values(self) -> set[str]:
        return {
            email.strip().lower()
            for email in self.auth_bootstrap_admin_emails.split(",")
            if email.strip()
        }


class IngestionSettings(BaseSettings):
    """Configuration for the optional local/cloud dataset ingestion workflow."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="LABEL_GUARDIAN_",
        case_sensitive=False,
        extra="ignore",
    )

    database_url: str = "postgresql+psycopg://label_guardian:label_guardian_dev@localhost:5432/label_guardian"
    storage_backend: Literal["gcs"] = "gcs"
    gcs_bucket: str | None = None
    gcs_project: str | None = None
    gcs_credentials_path: Path | None = None
    gcs_credentials_json: SecretStr | None = None
    gcs_public_url: str | None = None
    object_key_prefix: str = ""
    dataset_provider: str | None = None
    dataset_name: str | None = None
    dataset_release: str | None = None

    @field_validator("gcs_credentials_json", mode="before")
    @classmethod
    def normalize_empty_credentials_json(cls, value: object) -> object | None:
        if isinstance(value, str):
            return value.strip() or None
        return value

    @property
    def bucket_name(self) -> str:
        if not self.gcs_bucket:
            raise ValueError("LABEL_GUARDIAN_GCS_BUCKET is required for ingestion.")
        return self.gcs_bucket

    @property
    def object_url_base(self) -> str:
        return (self.gcs_public_url or f"gs://{self.bucket_name}").rstrip("/")

    def object_uri(self, object_key: str) -> str:
        key = object_key.lstrip("/")
        if self.gcs_public_url:
            return f"{self.gcs_public_url.rstrip('/')}/{key}"
        return f"gs://{self.bucket_name}/{key}"


class InferenceServiceSettings(BaseSettings):
    """Configuration for the standalone detector runtime."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    inference_app_name: str = "Label Guardian Inference Service"
    inference_app_version: str = "0.1.0"
    inference_app_env: Literal["development", "production", "test"] = "development"
    inference_auth_token: SecretStr | None = None
    inference_model_name: str = "yolo26x.pt"
    inference_model_version: str | None = None
    inference_model_cache_dir: Path = Path("/tmp/label-guardian-models")
    inference_confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    inference_allowed_object_prefixes: str = "datasets/official"
    inference_max_batch_size: int = Field(default=64, ge=1, le=256)

    @field_validator("inference_auth_token", mode="before")
    @classmethod
    def normalize_empty_inference_auth_token(cls, value: object) -> object | None:
        if isinstance(value, str):
            return value.strip() or None
        return value

    @property
    def allowed_object_prefix_values(self) -> list[str]:
        return [
            prefix.strip().strip("/")
            for prefix in self.inference_allowed_object_prefixes.split(",")
            if prefix.strip().strip("/")
        ]


@lru_cache
def get_settings() -> Settings:
    return Settings()
