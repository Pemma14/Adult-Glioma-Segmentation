import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__))).parent


class AppSettings(BaseModel):
    HOST: str = "0.0.0.0"
    PORT: int = 8500
    DEBUG: bool = False


class GliomaSettings(BaseModel):
    MODEL_VERSION: str
    UPLOAD_DIR: Path = PROJECT_ROOT / "data" / "uploads"
    OUTPUT_DIR: Path = PROJECT_ROOT / "data" / "outputs"
    MAX_UPLOAD_SIZE_MB: int = 512
    ALLOWED_EXTENSIONS: set[str] = {".nii", ".nii.gz"}

    @field_validator("UPLOAD_DIR", "OUTPUT_DIR", mode="before")
    @classmethod
    def _resolve_path(cls, value):
        if isinstance(value, (str, os.PathLike)):
            path = Path(value)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            return path
        return value

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024


class DicomSettings(BaseModel):
    TEMP_DIR: Path = PROJECT_ROOT / "data" / "dicom_temp"

    @field_validator("TEMP_DIR", mode="before")
    @classmethod
    def _resolve_path(cls, value):
        if isinstance(value, (str, os.PathLike)):
            path = Path(value)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            return path
        return value


class AuthSettings(BaseModel):
    API_KEY: str


class DbSettings(BaseModel):
    HOST: str = "database"
    PORT: int = 5432
    USER: str = "pema"
    PASSWORD: str = "password"
    NAME: str = "glioma_service"
    ECHO: bool = False
    POOL_SIZE: int = 5
    MAX_OVERFLOW: int = 10
    POOL_RECYCLE: int = 3600
    POOL_PRE_PING: bool = True

    @property
    def url_psycopg(self) -> str:
        return f"postgresql+psycopg://{self.USER}:{self.PASSWORD}@{self.HOST}:{self.PORT}/{self.NAME}"

    def get_engine_kwargs(self) -> dict:
        return {
            "url": self.url_psycopg,
            "echo": self.ECHO,
            "pool_size": self.POOL_SIZE,
            "max_overflow": self.MAX_OVERFLOW,
            "pool_recycle": self.POOL_RECYCLE,
            "pool_pre_ping": self.POOL_PRE_PING
        }


class MQSettings(BaseModel):
    HOST: str = "rabbitmq"
    PORT: int = 5672
    VIRTUAL_HOST: str = "/"
    USER: str = "guest"
    PASSWORD: str = "guest"
    QUEUE_NAME: str = "ml_task_queue"
    EXCHANGE_NAME: str = "ml_tasks_exchange"
    RESULTS_EXCHANGE_NAME: str = "ml_results_exchange"
    RESULTS_QUEUE_NAME: str = "ml_results_queue"
    RESULTS_ROUTING_KEY: str = "ml_results_queue"
    RETRY_ATTEMPTS: int = 3
    RETRY_MULTIPLIER: float = 0.5
    RETRY_MIN: int = 1
    RETRY_MAX: int = 5
    HEARTBEAT: int = 30
    TIMEOUT: int = 2

    @property
    def amqp_url(self) -> str:
        return f"amqp://{self.USER}:{self.PASSWORD}@{self.HOST}:{self.PORT}{self.VIRTUAL_HOST}?heartbeat={self.HEARTBEAT}"


class Settings(BaseSettings):
    app: AppSettings = AppSettings()
    db: DbSettings = DbSettings()
    auth: AuthSettings
    mq: MQSettings = MQSettings()
    glioma: GliomaSettings
    dicom: DicomSettings = DicomSettings()

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_nested_delimiter="__",
        env_prefix=""
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
