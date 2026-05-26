"""pydantic-settings 配置，所有运行参数通过 .env 注入。"""
from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class AblationFlags(BaseSettings):
    """消融实验开关 —— 每个开关都对应 graph 里某个节点的短路。"""
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    disable_rag: bool = Field(default=False, alias="DISABLE_RAG")
    disable_consistency: bool = Field(default=False, alias="DISABLE_CONSISTENCY")
    disable_case_library: bool = Field(default=False, alias="DISABLE_CASE_LIBRARY")
    disable_similar_cases: bool = Field(default=False, alias="DISABLE_SIMILAR_CASES")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- Ollama ----
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3.5:35b-a3b"
    ollama_vl_model: str = "qwen3-vl:8b"
    diagnosis_temperatures: Annotated[list[float], NoDecode] = Field(default_factory=lambda: [0.3, 0.5, 0.7])
    default_temperature: float = 0.2
    llm_timeout_seconds: int = 180

    # ---- Vision service ----
    vision_api_base_url: str = "http://localhost:8765"
    vision_api_timeout_seconds: int = 600

    # ---- Case DB ----
    case_db_backend: str = "JSONL"   # "JSONL" | "POSTGRES"
    case_db_jsonl_path: Path = PROJECT_ROOT / "logs" / "case_library.jsonl"
    postgres_dsn: str = "postgresql://eye_agent:eye_agent@localhost:5432/eye_agent"
    postgres_vector_dim: int = 1024

    # ---- Embedding ----
    embedding_model_name: str = "BAAI/bge-m3"
    embedding_device: str = "cuda"
    embedding_dim: int = 1024

    # ---- RAG ----
    rag_docs_dir: Path = PROJECT_ROOT / "scripts" / "knowledge_docs"
    rag_index_dir: Path = PROJECT_ROOT / "logs" / "rag_index"
    rag_top_k: int = 4

    # ---- Logs ----
    log_dir: Path = PROJECT_ROOT / "logs"
    llm_trace_dir: Path = PROJECT_ROOT / "logs" / "llm_traces"
    log_level: str = "INFO"

    # ---- 消融开关 ----
    ablation: AblationFlags = Field(default_factory=AblationFlags)

    @field_validator("diagnosis_temperatures", mode="before")
    @classmethod
    def _parse_temps(cls, v):
        if isinstance(v, str):
            return [float(x.strip()) for x in v.split(",") if x.strip()]
        return v

    def ensure_dirs(self) -> None:
        for p in (self.log_dir, self.llm_trace_dir, self.rag_index_dir):
            p.mkdir(parents=True, exist_ok=True)
        if self.case_db_backend.upper() == "JSONL":
            self.case_db_jsonl_path.parent.mkdir(parents=True, exist_ok=True)


# 模块级单例，所有 agent 共享
settings = Settings()
settings.ensure_dirs()
