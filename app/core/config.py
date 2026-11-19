from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    app_name: str = "Enterprise IT Support Agentic RAG Copilot"
    app_env: str = "development"
    gemini_api_key: str = ""
    tavily_api_key: str = ""
    pinecone_api_key: str = ""
    pinecone_index_name: str = "fde-it-support-rag-gemini"
    pinecone_namespace: str = "company-it-kb"
    embedding_model: str = "models/gemini-embedding-001"
    gemini_model: str = "gemini-3.5-flash-lite"
    top_k: int = 4
    max_retries: int = 1
    admin_api_key: str = "nova-admin-secret-key-2026"
    audit_db_path: str = str(BASE_DIR / "data" / "audit.db")
    upload_dir: str = str(BASE_DIR / "uploads")
    sample_kb_dir: str = str(BASE_DIR / "data" / "sample_kb")
    cors_origins: str = "http://localhost:8080,http://127.0.0.1:8080,http://localhost:3000,http://127.0.0.1:3000"
    max_upload_size_mb: int = 25

    @property
    def allowed_origins(self) -> list[str]:
        if isinstance(self.cors_origins, list):
            return self.cors_origins
        return [o.strip() for o in str(self.cors_origins).split(",") if o.strip()]

    model_config = SettingsConfigDict(env_file=str(BASE_DIR / ".env"), extra="ignore")




@lru_cache
def get_settings() -> Settings:
    return Settings()
