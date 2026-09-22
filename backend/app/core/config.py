from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "FCA Fines RAG Assistant"
    app_version: str = "1.0.0"
    allowed_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    llm_model: str = "llama3.2:3b"
    ollama_url: str = "http://localhost:11434"
    ollama_keep_alive: str = "30m"
    llm_num_predict: int = 60
    llm_num_ctx: int = 1536
    llm_timeout_seconds: int = 180
    enable_llm_synthesis: bool = False

    top_k: int = 3
    retrieval_k: int = 15
    min_relevance: float = 0.25
    max_context_chars: int = 2500
    max_chars_per_hit: int = 1800

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    def cors_origins(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.allowed_origins.split(",")
            if origin.strip()
        ]


settings = Settings()