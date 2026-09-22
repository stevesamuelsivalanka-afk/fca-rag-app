# from pydantic_settings import BaseSettings, SettingsConfigDict


# class Settings(BaseSettings):
#     # llama3.2:1b is substantially faster on a local CPU. Set LLM_MODEL to
#     # llama3.2:3b when answer quality is more important than latency.
#     llm_model: str = "llama3.2:3b" #1b
#     ollama_url: str = "http://localhost:11434"
#     ollama_keep_alive: str = "30m"

#     top_k: int = 3 #2
#     min_relevance: float = 0.25
#     max_context_chars: int =  2500 #1800#4000
#     llm_num_predict: int =60  #120
#     llm_num_ctx: int = 1536    #2048  #4096 
#     llm_timeout_seconds: int =180 #300  # 60#120

#     model_config = SettingsConfigDict(
#         env_file=".env",
#         extra="ignore"
#     )


# settings = Settings()

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ---------------------------------------------------------
    # LLM
    # ---------------------------------------------------------

    llm_model: str = "llama3.2:3b"

    ollama_url: str = "http://localhost:11434"

    ollama_keep_alive: str = "30m"

    # ---------------------------------------------------------
    # RETRIEVAL
    # ---------------------------------------------------------

    top_k: int = 3

    # Number of candidates retrieved internally before reranking.
    retrieval_k: int = 15

    # Minimum FAISS similarity accepted by the final pipeline.
    min_relevance: float = 0.20

    # ---------------------------------------------------------
    # LLM CONTEXT
    # ---------------------------------------------------------

    max_context_chars: int = 1800

    llm_num_predict: int = 40

    llm_num_ctx: int = 1024

    # Do NOT wait 180+ seconds for a CPU model.
    llm_timeout_seconds: int = 75

    # ---------------------------------------------------------
    # API
    # ---------------------------------------------------------

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()