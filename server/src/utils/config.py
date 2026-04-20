from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    DATABASE_URL: str
    DATABASE_URL_SYNC: str
    GROQ_API_KEY: str
    GROQ_MODEL: str = "qwen/qwen3-32b"
    GROQ_LLM_MAX_RETRIES: int = 5
    GROQ_AMBIGUITY_MODEL: str = "llama-3.3-70b-versatile"
    AGENT_CONCURRENCY: int = 3
    QDRANT_CONNECTION_STRING: str
    QDRANT_API_KEY: str
    GOOGLE_API_KEY: str
    REDIS_URI: str

    FRONTEND_URL: str = "http://localhost:3000"

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()
