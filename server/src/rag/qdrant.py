from qdrant_client import QdrantClient, AsyncQdrantClient
from src.utils.config import get_settings
from llama_index.storage.kvstore.redis import RedisKVStore

settings = get_settings()

qdrant_client = QdrantClient(
    url=settings.QDRANT_CONNECTION_STRING,
    api_key=settings.QDRANT_API_KEY,
    prefer_grpc=True,
    port=6334,
    timeout=60 # 60 seconds
)

aqdrant_client = AsyncQdrantClient(
    url=settings.QDRANT_CONNECTION_STRING,
    api_key=settings.QDRANT_API_KEY,
    prefer_grpc=True,
    port=6334,
    timeout=15 # 60 seconds
)

redis_kv = RedisKVStore(redis_uri=settings.REDIS_URI)

def get_qdrant_client():
    return qdrant_client