import redis
import hashlib
import orjson, zlib
from src.utils.config import get_settings

settings = get_settings()

r = redis.from_url(
    url = settings.REDIS_URI,
    decode_responses=False,  
)

CACHE_VERSION = "v1"

def norm(q: str) -> str:
    return " ".join(q.strip().lower().split())

def h(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()

def k(prefix: str, *parts: str) -> str:
    return f"{CACHE_VERSION}:{prefix}:" + ":".join(parts)

def dump(obj) -> bytes:
    return zlib.compress(orjson.dumps(obj))

def load(b: bytes):
    return orjson.loads(zlib.decompress(b))
