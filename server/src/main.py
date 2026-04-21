from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank 
from fastapi.middleware.cors import CORSMiddleware
from src.utils.logger import logger
from src.router import api
from src.utils.config import get_settings

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.reranker = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n=3
    )

    app.state.reranker.postprocess_nodes([], query_str="test")
    yield

app = FastAPI(title="Customer Support Agent API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.FRONTEND_URL,
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api.router, prefix="/api", tags=["API"])

@app.get("/health")
async def health_check(request: Request):
    logger.debug("This is a debug message 2")
    return {"status": "healthy"}
