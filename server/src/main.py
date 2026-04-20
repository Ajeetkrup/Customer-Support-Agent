from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from src.utils.logger import logger
from src.router import api
from src.utils.config import get_settings

settings = get_settings()

app = FastAPI(title="Customer Support Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api.router, prefix="/api", tags=["API"])

@app.get("/health")
async def health_check(request: Request):
    logger.debug("This is a debug message 2")
    return {"status": "healthy"}
