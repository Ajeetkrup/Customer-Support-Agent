from fastapi import APIRouter, Depends, HTTPException, status, Response, Cookie, Request
from typing import Optional
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession
from src.utils.logger import logger
from src.models.customer import Customer
from src.models.product import Product
from src.models.order import Order
from src.models.ticket import SupportTicket
from src.models.agent_log import AgentAuditLog
from src.utils.database import get_db
from src.utils.config import get_settings
from src.schemas.global_schema import MessageResponse, RetrieveKBResponse
from src.services.seed_db import seed_into_db

router = APIRouter()

@router.post("/seed-db", response_model=MessageResponse, status_code=status.HTTP_201_CREATED)
async def seed_db(request: Request, db: AsyncSession = Depends(get_db)):
    try:
        await seed_into_db(db)
        return {"message": "Database seeded successfully."}
    except Exception as e:
        logger.error(f"Router: Database error during seeding - error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error occurred during seeding. Please try again later."
        )

@router.post("/retrieve-knowledge-base", response_model=RetrieveKBResponse, status_code=status.HTTP_200_OK)
async def retrieve_kb(query: str, db: AsyncSession = Depends(get_db)):
    from src.rag.retrieve import retrieve_from_qdrant
    try:
        result = await retrieve_from_qdrant(query)
        # logger.info(f"Router: Knowledge base retrieved successfully. Result: {result}")
        return {"message": "Knowledge base retrieved successfully.", "result": result}
    except Exception as e:
        logger.error(f"Router: Knowledge base error during retrieval - error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Knowledge base error occurred during retrieval. Please try again later."
        )

@router.get("/injest-knowledge-base", response_model=MessageResponse, status_code=status.HTTP_201_CREATED)
async def injest_kb(request: Request, db: AsyncSession = Depends(get_db)):
    from src.rag.injest import injest_knowledge_base
    try:
        await injest_knowledge_base()
        return {"message": "Knowledge base ingested successfully."}
    except Exception as e:
        logger.error(f"Router: Knowledge base error during ingestion - error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Knowledge base error occurred during ingestion. Please try again later."
        )

@router.get("/run-support-agent", response_model=MessageResponse, status_code=status.HTTP_200_OK)
async def run_support_agent(request: Request, db: AsyncSession = Depends(get_db)):
    from src.run import main
    try:
        await main()
        return {"message": "Support agent run successfully."}
    except Exception as e:
        logger.error(f"Router: Support agent error during run - error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Support agent error occurred during run. Please try again later."
        )