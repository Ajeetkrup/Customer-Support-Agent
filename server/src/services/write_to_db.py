from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, and_, func, desc
from sqlalchemy.orm import selectinload
from src.models.customer import Customer
from src.models.product import Product
from src.models.order import Order
from src.models.ticket import SupportTicket
from src.models.agent_log import AgentAuditLog
from typing import Optional, List
from fastapi import HTTPException
from src.utils.logger import logger
from datetime import datetime, timedelta
import time
import json
from pathlib import Path
from src.utils.database import engine

async def write_audit_entry(audit_log: AgentAuditLog) -> None:
    try:
        async with AsyncSession(engine) as session:
            session.add(audit_log)
            await session.commit()
    except Exception as e:
        logger.error(f"Error writing audit entry to db: {e}")
        raise HTTPException(status_code=500, detail=f"Error writing audit entry to db: {e}")

