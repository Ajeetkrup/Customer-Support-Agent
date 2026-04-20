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

async def get_Order_from_db(order_id: str) -> Order:
    try:
        async with AsyncSession(engine) as session:
            result = await session.execute(select(Order).where(Order.order_id == order_id))
            order = result.scalar_one_or_none()
            return order
    except Exception as e:
        logger.error(f"Error retrieving order from db: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving order from db: {e}")

async def get_Product_from_db(product_id: str) -> Product:
    try:
        async with AsyncSession(engine) as session:
            result = await session.execute(select(Product).where(Product.product_id == product_id))
            product = result.scalar_one_or_none()
            return product
    except Exception as e:
        logger.error(f"Error retrieving product from db: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving product from db: {e}")

async def get_Customer_from_db(email: str) -> Customer:
    try:
        async with AsyncSession(engine) as session:
            result = await session.execute(select(Customer).where(Customer.email == email))
            customer = result.scalar_one_or_none()
            return customer
    except Exception as e:
        logger.error(f"Error retrieving customer from db: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving customer from db: {e}")
