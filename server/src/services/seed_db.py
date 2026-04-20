from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, and_, func, desc
from sqlalchemy.orm import selectinload
from src.models.customer import Customer
from src.models.product import Product
from src.models.order import Order
from src.models.ticket import SupportTicket
from src.models.agent_log import AgentAuditLog
from src.schemas.global_schema import MessageResponse
from typing import Optional, List
from fastapi import HTTPException
from src.utils.logger import logger
from datetime import datetime, timedelta
import time
import json
from pathlib import Path

async def load_json_file(file_name: str) -> list[dict]:
    file_path = Path(__file__).parent.parent / file_name
    return json.loads(file_path.read_text(encoding="utf-8"))

async def seed_customers(db: AsyncSession, customers: list[dict]) -> None:
    for customer in customers:
        try:
            customer = Customer(
                customer_id=customer["customer_id"],
                name=customer["name"],
                email=customer["email"],
                phone=customer["phone"],
                tier = customer["tier"],
                member_since=customer["member_since"],
                total_orders=customer["total_orders"],
                total_spent=customer["total_spent"],
                address=customer["address"],
                notes=customer["notes"],
            )
            db.add(customer)
            logger.info(f"Service: Customer seeded successfully - customer_id: {customer.customer_id}")
        except Exception as e:
            logger.error(f"Service: Error seeding customer - customer_id: {customer['customer_id']}, error: {str(e)}", exc_info=True)
            continue

async def seed_products(db: AsyncSession, products: list[dict]) -> None:
    for product in products:
        try:
            product = Product(
                 product_id=product["product_id"],
                 name = product["name"],
                 category = product["category"],
                 price = product["price"],
                 warranty_months = product["warranty_months"],
                 return_window_days = product["return_window_days"],
                 returnable = product["returnable"],
                 notes = product["notes"],
            )
            db.add(product)
            logger.info(f"Service: Product seeded successfully - product_id: {product.product_id}")
        except Exception as e:
            logger.error(f"Service: Error seeding product - product_id: {product['product_id']}, error: {str(e)}", exc_info=True)
            continue

async def seed_orders(db: AsyncSession, orders: list[dict]) -> None:
    for order in orders:
        try:
            order = Order(
                order_id=order["order_id"],
                customer_id=order["customer_id"],
                product_id=order["product_id"],
                quantity=order["quantity"],
                amount=order["amount"],
                status=order["status"],
                order_date=order["order_date"],
                delivery_date=order["delivery_date"],
                return_deadline=order["return_deadline"],
                refund_status=order["refund_status"],
                notes=order["notes"],
            )
            
            db.add(order)
            logger.info(f"Service: Order seeded successfully - order_id: {order.order_id}")
        except Exception as e:
            logger.error(f"Service: Error seeding order - order_id: {order['order_id']}, error: {str(e)}", exc_info=True)
            continue
            
async def seed_tickets(db: AsyncSession, tickets: list[dict]) -> None:
    for ticket in tickets:
        try:
            ticket = SupportTicket(
                ticket_id = ticket["ticket_id"],
                customer_id = ticket["customer_id"],
                customer_email = ticket["customer_email"],
                subject = ticket["subject"],
                body = ticket["body"],
                source = ticket["source"],
                tier = ticket["tier"],
                created_at = ticket["created_at"],
            )
            db.add(ticket)
            logger.info(f"Service: Ticket seeded successfully - ticket_id: {ticket.ticket_id}")
        except Exception as e:
            logger.error(f"Service: Error seeding ticket - ticket_id: {ticket['ticket_id']}, error: {str(e)}", exc_info=True)
            continue

async def seed(db: AsyncSession) -> None:
    customers = await load_json_file("customers.json")
    products = await load_json_file("products.json")
    orders = await load_json_file("orders.json")
    tickets = await load_json_file("tickets.json")

    try:
        await seed_customers(db, customers)
        await db.commit()
        await seed_products(db, products)
        await db.commit()
        await seed_orders(db, orders)
        await db.commit()
        await seed_tickets(db, tickets)
        await db.commit()
    except Exception as e:
        logger.error(f"Service: Error seeding database - error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Database error occurred while seeding"
        )

async def seed_into_db(db: AsyncSession) -> MessageResponse:
    try:
        logger.info(f"Service: Seeding database")
        await seed(db)
        return MessageResponse(message="Database seeded successfully")
    except Exception as e:
        await db.rollback()
        logger.error(f"Service: Error seeding database - error: {str(e)}", exc_info=True)
        error_str = str(e).lower()
        if "duplicate" in error_str or "unique" in error_str or "already exists" in error_str:
            raise HTTPException(
                status_code=409,
                detail="Blog with this title already exists or constraint violation occurred"
            )
        raise HTTPException(
            status_code=500,
            detail="Database error occurred while creating blog"
        )
    except ValueError as e:
        await db.rollback()
        logger.error(f"Service: Invalid input data creating blog - user_id: {user_id}, title: '{data.title}', error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input data: {str(e)}"
        )