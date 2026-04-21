from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict


class MessageResponse(BaseModel):
    message: str


class RetrieveKBResponse(BaseModel):
    message: str
    result: str


class CustomerSchema(BaseModel):
    """Maps `customers` table / Customer ORM — keys align with `src.models.customer.Customer`."""

    model_config = ConfigDict(from_attributes=True)

    customer_id: str
    name: str
    email: str
    phone: Optional[str] = None
    tier: str
    member_since: Optional[str] = None
    total_orders: int
    total_spent: float
    address: Optional[dict[str, Any]] = None
    notes: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @classmethod
    def from_orm_customer(cls, row: Any) -> dict[str, Any]:
        """SQLAlchemy Customer row → JSON-safe dict for agent tools."""
        return cls.model_validate(row).model_dump(mode="json")


class OrderSchema(BaseModel):
    """Maps `orders` table / Order ORM — keys align with `src.models.order.Order`."""

    model_config = ConfigDict(from_attributes=True)

    order_id: str
    customer_id: str
    product_id: str
    quantity: int
    amount: float
    status: str
    order_date: Optional[str] = None
    delivery_date: Optional[str] = None
    return_deadline: Optional[str] = None
    refund_status: Optional[str] = None
    notes: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @classmethod
    def from_orm_order(cls, row: Any) -> dict[str, Any]:
        """SQLAlchemy Order row → JSON-safe dict for agent tools."""
        return cls.model_validate(row).model_dump(mode="json")


class ProductSchema(BaseModel):
    """Maps `products` table / Product ORM — keys align with `src.models.product.Product`."""

    model_config = ConfigDict(from_attributes=True)

    product_id: str
    name: str
    category: str
    price: float
    warranty_months: int
    return_window_days: int
    returnable: bool
    notes: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @classmethod
    def from_orm_product(cls, row: Any) -> dict[str, Any]:
        """SQLAlchemy Product row → JSON-safe dict for agent tools."""
        return cls.model_validate(row).model_dump(mode="json")
class AnswerResponse(BaseModel):
    answer: str
