from sqlalchemy import Column, DateTime, ForeignKey, Integer, Numeric, String, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from src.models.database import Base


class Order(Base):
    __tablename__ = "orders"

    order_id = Column(String(64), primary_key=True, index=True)
    customer_id = Column(
        String(64),
        ForeignKey("customers.customer_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    product_id = Column(
        String(64),
        ForeignKey("products.product_id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    quantity = Column(Integer, nullable=False)
    amount = Column(Numeric(14, 2), nullable=False)
    status = Column(String(64), nullable=False, index=True)
    order_date = Column(String(32), nullable=True)
    delivery_date = Column(String(32), nullable=True)
    return_deadline = Column(String(32), nullable=True)
    refund_status = Column(String(64), nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    customer = relationship("Customer", back_populates="orders")
    product = relationship("Product", back_populates="orders")
