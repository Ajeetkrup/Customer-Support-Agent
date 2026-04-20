from sqlalchemy import Boolean, Column, DateTime, Integer, Numeric, String, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from src.models.database import Base


class Product(Base):
    __tablename__ = "products"

    product_id = Column(String(64), primary_key=True, index=True)
    name = Column(String(512), nullable=False)
    category = Column(String(128), nullable=False, index=True)
    price = Column(Numeric(14, 2), nullable=False)
    warranty_months = Column(Integer, default=0, nullable=False)
    return_window_days = Column(Integer, default=0, nullable=False)
    returnable = Column(Boolean, default=True, nullable=False)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    orders = relationship("Order", back_populates="product")
