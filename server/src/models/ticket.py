from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from src.models.database import Base


class SupportTicket(Base):
    __tablename__ = "support_tickets"

    ticket_id = Column(String(64), primary_key=True, index=True)
    customer_id = Column(
        String(64),
        ForeignKey("customers.customer_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    customer_email = Column(String(255), nullable=False, index=True)
    subject = Column(Text, nullable=False)
    body = Column(Text, nullable=False)
    source = Column(String(64), nullable=False)
    tier = Column(Integer, nullable=True)
    created_at = Column(String(64), nullable=True)
    created_row_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    customer = relationship("Customer", back_populates="tickets")
    audit_logs = relationship("AgentAuditLog", back_populates="ticket")
