"""
Agent audit trail: Pydantic models for JSONL persistence and ORM for optional DB storage.

The `entry` JSONB on `AgentAuditLog` stores a full `AgentAuditEntry` document (nothing truncated).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from src.models.database import Base


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ToolCallAuditRecord(BaseModel):
    """One tool invocation in the audit trace (full args + full result payload)."""

    tool_name: str = ""
    args: dict[str, Any] = Field(default_factory=dict)
    result: Optional[Any] = None
    success: bool = False
    error: Optional[str] = None
    timestamp: str = ""


class AgentAuditEntry(BaseModel):
    """Complete record for one agent run — mirrors `audit.write_audit_entry` fields."""

    ticket_id: str = ""
    customer_email: str = ""
    subject: str = ""
    body: str = ""
    tier: Optional[int] = None
    source: str = ""
    created_at: str = ""
    expected_action: Optional[str] = None
    processed_at: datetime = Field(default_factory=_utc_now)

    customer: Optional[dict[str, Any]] = None
    order: Optional[dict[str, Any]] = None
    product: Optional[dict[str, Any]] = None
    kb_results: str = ""

    threat_signals: list[str] = Field(default_factory=list)
    threat_hint: str = ""
    threatening_language_suspected: bool = False
    needs_intent_clarification: bool = False
    ambiguity_reason: str = ""

    urgency: str = ""
    category: str = ""
    confidence: float = 0.0
    can_resolve_autonomously: bool = False
    fraud_suspected: bool = False

    route: str = ""
    resolution_action: str = ""
    resolution_reason: str = ""
    final_reply: str = ""
    escalation_summary: str = ""

    tool_call_trace: list[ToolCallAuditRecord] = Field(default_factory=list)
    error_log: list[str] = Field(default_factory=list)


class AgentAuditLog(Base):
    """
    Persisted agent run: `entry` is the entire `AgentAuditEntry` serialized to JSON (no truncation).
    """

    __tablename__ = "agent_audit_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    ticket_id = Column(
        String(64),
        ForeignKey("support_tickets.ticket_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    entry = Column(JSONB, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    ticket = relationship("SupportTicket", back_populates="audit_logs")
