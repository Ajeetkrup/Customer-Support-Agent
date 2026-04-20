"""
state.py — LangGraph agent state definition for the Customer Support Agent.

Every field is optional by default so nodes can be added incrementally.
The tool_call_trace captures every tool invocation for full explainability.
"""

from __future__ import annotations
from typing import Any, Optional
from typing_extensions import TypedDict


class ToolCallRecord(TypedDict, total=False):
    """A single tool invocation record for the audit trail."""
    tool_name: str
    args: dict
    result: Any                 # full JSON-serializable payload (preferred)
    result_summary: str         # legacy string copy; deprecated
    success: bool
    error: Optional[str]
    timestamp: str              # ISO-8601


class AgentState(TypedDict, total=False):
    # ── Ticket Input ──────────────────────────────────────────────────────────
    ticket_id: str
    customer_email: str
    subject: str
    body: str
    tier: int                 # ticket-declared tier (not trusted for policy)
    source: str
    created_at: str
    expected_action: str      # benchmark / eval hint when present (optional)

    # ── Fetched Context ───────────────────────────────────────────────────────
    customer: Optional[dict]  # from get_customer()
    order: Optional[dict]     # from get_order()
    product: Optional[dict]   # from get_product()
    kb_results: str           # from search_knowledge_base()
    threatening_language_suspected: bool  # from check_threatening_intent(subject, body)
    threat_signals: list                # short labels from scan — optional audit context
    threat_hint: str                     # coaching line for polite tone in LLM prompts
    needs_intent_clarification: bool     # from check_ticket_ambiguity (LLM tool)
    ambiguity_reason: str               # short rationale from the ambiguity model

    # ── Triage Output ─────────────────────────────────────────────────────────
    urgency: str              # low | medium | high | urgent
    category: str             # refund | return | cancellation | exchange |
                              #   warranty | inquiry | ambiguous | fraud
    confidence: float         # 0.0 – 1.0
    can_resolve_autonomously: bool
    fraud_suspected: bool

    # ── Resolution ────────────────────────────────────────────────────────────
    resolution_action: str    # e.g. "issue_refund" | "send_info" | "cancel_order"
    resolution_reason: str    # human-readable explanation

    # ── Communication ─────────────────────────────────────────────────────────
    final_reply: str          # message sent to customer
    escalation_summary: str   # structured summary for human agent

    # ── Audit / Explainability ────────────────────────────────────────────────
    tool_call_trace: list      # list[ToolCallRecord]
    error_log: list            # list[str] — soft errors during processing

    # ── Routing ───────────────────────────────────────────────────────────────
    route: str                # "resolve" | "escalate" | "clarify"
