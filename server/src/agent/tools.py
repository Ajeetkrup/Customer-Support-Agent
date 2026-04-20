"""
tools.py — All 8 mocked tools for the Customer Support Agent.

Design principles:
  - Every tool loads data from the JSON/MD files (no external calls)
  - Tool failures are simulated deterministically + probabilistically
  - @with_retry wraps every tool: exponential backoff, max 3 attempts
  - Failed tools after retries are logged to stderr (no auxiliary JSONL files)
  - Pydantic models validate outputs before they reach the agent
  - issue_refund() is guarded: eligibility must have been checked first
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import random
import re
import time
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ValidationError, field_validator

from src.services.retrieve_from_db import get_Order_from_db, get_Customer_from_db, get_Product_from_db
from src.schemas.global_schema import CustomerSchema, OrderSchema, ProductSchema
from src.utils.logger import logger
from fastapi import HTTPException
from src.rag.retrieve import retrieve_from_qdrant

try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None  

# Track which orders have had eligibility checked (gate for issue_refund)
_eligibility_checked: set[str] = set()

# ── Pydantic output schemas ───────────────────────────────────────────────────

class OrderResult(BaseModel):
    order_id: str
    customer_id: str
    product_id: str
    quantity: int
    amount: float
    status: str
    order_date: str
    delivery_date: Optional[str]
    return_deadline: Optional[str]
    refund_status: Optional[str]
    notes: str

class RefundEligibilityResult(BaseModel):
    order_id: str
    eligible: bool
    reason: str
    refund_amount: Optional[float]


class ReturnEligibilityResult(BaseModel):
    """Return/refund window vs ticket filing date (last refundable date = order return_deadline)."""

    order_id: str
    eligible: bool
    reason: str
    ticket_created_at: str
    last_refundable_date: Optional[str]
    within_return_window: bool

class RefundResult(BaseModel):
    order_id: str
    amount: float
    status: str
    processed_at: str

class ReplyResult(BaseModel):
    ticket_id: str
    status: str
    sent_at: str

class EscalationResult(BaseModel):
    ticket_id: str
    priority: str
    routed_to: str
    created_at: str

class KBResult(BaseModel):
    query: str
    results: str


class ThreatIntentResult(BaseModel):
    """Heuristic scan for intimidating / legal / ultimatum language in ticket text."""

    threatening: bool
    signals: list[str]
    hint_for_agent: str


class AmbiguityCheckResult(BaseModel):
    """LLM output: whether we must ask for order ID / intended action before proceeding."""

    needs_clarification: bool
    reason: str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _dead_letter(tool_name: str, args: dict, error: str) -> None:
    logger.error("[dead-letter] tool=%s args=%s error=%s at=%s", tool_name, args, error, _now_iso())

# ── Retry decorator ───────────────────────────────────────────────────────────

def with_retry(max_attempts: int = 3, backoff: float = 1.5, exceptions=(TimeoutError, ValueError)):
    """Async retry with exponential backoff. Appends to dead-letter on exhaustion."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exc = None
            logger.debug("[TOOL:%s] Starting invocation...", func.__name__)
            for attempt in range(1, max_attempts + 1):
                try:
                    res = await func(*args, **kwargs)
                    logger.debug("[TOOL:%s] Execution complete on attempt %s.", func.__name__, attempt)
                    return res
                except exceptions as exc:
                    last_exc = exc
                    if attempt < max_attempts:
                        wait = backoff ** attempt
                        logger.warning(
                            "[TOOL:%s] Attempt %s failed (%s: %s). Retrying in %.1fs…",
                            func.__name__,
                            attempt,
                            type(exc).__name__,
                            exc,
                            wait,
                        )
                        await asyncio.sleep(wait)
                    else:
                        logger.error(
                            "[TOOL:%s] All %s attempts exhausted. Final error: %s: %s",
                            func.__name__,
                            max_attempts,
                            type(exc).__name__,
                            exc,
                        )
                        _dead_letter(func.__name__, {"args": str(args), "kwargs": str(kwargs)}, str(exc))
                        raise
        return wrapper
    return decorator

# ── Tool 1: get_order ─────────────────────────────────────────────────────────

@with_retry(max_attempts=3, backoff=1.5)
async def get_order(order_id: str) -> dict:
    """Return order details."""
    try:
        order = await get_Order_from_db(order_id)
        if order is None:
            raise ValueError(f"Order {order_id} not found in system")
        return OrderSchema.from_orm_order(order)
    except Exception as e:
        logger.error(f"Error getting order in tool: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting order in tool: {e}")

# ── Eligibility core (ticket created_at vs return_deadline) ─────────────────


def _parse_ticket_created_at(iso_s: str) -> datetime:
    """Parse ticket created_at (ISO-8601; date-only allowed)."""
    raw = (iso_s or "").strip()
    if not raw:
        raise ValueError("ticket_created_at is required")
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    if len(raw) == 10 and raw[4] == "-" and raw[7] == "-":
        raw = raw + "T00:00:00+00:00"
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_last_refundable_date(deadline_str: str) -> datetime.date:
    ds = deadline_str.strip()
    if len(ds) == 10 and ds[4] == "-" and ds[7] == "-":
        return datetime.strptime(ds, "%Y-%m-%d").date()
    dt = _parse_ticket_created_at(ds)
    return dt.astimezone(timezone.utc).date()


async def _compute_return_and_refund_eligibility(
    order_id: str, ticket_created_at: str,
) -> tuple[RefundEligibilityResult, ReturnEligibilityResult]:
    """
    Compare ticket filing date to order return_deadline (last refundable date).
    VIP / order-note exceptions apply when the calendar window has passed.
    """
    ticket_iso = (ticket_created_at or "").strip()
    try:
        order = await get_Order_from_db(order_id)

        
        if order is None:
            r = RefundEligibilityResult(
                order_id=order_id,
                eligible=False,
                reason=f"Order {order_id} does not exist in system.",
                refund_amount=None,
            )
            ret = ReturnEligibilityResult(
                order_id=order_id,
                eligible=False,
                reason=r.reason,
                ticket_created_at=ticket_iso or "(missing)",
                last_refundable_date=None,
                within_return_window=False,
            )
            return r, ret

        deadline_str = order.get("return_deadline")
        try:
            ticket_dt = _parse_ticket_created_at(ticket_iso)
        except (ValueError, TypeError) as exc:
            msg = f"Invalid ticket_created_at: {exc}"
            r = RefundEligibilityResult(
                order_id=order_id, eligible=False, reason=msg, refund_amount=None,
            )
            ret = ReturnEligibilityResult(
                order_id=order_id,
                eligible=False,
                reason=msg,
                ticket_created_at=ticket_iso or "(missing)",
                last_refundable_date=deadline_str,
                within_return_window=False,
            )
            return r, ret

        ticket_date = ticket_dt.date()
        within_window = True
        if deadline_str:
            last_ref_date = _parse_last_refundable_date(deadline_str)
            within_window = ticket_date <= last_ref_date

        ret_base = ReturnEligibilityResult(
            order_id=order_id,
            eligible=False,
            reason="",
            ticket_created_at=ticket_iso,
            last_refundable_date=deadline_str,
            within_return_window=within_window,
        )

        if order.get("refund_status") == "refunded":
            reason = "Refund has already been processed for this order."
            r = RefundEligibilityResult(
                order_id=order_id, eligible=False, reason=reason, refund_amount=None,
            )
            ret = ret_base.model_copy(update={"eligible": False, "reason": reason})
            return r, ret

        notes = order.get("notes", "")
        notes_l = notes.lower()

        if deadline_str and not within_window:
            if "pre-approved extended return exception" in notes or "VIP" in notes:
                ret_base = ret_base.model_copy(
                    update={
                        "reason": (
                            "Ticket filed after last refundable date but VIP / pre-approved "
                            "extended return exception applies."
                        ),
                    },
                )
            else:
                reason = (
                    f"Ticket filed on {ticket_date.isoformat()}, after last refundable date "
                    f"{deadline_str}. Return window is closed."
                )
                r = RefundEligibilityResult(
                    order_id=order_id, eligible=False, reason=reason, refund_amount=None,
                )
                ret = ret_base.model_copy(update={"eligible": False, "reason": reason})
                return r, ret

        if "registered online" in notes_l:
            reason = "Item has been registered online and is non-returnable per policy."
            r = RefundEligibilityResult(
                order_id=order_id, eligible=False, reason=reason, refund_amount=None,
            )
            ret = ret_base.model_copy(update={"eligible": False, "reason": reason})
            return r, ret

        if within_window:
            ok_reason = (
                f"Ticket filed on or before last refundable date ({deadline_str}). "
                "Within return window."
            )
        else:
            ok_reason = (
                "Outside standard calendar window but VIP / pre-approved exception applies; "
                "return/refund allowed."
            )

        r = RefundEligibilityResult(
            order_id=order_id,
            eligible=True,
            reason=f"{ok_reason} Meets refund criteria.",
            refund_amount=order["amount"],
        )
        ret = ret_base.model_copy(update={"eligible": True, "reason": ok_reason})
        return r, ret
    except Exception as e:
        logger.error(f"Error getting order in tool: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting order in tool: {e}")


# ── Tool 2a: check_return_eligibility ─────────────────────────────────────────

@with_retry(max_attempts=3, backoff=1.5)
async def check_return_eligibility(order_id: str, ticket_created_at: str) -> dict:
    """
    Check return eligibility using ticket filing time vs order return_deadline
    (last refundable date). Does not register the refund safety gate — use
    check_refund_eligibility before issue_refund.
    """
    _, ret = _compute_return_and_refund_eligibility(order_id, ticket_created_at)
    return ret


# ── Tool 2b: check_refund_eligibility ─────────────────────────────────────────

@with_retry(max_attempts=3, backoff=1.5)
async def check_refund_eligibility(order_id: str, ticket_created_at: str) -> dict:
    """
    Check refund eligibility (same window rules as check_return_eligibility).
    """
    refund_res, _ = _compute_return_and_refund_eligibility(order_id, ticket_created_at)

    if order_id in _orders:
        _eligibility_checked.add(order_id)

    return refund_res.model_dump()

# ── Tool 3: get_customer ──────────────────────────────────────────────────────

@with_retry(max_attempts=3, backoff=1.5)
async def get_customer(email: str) -> Optional[dict]:
    """Return customer profile. Returns None for unknown emails (not an error)."""
    try:
        customer = await get_Customer_from_db(email)
        if customer is None:
            return None
        return CustomerSchema.from_orm_customer(customer)
    except Exception as e:
        logger.error(f"Error getting customer in tool: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting customer in tool: {e}")

@with_retry(max_attempts=3, backoff=1.5)
async def list_orders_for_customer(customer_id: str) -> list[dict]:
    """Return all orders for a customer_id, newest order_date first."""
    rows = [
        OrderResult(**o)
        for o in _orders.values()
        if o.get("customer_id") == customer_id
    ]
    rows.sort(key=lambda x: x.get("order_date", ""), reverse=True)
    return rows


# ── Tool 4: issue_refund ──────────────────────────────────────────────────────

@with_retry(max_attempts=3, backoff=1.5)
async def issue_refund(order_id: str, amount: float) -> dict:
    """
    IRREVERSIBLE — issues a refund.
    Guards:
      - eligibility must have been checked first (via check_refund_eligibility)
      - amount must be > 0
      - order must exist
    """
    if order_id not in _eligibility_checked:
        raise ValueError(
            f"SAFETY GATE: issue_refund called for {order_id} without prior eligibility check. "
            "Call check_refund_eligibility first."
        )

    if amount <= 0:
        raise ValueError(f"Refund amount must be positive. Got: {amount}")

    try:
        order = await get_Order_from_db(order_id)

        if order is None:
            raise ValueError(f"Cannot refund non-existent order {order_id}")

        record = {
            "order_id": order_id,
            "amount": amount,
            "status": "processed",
            "processed_at": _now_iso(),
        }

        validated = RefundResult(**record)
        return validated.model_dump()
    except Exception as e:
        logger.error(f"Error getting order in tool: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting order in tool: {e}")

# ── Tool 5: get_product ───────────────────────────────────────────────────────

@with_retry(max_attempts=3, backoff=1.5)
async def get_product(product_id: str) -> Optional[dict]:
    """Return product metadata."""
    try:
        product = await get_Product_from_db(product_id)
        if product is None:
            return None
        return ProductSchema.from_orm_product(product)
    except Exception as e:
        logger.error(f"Error getting product in tool: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting product in tool: {e}")

# ── Tool 6: send_reply ────────────────────────────────────────────────────────

@with_retry(max_attempts=3, backoff=1.5)
async def send_reply(ticket_id: str, message: str) -> dict:
    """Send a reply to the customer (audit trail lives in agent `audit_log.jsonl`)."""
    if not ticket_id or not message:
        raise ValueError("ticket_id and message are required for send_reply")

    record = {
        "ticket_id": ticket_id,
        "message": message,
        "status": "sent",
        "sent_at": _now_iso(),
    }

    validated = ReplyResult(ticket_id=ticket_id, status="sent", sent_at=record["sent_at"])
    return validated

# ── Tool 7: search_knowledge_base ─────────────────────────────────────────────

@with_retry(max_attempts=3, backoff=1.5)
async def search_knowledge_base(query: str) -> dict:
    """
    Keyword-based semantic search over the knowledge base.
    Returns the most relevant section(s) as plain text.
    """
    try:
        result_text = await retrieve_from_qdrant(query)
        validated = KBResult(query=query, results=result_text)
        return validated.model_dump()
    except Exception as e:
        logger.error(f"Error searching knowledge base: {e}")
        raise HTTPException(status_code=500, detail=f"Error searching knowledge base: {e}")


# Dedicated Groq model for ambiguity classification (Llama — separate from agent’s GROQ_MODEL).
_DEFAULT_AMBIGUITY_GROQ_MODEL = "llama-3.3-70b-versatile"

_THREAT_REGEXES = (
    (re.compile(r"\b(lawyer|attorney|solicitor)\b", re.I), "legal_reference"),
    (re.compile(r"\b(lawsuit|sue|suing|litigation)\b", re.I), "legal_action"),
    (re.compile(r"\blegal action\b", re.I), "legal_action"),
    (re.compile(r"\b(court|fcc|bbb|better business)\b", re.I), "authority_threat"),
    (re.compile(r"\b(dispute\s+with\s+(my\s+)?bank|chargeback|reverse\s+the\s+charge)\b", re.I), "financial_threat"),
    (re.compile(r"\bor\s+else\b", re.I), "ultimatum"),
    (re.compile(r"\b(contacting\s+(my\s+)?lawyer|my\s+lawyer\s+will)\b", re.I), "legal_threat"),
)


def _analyze_threatening_intent(text: str) -> ThreatIntentResult:
    """Rule-based detector (mock); safe for deterministic tests."""
    hay = text.strip()
    seen: list[str] = []
    matched: set[str] = set()
    for pattern, sig in _THREAT_REGEXES:
        if pattern.search(hay):
            matched.add(sig)

    threatening = len(matched) > 0
    seen = sorted(matched)

    if threatening:
        hint = (
            "Threatening or high-pressure language was detected (see signals). "
            "Respond with calm, professional empathy; do not escalate tone; follow policy anyway."
        )
    else:
        hint = "No intimidating or ultimatum-style language flagged by automated scan."

    return ThreatIntentResult(threatening=threatening, signals=seen, hint_for_agent=hint)


@with_retry(max_attempts=3, backoff=1.5)
async def check_threatening_intent(subject: str, body: str) -> dict:
    """
    Scan subject + body for intimidating, legal-threat, chargeback, or ultimatum cues.
    Does not call external NLP APIs — rule-based mock for the agent pipeline.
    """
    await asyncio.sleep(0.02)
    merged = f"{subject or ''}\n{body or ''}"
    result = _analyze_threatening_intent(merged)
    return result.model_dump()


def _extract_json_object(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```\s*$", "", t)
    t = t.strip()
    if not t.startswith("{"):
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            t = m.group(0).strip()
    return t


_AMBIGUITY_SYSTEM = """You classify ecommerce support tickets for the Customer Support Agent.
You MUST respond with only a single JSON object with keys needs_clarification (boolean) and reason (short string).
No markdown fences."""


def _build_ambiguity_user_prompt(
    subject: str,
    body: str,
    *,
    has_order_id_in_ticket: bool,
    has_customer_profile: bool,
    has_resolved_order: bool,
) -> str:
    return f"""Decide if support must ask the customer for missing information before taking action.

Set needs_clarification to TRUE when ANY apply:
- The message is too vague to act (e.g. only "help", "thing isn't working", no clear request).
- It is unclear WHAT the customer wants (refund, return, cancel, tracking, warranty, wrong item, etc.).
- There is no order reference AND no resolved order AND the customer did not state a clear, actionable request.

Set needs_clarification to FALSE when:
- The customer clearly states what they want AND (there is an order ID in the text OR an order was loaded OR it is a clear general policy question that does not require an order to answer).

Known from tools:
- ORD-* pattern appears in ticket text: {has_order_id_in_ticket}
- Customer profile found for this email: {has_customer_profile}
- Order record was loaded: {has_resolved_order}

Subject: {subject}
Body: {body}
"""


@with_retry(
    max_attempts=3,
    backoff=1.5,
    exceptions=(TimeoutError, ValueError, ValidationError, RuntimeError),
)
async def check_ticket_ambiguity(
    subject: str,
    body: str,
    *,
    has_order_id_in_ticket: bool,
    has_customer_profile: bool,
    has_resolved_order: bool,
) -> dict:
    """
    Uses Groq **Llama** (`GROQ_AMBIGUITY_MODEL`, default llama-3.3-70b-versatile), not the main agent
    model, to judge whether we should ask for order ID / intended action. Plain JSON in the reply.
    """
    await asyncio.sleep(0.03)

    if ChatGroq is None:
        return default_ambiguity_fallback()

    api_key = (os.environ.get("GROQ_API_KEY") or "").strip()
    if not api_key:
        return default_ambiguity_fallback()

    model_id = (os.environ.get("GROQ_AMBIGUITY_MODEL") or "").strip() or _DEFAULT_AMBIGUITY_GROQ_MODEL
    llm = ChatGroq(
        model=model_id,
        temperature=0,
        max_tokens=512,
        groq_api_key=api_key,
    )

    human = _build_ambiguity_user_prompt(
        subject,
        body,
        has_order_id_in_ticket=has_order_id_in_ticket,
        has_customer_profile=has_customer_profile,
        has_resolved_order=has_resolved_order,
    )
    human += '\nRespond with JSON only: {"needs_clarification": <true|false>, "reason": "<one sentence>"}'

    raw = await llm.ainvoke(
        [
            ("system", _AMBIGUITY_SYSTEM),
            ("human", human),
        ]
    )
    content = getattr(raw, "content", raw)
    if isinstance(content, list):
        content = "".join(
            str(b.get("text", "")) if isinstance(b, dict) else str(b)
            for b in content
        )
    blob = _extract_json_object(str(content))
    parsed = AmbiguityCheckResult.model_validate_json(blob)
    return parsed.model_dump()


def default_ambiguity_fallback() -> dict:
    """When the LLM tool fails — safer to ask for clarification."""
    return {
        "needs_clarification": True,
        "reason": "Ambiguity check failed; defaulting to requesting order ID and intended action.",
    }


# ── Tool 8: escalate ──────────────────────────────────────────────────────────

@with_retry(max_attempts=3, backoff=1.5)
async def escalate(ticket_id: str, summary: str, priority: str) -> dict:
    """Route ticket to a human agent with a structured summary."""

    valid_priorities = {"low", "medium", "high", "urgent"}
    if priority not in valid_priorities:
        raise ValueError(f"Invalid priority '{priority}'. Must be one of: {valid_priorities}")

    record = {
        "ticket_id": ticket_id,
        "summary": summary,
        "priority": priority,
        "routed_to": "human_support_team",
        "created_at": _now_iso(),
    }

    validated = EscalationResult(
        ticket_id=ticket_id,
        priority=priority,
        routed_to="human_support_team",
        created_at=record["created_at"],
    )
    return validated.model_dump()
