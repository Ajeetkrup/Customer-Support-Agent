"""
agent.py -- LangGraph StateGraph for the Customer Support Agent.

Graph nodes:
  gather_context -> triage -> [resolve | escalate | clarify] -> END

The agent uses Groq only (`qwen/qwen3-32b` by default) with retries on the same model.
All tool calls go through the tool functions in tools.py.
Every decision is recorded in the AgentState for full explainability.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import datetime, timezone
from typing import Optional, Literal
from pydantic import BaseModel, Field, ValidationError

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

from src.audit import print_audit_summary, write_audit_entry
from src.agent.prompts import (
    SYSTEM_PROMPT, TRIAGE_PROMPT, RESOLVE_PROMPT,
    ESCALATE_PROMPT, CLARIFY_PROMPT,
)
from src.agent.state import AgentState, ToolCallRecord
from src.agent.tools import (
    get_customer, get_order, get_product,
    list_orders_for_customer,
    check_return_eligibility,
    check_refund_eligibility, issue_refund,
    send_reply, search_knowledge_base, escalate,
    check_threatening_intent,
    check_ticket_ambiguity,
    default_ambiguity_fallback,
)
from src.utils.logger import logger

class TriageSchema(BaseModel):
    urgency: Literal["low", "medium", "high", "urgent"] = Field(description="Priority of the ticket")
    category: Literal["refund", "return", "cancellation", "exchange", "warranty", "inquiry", "ambiguous", "fraud"] = Field(description="Classification of the ticket")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")
    can_resolve_autonomously: bool = Field(description="Whether the agent can resolve this without human intervention")
    fraud_suspected: bool = Field(description="True if fraud or social engineering is suspected")
    route: Literal["resolve", "escalate", "clarify"] = Field(description="Next routing step")

class ResolveSchema(BaseModel):
    resolution_action: str = Field(
        description=(
            "Action taken, e.g. issue_refund, send_info, approve_exchange_or_refund "
            "(wrong colour/size/item — exchange or refund per customer/policy)"
        ),
    )
    resolution_reason: str = Field(description="Human-readable explanation of the resolution")
    should_issue_refund: bool = Field(description="Whether a refund should be explicitly issued")
    refund_amount: Optional[float] = Field(description="Amount to refund, if applicable", default=None)
    customer_reply: str = Field(description="The polite, professional final message to send to the customer")

class EscalateSchema(BaseModel):
    priority: Literal["low", "medium", "high", "urgent"] = Field(description="Priority of the escalation")
    escalation_summary: str = Field(description="A concise summary of why the ticket is being escalated")
    customer_reply: str = Field(description="Polite message informing the customer about the escalation to a human agent")

class ClarifySchema(BaseModel):
    customer_reply: str = Field(description="Message asking the customer for missing details")

load_dotenv()

# Groq only — override with GROQ_MODEL if needed
_DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"


def _groq_model_id() -> str:
    mid = (os.environ.get("GROQ_MODEL") or "").strip()
    return mid if mid else _DEFAULT_GROQ_MODEL


def _build_groq_llm(model: str) -> ChatGroq:
    return ChatGroq(
        model=model,
        temperature=0,
        max_tokens=4096,
        groq_api_key=os.environ["GROQ_API_KEY"],
    )

# ── Helpers ───────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe_tool_result(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, (dict, list, str, int, float, bool)):
        return value
    return str(value)


def _record_tool_call(
    state: AgentState,
    tool_name: str,
    args: dict,
    result: object,
    success: bool,
    error: str | None = None,
) -> None:
    """Append a ToolCallRecord to state['tool_call_trace']."""
    ticket_id = state.get("ticket_id", "UNKNOWN")
    logger.info("[%s] TOOL CALL: %s | args=%s", ticket_id, tool_name, args)
    if success:
        preview = str(result)[:100] + "..." if result is not None and len(str(result)) > 100 else str(result)
        logger.info("[%s] TOOL SUCCESS: %s | result=%s", ticket_id, tool_name, preview)
    else:
        logger.error("[%s] TOOL ERROR: %s | error=%s", ticket_id, tool_name, error)

    trace = state.get("tool_call_trace") or []
    record: ToolCallRecord = {
        "tool_name": tool_name,
        "args": args,
        "result": _json_safe_tool_result(result),
        "success": success,
        "error": error,
        "timestamp": _now_iso(),
    }
    trace.append(record)
    state["tool_call_trace"] = trace


def _log_error(state: AgentState, msg: str) -> None:
    ticket_id = state.get("ticket_id", "UNKNOWN")
    logger.warning("[%s] ERROR LOG: %s", ticket_id, msg)
    errors = state.get("error_log") or []
    errors.append(msg)
    state["error_log"] = errors


def _extract_order_id(text: str) -> str | None:
    """Extract first ORD-XXXX pattern from ticket body."""
    match = re.search(r'ORD-\d+', text)
    return match.group() if match else None


def _kb_escalation_permitted(state: AgentState) -> tuple[bool, list[str]]:
    """
    Knowledge Base §7 — escalate only when at least one guideline applies.
    Returns (permitted, list of matching guideline labels for audit/logging).
    """
    reasons: list[str] = []
    conf = float(state.get("confidence") or 0.0)
    cat = (state.get("category") or "").strip()
    order = state.get("order")
    oa = float(order.get("amount") or 0.0) if order else 0.0

    if conf < 0.6:
        reasons.append("confidence below 0.6")

    if state.get("fraud_suspected"):
        reasons.append("fraud or social engineering suspected")

    if cat == "warranty":
        reasons.append("warranty claim (warranty team)")

    # Monetary refund / return path where order value exceeds $200 — needs human approval per KB & runtime policy.
    # Exchange-only / category exchange is excluded here (wrong-item exchange may be autonomous even when amount > $200).
    if order and oa > 200.0 and cat in ("refund", "return"):
        reasons.append("refund or return case with order amount over $200")

    body_l = (state.get("body") or "").lower()
    subj_l = (state.get("subject") or "").lower()
    blob = f"{subj_l} {body_l}"
    wants_replacement = any(
        p in blob for p in ("replacement", "replace it", "send a new", "send me a new")
    )
    mentions_damage = any(
        p in blob for p in ("damaged", "cracked", "broken", "defect", "arrived damaged", "water tank")
    )
    wants_refund_wording = any(p in blob for p in ("refund", "money back", "full refund"))
    if mentions_damage and wants_replacement and not wants_refund_wording:
        reasons.append("replacement (not refund) for damaged item")

    errs_combined = " ".join(state.get("error_log") or []).lower()
    ticket_text = f"{state.get('subject') or ''} {state.get('body') or ''}"
    cited_order = _extract_order_id(ticket_text)
    if cited_order and order is None:
        if "get_order" in errs_combined or "timed out" in errs_combined or "not found" in errs_combined:
            reasons.append("conflicting or unavailable order data vs customer claim")

    cust = state.get("customer") or {}
    tier_l = (cust.get("tier") or "").lower()
    # Borderline supervisor case (does not overlap with confidence < 0.6 rule above).
    if tier_l == "premium" and 0.6 <= conf < 0.65 and cat in ("return", "refund"):
        reasons.append("borderline premium case — supervisor judgment")

    return (len(reasons) > 0, reasons)


def _apply_kb_escalation_gate(state: AgentState) -> None:
    """If route is escalate but KB §7 does not allow it, reroute to resolve or clarify."""
    if state.get("route") != "escalate":
        return
    permitted, matched = _kb_escalation_permitted(state)
    if permitted:
        return
    _log_error(
        state,
        "Human escalation blocked: ticket does not match Knowledge Base §7 escalation guidelines "
        f"(had route=escalate). Routing autonomously instead.",
    )
    _c = float(state.get("confidence") or 0.0)
    if state.get("needs_intent_clarification") or (state.get("category") == "ambiguous" and _c < 0.5):
        state["route"] = "clarify"
        return
    if not state.get("order") and not state.get("customer"):
        state["route"] = "clarify"
        return
    state["route"] = "resolve"
    state["can_resolve_autonomously"] = True


def _format_threat_context(state: AgentState) -> str:
    """Inject into LLM prompts — flag threats but require polite replies."""
    if state.get("threatening_language_suspected"):
        signals = state.get("threat_signals") or []
        hint = state.get("threat_hint") or ""
        return (
            f"**Intent scan:** possibly threatening / high-pressure language detected "
            f"(signals: {signals}). "
            f"Respond with calm professionalism only; acknowledge concern without escalating tone.\n"
            f"{hint}"
        )
    return "**Intent scan:** no threatening-language patterns flagged."


def _infer_order_id_from_customer_orders(
    orders: list[dict], subject: str, body: str,
) -> str | None:
    """
    When the ticket omits ORD-*, pick an order after lookup by email → customer_id.
    Prefer open/cancellable statuses when the customer asks to cancel a recent order.
    Otherwise use the most recent order by order_date.
    """
    if not orders:
        return None
    if len(orders) == 1:
        return orders[0].get("order_id")

    full = f"{subject} {body}".lower()
    cancelish = any(
        k in full
        for k in (
            "cancel",
            "cancellation",
            "just placed",
            "before it ships",
            "before it ship",
            "hasn't shipped",
            "has not shipped",
            "stop my order",
        )
    )
    if cancelish:
        for status in ("processing", "pending", "packed", "shipped"):
            for o in orders:
                if o.get("status") == status:
                    return o.get("order_id")
    # Most recent purchase
    return orders[0].get("order_id")


def _is_rate_limit_error(err_str: str) -> bool:
    lower = err_str.lower()
    return (
        "429" in err_str
        or "rate_limit" in lower
        or "too many requests" in lower
        or "capacity" in lower and "exceed" in lower
    )


def _max_llm_retries() -> int:
    raw = os.environ.get("GROQ_LLM_MAX_RETRIES", "5").strip()
    try:
        n = int(raw)
    except ValueError:
        n = 5
    return max(1, min(n, 25))


def _json_only_suffix(schema: type[BaseModel]) -> str:
    keys = ", ".join(f'"{k}"' for k in schema.model_fields)
    return (
        f"\n\nReturn ONLY a single JSON object with these keys: {keys}. "
        "Use double quotes for keys and strings. No markdown fences, no other text."
    )


def _append_schema_json_instruction(messages: list, schema: type[BaseModel]) -> list:
    """Augment the last human message so Qwen returns parseable JSON (avoids Groq tool_use)."""
    if not messages:
        return messages
    suffix = _json_only_suffix(schema)
    out: list = []
    for i, item in enumerate(messages):
        if i == len(messages) - 1 and item[0] == "human":
            out.append((item[0], item[1] + suffix))
        else:
            out.append(item)
    return out


def _message_content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts) if parts else str(content)
    return str(content)


def _strip_markdown_json_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```\s*$", "", t)
    return t.strip()


def _extract_json_object(text: str) -> str:
    """Best-effort: fenced JSON or first {...} block."""
    t = _strip_markdown_json_fence(text)
    if t.startswith("{"):
        return t
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        return m.group(0).strip()
    return t


def _parse_llm_json(text: str, schema: type[BaseModel]) -> BaseModel:
    blob = _extract_json_object(text)
    return schema.model_validate_json(blob)


async def _llm_invoke(
    messages: list,
    schema: type[BaseModel] | None = None,
) -> object:
    """
    Invoke Groq without LangChain structured tool-calling (Groq+Qwen often returns
    tool_use_failed). Plain completion + JSON parsed with Pydantic instead.
    Retries on API errors and on invalid JSON / schema mismatch.
    """
    if not (os.environ.get("GROQ_API_KEY") or "").strip():
        raise RuntimeError("GROQ_API_KEY is not set.")

    model_id = _groq_model_id()
    max_retries = _max_llm_retries()
    llm = _build_groq_llm(model_id)
    to_send = _append_schema_json_instruction(messages, schema) if schema else messages

    logger.info("[LLM] Groq `%s` JSON mode (up to %s attempts)...", model_id, max_retries)
    last_exc: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.debug("[LLM] attempt %s/%s...", attempt, max_retries)
            raw = await llm.ainvoke(to_send)
            text = _message_content_to_text(raw.content)
            if schema:
                parsed = _parse_llm_json(text, schema)
                logger.info("[LLM] Success (%s)", model_id)
                return parsed
            logger.info("[LLM] Success (%s)", model_id)
            return raw
        except Exception as exc:
            last_exc = exc
            err_str = str(exc)
            kind = "parse/schema" if isinstance(exc, ValidationError) else "api/other"
            logger.warning("[LLM] failed (%s): %s", kind, err_str[:220])
            if attempt >= max_retries:
                break
            if _is_rate_limit_error(err_str):
                wait_match = re.search(r"try again in (\d+(?:\.\d+)?)s", err_str)
                wait_secs = float(wait_match.group(1)) + 2.0 if wait_match else min(90.0, 5.0 * attempt)
                logger.info("[rate-limit] retry same model in %.1fs...", wait_secs)
            else:
                wait_secs = min(30.0, 1.5**attempt)
                logger.info("[retry] same model in %.1fs...", wait_secs)
            await asyncio.sleep(wait_secs)

    assert last_exc is not None
    raise last_exc


# ── Node 1: gather_context ───────────────────────────────────────────────────

async def gather_context(state: AgentState) -> AgentState:
    """
    Fetch all available context for the ticket.
    Always executes multiple tool calls:
      1. get_customer
      2. search_knowledge_base
      3. check_threatening_intent(subject, body) → sets threatening_language_suspected / threat_signals
      4. get_order / get_product as applicable
      5. check_ticket_ambiguity (LLM) → needs_intent_clarification when order/intent is unclear
    """
    ticket_id = state.get("ticket_id", "UNKNOWN")
    logger.info("[%s] ENTERING NODE: gather_context", ticket_id)

    email = state["customer_email"]
    body = state.get("body", "")
    subject = state.get("subject", "")
    full_text = f"{subject} {body}"

    # ── Tool call 1: get_customer ─────────────────────────────────────────
    customer = None
    try:
        customer = await get_customer(email)
        _record_tool_call(state, "get_customer", {"email": email}, customer, True)
        state["customer"] = customer
    except Exception as exc:
        _record_tool_call(state, "get_customer", {"email": email}, None, False, str(exc))
        _log_error(state, f"get_customer failed: {exc}")
        state["customer"] = None

    # ── Tool call 2: search_knowledge_base ────────────────────────────────
    kb_query = f"{subject} {body[:150]}"
    try:
        kb_result = await search_knowledge_base(kb_query)
        _record_tool_call(state, "search_knowledge_base", {"query": kb_query}, kb_result, True)
        state["kb_results"] = kb_result.get("results", "")
    except Exception as exc:
        _record_tool_call(state, "search_knowledge_base", {"query": kb_query}, None, False, str(exc))
        _log_error(state, f"search_knowledge_base failed: {exc}")
        state["kb_results"] = ""

    # ── Tool call 3: threatening / intimidation intent (subject + body) ────────
    try:
        ti = await check_threatening_intent(subject, body)
        _record_tool_call(state, "check_threatening_intent", {"subject": subject, "body": body[:300]}, ti, True)
        state["threatening_language_suspected"] = bool(ti.get("threatening"))
        state["threat_signals"] = ti.get("signals") or []
        state["threat_hint"] = ti.get("hint_for_agent") or ""
    except Exception as exc:
        _record_tool_call(state, "check_threatening_intent", {"subject": subject}, None, False, str(exc))
        _log_error(state, f"check_threatening_intent failed: {exc}")
        state["threatening_language_suspected"] = False
        state["threat_signals"] = []
        state["threat_hint"] = ""

    # ── Tool call 4: get_order (explicit ORD-* in ticket, else lookup by email) ───
    order_id = _extract_order_id(full_text)
    order = None

    if not order_id and customer and customer.get("customer_id"):
        cid = customer["customer_id"]
        try:
            cust_orders = await list_orders_for_customer(cid)
            _record_tool_call(
                state,
                "list_orders_for_customer",
                {"customer_id": cid},
                {"count": len(cust_orders), "order_ids": [o.get("order_id") for o in cust_orders]},
                True,
            )
            order_id = _infer_order_id_from_customer_orders(
                cust_orders, subject, body,
            )
        except Exception as exc:
            _record_tool_call(state, "list_orders_for_customer", {"customer_id": cid}, None, False, str(exc))
            _log_error(state, f"list_orders_for_customer failed: {exc}")

    if order_id:
        try:
            order = await get_order(order_id)
            _record_tool_call(state, "get_order", {"order_id": order_id}, order, True)
            state["order"] = order
        except TimeoutError as exc:
            _record_tool_call(state, "get_order", {"order_id": order_id}, None, False, str(exc))
            _log_error(state, f"get_order timed out for {order_id}: {exc}")
            state["order"] = None
        except ValueError as exc:
            _record_tool_call(state, "get_order", {"order_id": order_id}, None, False, str(exc))
            _log_error(state, f"get_order error for {order_id}: {exc}")
            state["order"] = None
    else:
        state["order"] = None

    # ── Tool call 5 (conditional): get_product ──────────────────────────────
    product = None
    if order and order.get("product_id"):
        try:
            product = await get_product(order["product_id"])
            _record_tool_call(state, "get_product", {"product_id": order["product_id"]}, product, True)
            state["product"] = product
        except Exception as exc:
            _record_tool_call(state, "get_product", {"product_id": order["product_id"]}, None, False, str(exc))
            _log_error(state, f"get_product failed: {exc}")
            state["product"] = None
    else:
        state["product"] = None

    ord_pattern_in_ticket = _extract_order_id(full_text)
    try:
        amb = await check_ticket_ambiguity(
            subject,
            body,
            has_order_id_in_ticket=bool(ord_pattern_in_ticket),
            has_customer_profile=customer is not None,
            has_resolved_order=state.get("order") is not None,
        )
        _record_tool_call(
            state,
            "check_ticket_ambiguity",
            {
                "subject": subject[:120],
                "body_preview": body[:160],
                "has_order_id_in_ticket": bool(ord_pattern_in_ticket),
                "has_customer_profile": customer is not None,
                "has_resolved_order": state.get("order") is not None,
            },
            amb,
            True,
        )
        state["needs_intent_clarification"] = bool(amb.get("needs_clarification"))
        state["ambiguity_reason"] = str(amb.get("reason") or "")
    except Exception as exc:
        fb = default_ambiguity_fallback()
        _record_tool_call(
            state,
            "check_ticket_ambiguity",
            {"subject": subject[:120]},
            fb,
            False,
            str(exc),
        )
        _log_error(state, f"check_ticket_ambiguity failed: {exc}")
        state["needs_intent_clarification"] = bool(fb.get("needs_clarification"))
        state["ambiguity_reason"] = str(fb.get("reason") or "")

    return state


# ── Node 2: triage ────────────────────────────────────────────────────────────

async def triage(state: AgentState) -> AgentState:
    """
    Use LLM to classify the ticket and decide routing.
    Sets: urgency, category, confidence, can_resolve_autonomously, fraud_suspected, route
    """
    ticket_id = state.get("ticket_id", "UNKNOWN")
    logger.info("[%s] ENTERING NODE: triage", ticket_id)

    if state.get("needs_intent_clarification"):
        state["urgency"] = "low"
        state["category"] = "ambiguous"
        state["confidence"] = 0.45
        state["can_resolve_autonomously"] = False
        state["fraud_suspected"] = False
        state["route"] = "clarify"
        logger.info(
            "[%s] Route preset: clarify (ambiguous — need order ID & intended action)",
            ticket_id,
        )
        return state

    prompt = TRIAGE_PROMPT.format(
        subject=state.get("subject", ""),
        body=state.get("body", ""),
        customer_email=state.get("customer_email", ""),
        customer=json.dumps(state.get("customer"), indent=2),
        order=json.dumps(state.get("order"), indent=2),
        product=json.dumps(state.get("product"), indent=2),
        kb_results=(state.get("kb_results") or "")[:1000],
        threat_context=_format_threat_context(state),
    )

    try:
        response = await _llm_invoke([
            ("system", SYSTEM_PROMPT),
            ("human", prompt),
        ], schema=TriageSchema)
        parsed = response.model_dump()

        state["urgency"] = parsed.get("urgency", "medium")
        state["category"] = parsed.get("category", "ambiguous")
        state["confidence"] = float(parsed.get("confidence", 0.5))
        state["can_resolve_autonomously"] = bool(parsed.get("can_resolve_autonomously", False))
        state["fraud_suspected"] = bool(parsed.get("fraud_suspected", False))
        state["route"] = parsed.get("route", "escalate")

        # Safety overrides
        if state["confidence"] < 0.6:
            state["route"] = "escalate"
            state["can_resolve_autonomously"] = False

        if state["fraud_suspected"]:
            state["route"] = "escalate"

        if state["category"] == "warranty":
            state["route"] = "escalate"
            state["can_resolve_autonomously"] = False

        # Wrong variant / exchange (e.g. TKT-011): autonomous resolve per Exchange Policy — not human queue by default
        if (
            state["category"] == "exchange"
            and state["route"] == "escalate"
            and not state["fraud_suspected"]
            and state["confidence"] >= 0.6
        ):
            state["route"] = "resolve"
            state["can_resolve_autonomously"] = True

        _apply_kb_escalation_gate(state)

    except Exception as exc:
        _log_error(state, f"triage LLM call failed: {exc}")
        state.setdefault("urgency", "medium")
        state.setdefault("category", "ambiguous")
        state.setdefault("confidence", 0.0)
        state.setdefault("can_resolve_autonomously", False)
        state.setdefault("fraud_suspected", False)
        state["route"] = "escalate"
        _apply_kb_escalation_gate(state)

    return state


# ── Node 3: resolve ───────────────────────────────────────────────────────────

async def resolve(state: AgentState) -> AgentState:
    """
    Autonomously resolve the ticket.
    Calls check_return_eligibility and check_refund_eligibility (ticket date vs return_deadline)
    before issue_refund when applicable.
    Ends with send_reply.
    """
    ticket_id = state.get("ticket_id", "UNKNOWN")
    logger.info("[%s] ENTERING NODE: resolve", ticket_id)

    order = state.get("order")
    ticket_id = state["ticket_id"]

    # ── Eligibility check (if refund/return category) ─────────────────────
    eligibility = None
    if order and state.get("category") in ("refund", "return", "exchange"):
        order_id = order["order_id"]
        ticket_created_at = (state.get("created_at") or "").strip()
        if not ticket_created_at:
            ticket_created_at = _now_iso()
            _log_error(state, "created_at missing on ticket; using server time for eligibility window.")
        try:
            ret_chk = await check_return_eligibility(order_id, ticket_created_at)
            _record_tool_call(
                state,
                "check_return_eligibility",
                {"order_id": order_id, "ticket_created_at": ticket_created_at},
                ret_chk,
                True,
            )
        except Exception as exc:
            _record_tool_call(
                state,
                "check_return_eligibility",
                {"order_id": order_id, "ticket_created_at": ticket_created_at},
                None,
                False,
                str(exc),
            )
            _log_error(state, f"check_return_eligibility failed: {exc}")
        try:
            eligibility = await check_refund_eligibility(order_id, ticket_created_at)
            _record_tool_call(
                state,
                "check_refund_eligibility",
                {"order_id": order_id, "ticket_created_at": ticket_created_at},
                eligibility,
                True,
            )
        except Exception as exc:
            _record_tool_call(
                state,
                "check_refund_eligibility",
                {"order_id": order_id, "ticket_created_at": ticket_created_at},
                None,
                False,
                str(exc),
            )
            _log_error(state, f"check_refund_eligibility failed: {exc}")
            eligibility = {
                "eligible": False,
                "reason": f"Eligibility check failed: {exc}",
                "refund_amount": None,
            }

    # ── LLM decides resolution action ─────────────────────────────────────
    customer = state.get("customer") or {}
    customer_name = customer.get("name", "Customer").split()[0]

    prompt = RESOLVE_PROMPT.format(
        ticket_id=ticket_id,
        subject=state.get("subject", ""),
        body=state.get("body", ""),
        customer_name=customer_name,
        customer_tier=customer.get("tier", "unknown"),
        order=json.dumps(order, indent=2),
        product=json.dumps(state.get("product"), indent=2),
        eligibility=json.dumps(eligibility, indent=2),
        kb_results=(state.get("kb_results") or "")[:800],
        category=state.get("category", ""),
        urgency=state.get("urgency", ""),
        confidence=state.get("confidence", 0.5),
        threat_context=_format_threat_context(state),
        order_amount=(
            f"{float(order.get('amount') or 0):.2f}"
            if order
            else "0.00"
        ),
    )

    resolution = {}
    try:
        response = await _llm_invoke([
            ("system", SYSTEM_PROMPT),
            ("human", prompt),
        ], schema=ResolveSchema)
        resolution = response.model_dump()
    except Exception as exc:
        _log_error(state, f"resolve LLM call failed: {exc}")
        resolution = {
            "resolution_action": "escalate_due_to_error",
            "resolution_reason": str(exc),
            "should_issue_refund": False,
            "customer_reply": (
                f"Hi {customer_name}, we're experiencing a technical issue processing your request. "
                "A specialist will be in touch shortly."
            ),
        }

    state["resolution_action"] = resolution.get("resolution_action", "unknown")
    state["resolution_reason"] = resolution.get("resolution_reason", "")

    # ── Policy: no autonomous refund when order amount > $200 (exchange path still OK) ─
    if order and resolution.get("should_issue_refund"):
        oa = float(order.get("amount") or 0)
        if oa > 200.0:
            resolution["should_issue_refund"] = False
            _log_error(
                state,
                f"Autonomous refund blocked: order amount ${oa:.2f} exceeds $200 cap (exchange still allowed per KB).",
            )
            state["resolution_reason"] = (
                (state["resolution_reason"] or "")
                + " Autonomous refund not permitted over $200; specialist approval required for monetary refund."
            ).strip()

    # ── Issue refund if LLM decides to and eligibility confirmed ──────────
    if resolution.get("should_issue_refund") and eligibility and eligibility.get("eligible"):
        refund_amount = resolution.get("refund_amount") or (eligibility.get("refund_amount") or 0)
        if refund_amount > 0:
            try:
                refund_result = await issue_refund(order["order_id"], refund_amount)
                _record_tool_call(
                    state, "issue_refund",
                    {"order_id": order["order_id"], "amount": refund_amount},
                    refund_result, True,
                )
            except Exception as exc:
                _record_tool_call(
                    state, "issue_refund",
                    {"order_id": order["order_id"], "amount": refund_amount},
                    None, False, str(exc),
                )
                _log_error(state, f"issue_refund failed: {exc}")

    # ── Send reply ─────────────────────────────────────────────────────────
    reply_text = resolution.get("customer_reply", "Thank you for contacting customer support.")
    state["final_reply"] = reply_text

    try:
        reply_result = await send_reply(ticket_id, reply_text)
        _record_tool_call(state, "send_reply", {"ticket_id": ticket_id}, reply_result, True)
    except Exception as exc:
        _record_tool_call(state, "send_reply", {"ticket_id": ticket_id}, None, False, str(exc))
        _log_error(state, f"send_reply failed: {exc}")

    return state


# ── Node 4: escalate_ticket ───────────────────────────────────────────────────

async def escalate_ticket(state: AgentState) -> AgentState:
    """
    Escalate the ticket to a human agent with a structured summary.
    Calls escalate() + send_reply().
    """
    ticket_id = state.get("ticket_id", "UNKNOWN")
    logger.info("[%s] ENTERING NODE: escalate_ticket", ticket_id)

    permitted, _ = _kb_escalation_permitted(state)
    if not permitted:
        _log_error(
            state,
            "escalate_ticket: blocked by Knowledge Base §7 — handling via resolve instead of human queue.",
        )
        return await resolve(state)

    ticket_id = state["ticket_id"]
    customer = state.get("customer") or {}
    customer_name = customer.get("name", "Customer").split()[0]

    escalation_reasons = []
    if state.get("fraud_suspected"):
        escalation_reasons.append("Fraud/social engineering suspected")
    if state.get("confidence", 1.0) < 0.6:
        escalation_reasons.append(f"Low confidence ({state.get('confidence', 0):.2f})")
    if state.get("category") == "warranty":
        escalation_reasons.append("Warranty claim -- requires warranty team")
    errors = state.get("error_log", [])
    if errors:
        escalation_reasons.append(f"Tool errors: {'; '.join(errors[-2:])}")
    if not escalation_reasons:
        escalation_reasons.append("Cannot resolve autonomously -- policy requires human review")

    escalation_reason = "; ".join(escalation_reasons)

    prompt = ESCALATE_PROMPT.format(
        ticket_id=ticket_id,
        subject=state.get("subject", ""),
        body=state.get("body", ""),
        customer_name=customer_name,
        customer_tier=customer.get("tier", "unknown"),
        customer_email=state.get("customer_email", ""),
        order=json.dumps(state.get("order"), indent=2),
        product=json.dumps(state.get("product"), indent=2),
        kb_results=(state.get("kb_results") or "")[:600],
        category=state.get("category", "unknown"),
        urgency=state.get("urgency", "medium"),
        confidence=state.get("confidence", 0.0),
        fraud_suspected=state.get("fraud_suspected", False),
        escalation_reason=escalation_reason,
    )

    escalation_data = {}
    try:
        response = await _llm_invoke([
            ("system", SYSTEM_PROMPT),
            ("human", prompt),
        ], schema=EscalateSchema)
        escalation_data = response.model_dump()
    except Exception as exc:
        _log_error(state, f"escalate LLM call failed: {exc}")
        escalation_data = {
            "priority": state.get("urgency", "medium"),
            "escalation_summary": f"Ticket {ticket_id} needs human review. Reason: {escalation_reason}",
            "customer_reply": (
                f"Hi {customer_name}, your case is being reviewed by a specialist. "
                "We'll be in touch shortly."
            ),
        }

    priority = escalation_data.get("priority", state.get("urgency", "medium"))
    summary = escalation_data.get("escalation_summary", "")
    reply_text = escalation_data.get("customer_reply", "")

    state["escalation_summary"] = summary
    state["final_reply"] = reply_text

    # ── Call escalate() tool ──────────────────────────────────────────────
    try:
        esc_result = await escalate(ticket_id, summary, priority)
        _record_tool_call(state, "escalate", {"ticket_id": ticket_id, "priority": priority}, esc_result, True)
    except Exception as exc:
        _record_tool_call(state, "escalate", {"ticket_id": ticket_id}, None, False, str(exc))
        _log_error(state, f"escalate tool failed: {exc}")

    # ── Send customer reply ───────────────────────────────────────────────
    try:
        reply_result = await send_reply(ticket_id, reply_text)
        _record_tool_call(state, "send_reply", {"ticket_id": ticket_id}, reply_result, True)
    except Exception as exc:
        _record_tool_call(state, "send_reply", {"ticket_id": ticket_id}, None, False, str(exc))
        _log_error(state, f"send_reply (escalate) failed: {exc}")

    return state


# ── Node 5: clarify ───────────────────────────────────────────────────────────

async def clarify(state: AgentState) -> AgentState:
    """
    Request more information from the customer for ambiguous tickets.
    Still calls send_reply() (tool call #3+).
    """
    ticket_id = state.get("ticket_id", "UNKNOWN")
    logger.info("[%s] ENTERING NODE: clarify", ticket_id)

    ticket_id = state["ticket_id"]
    customer = state.get("customer") or {}
    customer_name = customer.get("name", "Customer").split()[0]

    if state.get("needs_intent_clarification"):
        first = (customer.get("name") or "").strip().split()
        greet_name = first[0] if first else ""
        opening = f"Hi {greet_name}, " if greet_name else "Hello, "
        reply_text = (
            f"{opening}thanks for contacting us. To help you, please reply with:\n\n"
            "1. **Your order ID** if you have it (for example ORD-1234).\n"
            "2. **What you want us to do** — such as a **refund**, **return**, **cancellation**, "
            "**order tracking / shipping**, **warranty** help, or another **specific request**.\n"
            "3. A **short description** of the problem (which product and what went wrong).\n\n"
            "**Without an order reference and a clear intended action, we cannot process your request yet.**"
        )
        state["final_reply"] = reply_text
        state["resolution_action"] = "request_clarification"
        state["resolution_reason"] = (
            "Ambiguous ticket from gather_context — requested order ID and intended action before proceeding."
        )
        try:
            reply_result = await send_reply(ticket_id, reply_text)
            _record_tool_call(state, "send_reply", {"ticket_id": ticket_id}, reply_result, True)
        except Exception as exc:
            _record_tool_call(state, "send_reply", {"ticket_id": ticket_id}, None, False, str(exc))
            _log_error(state, f"send_reply (clarify preset) failed: {exc}")
        return state

    prompt = CLARIFY_PROMPT.format(
        ticket_id=ticket_id,
        subject=state.get("subject", ""),
        body=state.get("body", ""),
        customer_name=customer_name,
        customer_email=state.get("customer_email", ""),
        customer=json.dumps(customer, indent=2),
    )

    clarify_data = {}
    try:
        response = await _llm_invoke([
            ("system", SYSTEM_PROMPT),
            ("human", prompt),
        ], schema=ClarifySchema)
        clarify_data = response.model_dump()
    except Exception as exc:
        _log_error(state, f"clarify LLM call failed: {exc}")
        clarify_data = {
            "customer_reply": (
                f"Hi {customer_name}, thank you for reaching out. "
                "Could you please share your order number and describe the issue in more detail? "
                "This will help us resolve your request as quickly as possible."
            )
        }

    reply_text = clarify_data.get("customer_reply", "")
    state["final_reply"] = reply_text
    state["resolution_action"] = "request_clarification"
    state["resolution_reason"] = "Insufficient information to act -- requested clarification from customer."

    try:
        reply_result = await send_reply(ticket_id, reply_text)
        _record_tool_call(state, "send_reply", {"ticket_id": ticket_id}, reply_result, True)
    except Exception as exc:
        _record_tool_call(state, "send_reply", {"ticket_id": ticket_id}, None, False, str(exc))
        _log_error(state, f"send_reply (clarify) failed: {exc}")

    return state


# ── Node 6: finish ────────────────────────────────────────────────────────────

async def finish(state: AgentState) -> AgentState:
    """Write audit log and print per-ticket summary."""
    ticket_id = state.get("ticket_id", "UNKNOWN")
    logger.info("[%s] ENTERING NODE: finish", ticket_id)
    write_audit_entry(state)
    print_audit_summary(state)
    return state


# ── Router ────────────────────────────────────────────────────────────────────

def route_after_triage(state: AgentState) -> str:
    """Route to resolve, escalate, or clarify based on triage output."""
    return state.get("route", "escalate")


# ── Build Graph ───────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("gather_context", gather_context)
    graph.add_node("triage", triage)
    graph.add_node("resolve", resolve)
    graph.add_node("escalate_ticket", escalate_ticket)
    graph.add_node("clarify", clarify)
    graph.add_node("finish", finish)

    graph.set_entry_point("gather_context")
    graph.add_edge("gather_context", "triage")

    graph.add_conditional_edges(
        "triage",
        route_after_triage,
        {
            "resolve": "resolve",
            "escalate": "escalate_ticket",
            "clarify": "clarify",
        },
    )

    graph.add_edge("resolve", "finish")
    graph.add_edge("escalate_ticket", "finish")
    graph.add_edge("clarify", "finish")
    graph.add_edge("finish", END)

    return graph.compile()


# Module-level compiled graph (reused across concurrent ticket runs)
SUPPORT_AGENT = build_graph()
