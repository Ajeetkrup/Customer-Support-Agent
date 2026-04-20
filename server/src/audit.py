"""
audit.py — Structured audit log writer for the Customer Support Agent.

Writes one JSONL line per ticket to audit_log.jsonl.
Each entry models the full run via `AgentAuditEntry` (no truncated fields).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from src.agent.state import AgentState
from src.models.agent_log import AgentAuditEntry
from src.models.agent_log import ToolCallAuditRecord

AUDIT_LOG = Path(__file__).parent / "audit_log.jsonl"


def write_audit_entry(state: AgentState) -> None:
    """
    Persist a complete audit record for one ticket.
    Called at the end of each agent run.
    """
    trace_raw = state.get("tool_call_trace") or []
    trace: list[ToolCallAuditRecord] = []
    for r in trace_raw:
        if not isinstance(r, dict):
            continue
        res = r.get("result")
        if res is None and r.get("result_summary") is not None:
            res = r["result_summary"]
        trace.append(
            ToolCallAuditRecord(
                tool_name=r.get("tool_name") or "",
                args=r.get("args") or {},
                result=_json_safe_result(res),
                success=bool(r.get("success")),
                error=r.get("error"),
                timestamp=r.get("timestamp") or "",
            ),
        )

    entry = AgentAuditEntry(
        ticket_id=state.get("ticket_id") or "",
        customer_email=state.get("customer_email") or "",
        subject=state.get("subject") or "",
        body=state.get("body") or "",
        tier=state.get("tier"),
        source=state.get("source") or "",
        created_at=state.get("created_at") or "",
        expected_action=state.get("expected_action"),
        processed_at=datetime.now(timezone.utc),
        customer=state.get("customer"),
        order=state.get("order"),
        product=state.get("product"),
        kb_results=state.get("kb_results") or "",
        threat_signals=list(state.get("threat_signals") or []),
        threat_hint=state.get("threat_hint") or "",
        threatening_language_suspected=bool(state.get("threatening_language_suspected", False)),
        needs_intent_clarification=bool(state.get("needs_intent_clarification", False)),
        ambiguity_reason=state.get("ambiguity_reason") or "",
        urgency=state.get("urgency") or "",
        category=state.get("category") or "",
        confidence=float(state.get("confidence") or 0.0),
        can_resolve_autonomously=bool(state.get("can_resolve_autonomously", False)),
        fraud_suspected=bool(state.get("fraud_suspected", False)),
        route=state.get("route") or "",
        resolution_action=state.get("resolution_action") or "",
        resolution_reason=state.get("resolution_reason") or "",
        final_reply=state.get("final_reply") or "",
        escalation_summary=state.get("escalation_summary") or "",
        tool_call_trace=trace,
        error_log=list(state.get("error_log") or []),
    )

    with open(AUDIT_LOG, "a", encoding="utf-8") as f:
        f.write(entry.model_dump_json() + "\n")


def _json_safe_result(value: object) -> object:
    """Ensure tool results round-trip to JSON without truncating nested structures."""
    if value is None:
        return None
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, (dict, list, str, int, float, bool)):
        return value
    return str(value)


def print_audit_summary(state: AgentState) -> None:
    """Print a concise per-ticket summary to stdout."""
    tid = state.get("ticket_id", "???")
    route = state.get("route", "unknown")
    confidence = state.get("confidence", 0.0)
    action = state.get("resolution_action", (state.get("escalation_summary") or "")[:60])
    errors = state.get("error_log", [])
    fraud = state.get("fraud_suspected", False)
    threat = state.get("threatening_language_suspected", False)

    icon = {"resolve": "✅", "escalate": "🔺", "clarify": "❓"}.get(route, "⚙")
    fraud_tag = " 🚨FRAUD" if fraud else ""
    threat_tag = " ⚡THREAT" if threat else ""

    print(f"  {icon} [{tid}] route={route} | confidence={confidence:.2f} | "
          f"action={action}{fraud_tag}{threat_tag}")
    if errors:
        for e in errors:
            print(f"       ⚠  {e}")
