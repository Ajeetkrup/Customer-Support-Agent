"""
run.py -- Async runner for the Customer Support Agent.

Processes all tickets CONCURRENTLY using asyncio.gather() with a semaphore
that limits burst to AGENT_CONCURRENCY tickets at once (default: 3) to
respect Groq's TPM rate limits while still being genuinely concurrent.

Usage:
    python -X utf8 run.py
    AGENT_CONCURRENCY=5 python -X utf8 run.py
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import sys
import time
from pathlib import Path

# Force UTF-8 stdout on Windows to handle Unicode print statements
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

from dotenv import load_dotenv

load_dotenv()

from src.utils.logger import logger

# Validate Groq API key early (agent uses Groq qwen/qwen3-32b only)
if not (os.environ.get("GROQ_API_KEY") or "").strip():
    logger.error("GROQ_API_KEY not found in environment or .env file.")
    logger.error("Example: GROQ_API_KEY=your_key_here")
    sys.exit(1)

from src.agent.agent import SUPPORT_AGENT
from src.agent.state import AgentState

BASE = Path(__file__).parent
TICKETS_FILE = BASE / "tickets.json"

# Semaphore concurrency (default 3 to respect provider rate limits)
_CONCURRENCY = int(os.environ.get("AGENT_CONCURRENCY", "3"))

# Output files -- cleared on each run
OUTPUT_FILES = [
    BASE / "audit_log.jsonl",
    BASE / "replies_sent.jsonl",
    BASE / "escalations.jsonl",
    BASE / "dead_letter.jsonl",
    BASE / "refunds_issued.jsonl",
]


def _clear_output_files() -> None:
    """Clear all output JSONL files before a fresh run."""
    logger.info("[SYSTEM] Starting to clear output files...")
    for f in OUTPUT_FILES:
        if f.exists():
            logger.info("[SYSTEM] Deleting %s", f.name)
            f.unlink()
    logger.info("[SYSTEM] Cleared previous run output files.")


def _load_tickets() -> list[dict]:
    logger.info("[SYSTEM] Loading tickets from tickets.json...")
    tickets_data = json.loads(TICKETS_FILE.read_text(encoding="utf-8"))
    logger.info("[SYSTEM] Successfully loaded %s tickets.", len(tickets_data))
    return tickets_data


async def process_ticket(ticket: dict, sem: asyncio.Semaphore) -> AgentState:
    """
    Run the LangGraph agent for a single ticket.
    Semaphore limits concurrent LLM throughput. Never raises -- all errors logged.
    """
    logger.info("[%s] Waiting for concurrency semaphore...", ticket["ticket_id"])
    async with sem:
        logger.info("[%s] Semaphore acquired. Initializing state...", ticket["ticket_id"])
        initial_state: AgentState = {
            "ticket_id": ticket["ticket_id"],
            "customer_email": ticket["customer_email"],
            "subject": ticket["subject"],
            "body": ticket["body"],
            "tier": ticket.get("tier", 1),
            "source": ticket.get("source", "unknown"),
            "created_at": ticket.get("created_at", ""),
            "expected_action": ticket.get("expected_action", ""),
            # Initialise mutable fields
            "tool_call_trace": [],
            "error_log": [],
        }
        logger.info("[%s] State initialized.", ticket["ticket_id"])

        try:
            logger.info("[%s] STARTING TICKET PROCESSING", ticket["ticket_id"])
            final_state = initial_state
            async for step in SUPPORT_AGENT.astream(initial_state):
                for node_name, node_state in step.items():
                    logger.info("[%s] NODE COMPLETED: %s", ticket["ticket_id"], node_name)
                    final_state = node_state
            logger.info("[%s] FINISHED TICKET PROCESSING", ticket["ticket_id"])
            return final_state
        except Exception as exc:
            # Ticket-level safety net -- this ticket fails but others continue
            logger.exception(
                "[CRASH] [%s] Uncaught exception during process_ticket",
                ticket["ticket_id"],
            )
            initial_state["route"] = "escalate"
            initial_state["error_log"] = [f"Agent crash: {exc}"]
            initial_state["resolution_action"] = "agent_error"
            initial_state["resolution_reason"] = str(exc)
            initial_state["confidence"] = 0.0
            initial_state["urgency"] = "urgent"
            initial_state["category"] = "ambiguous"
            return initial_state


def _print_run_summary(results: list[AgentState], elapsed: float) -> None:
    total = len(results)
    resolved  = sum(1 for r in results if r.get("route") == "resolve")
    escalated = sum(1 for r in results if r.get("route") == "escalate")
    clarified = sum(1 for r in results if r.get("route") == "clarify")
    fraud_flagged = sum(1 for r in results if r.get("fraud_suspected"))
    errors = sum(1 for r in results if r.get("error_log"))
    avg = (elapsed / total) if total else 0.0

    logger.info("\n%s", "=" * 60)
    logger.info("  CUSTOMER SUPPORT AGENT -- RUN SUMMARY")
    logger.info("%s", "=" * 60)
    logger.info("  Total tickets processed : %s", total)
    logger.info("  [OK]  Resolved autonomously : %s", resolved)
    logger.info("  [UP]  Escalated to human    : %s", escalated)
    logger.info("  [?]   Clarification sent    : %s", clarified)
    logger.info("  [!!]  Fraud flagged         : %s", fraud_flagged)
    logger.info("  [W]   Tickets with errors   : %s", errors)
    logger.info("  [T]   Total time            : %.1fs (%.1fs avg)", elapsed, avg)
    logger.info("%s", "=" * 60)
    logger.info("  Output files written to: %s", BASE)
    logger.info("  - audit_log.jsonl       (full decision trace for every ticket)")
    logger.info("  - replies_sent.jsonl    (all customer replies)")
    logger.info("  - escalations.jsonl     (all escalations with summaries)")
    logger.info("  - refunds_issued.jsonl  (all refunds -- IRREVERSIBLE log)")
    logger.info("  - dead_letter.jsonl     (tool failures after exhausted retries)")


async def main() -> None:
    logger.info("%s", "=" * 52)
    logger.info("   Customer Support Agent — resolution run")
    logger.info("%s\n", "=" * 52)

    logger.info("[SYSTEM] Entering main(). Initializing semaphore...")
    sem = asyncio.Semaphore(_CONCURRENCY)

    _clear_output_files()
    tickets = _load_tickets()
    logger.info("[IN] Loaded %s tickets from %s", len(tickets), TICKETS_FILE.name)
    logger.info("[>>] Processing tickets CONCURRENTLY (max %s at once)...", _CONCURRENCY)
    logger.info("%s", "-" * 60)

    start = time.perf_counter()

    logger.info("[SYSTEM] Spawning tasks using asyncio.gather()...")
    # CONCURRENT PROCESSING -- asyncio.gather fans out all tickets, semaphore throttles burst
    results = await asyncio.gather(
        *[process_ticket(t, sem) for t in tickets],
        return_exceptions=True,  # Changed to True to prevent one failed task from taking down the cluster, although process_ticket handles it
    )
    logger.info("[SYSTEM] asyncio.gather() complete.")

    elapsed = time.perf_counter() - start

    logger.info("%s", "-" * 60)
    _print_run_summary(list(results), elapsed)


if __name__ == "__main__":
    asyncio.run(main())
