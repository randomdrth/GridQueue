"""Records every tool call, with the agent that made it, for the /chat tool_calls list.

One Recorder per request, passed as hooks to every Runner.run in the pipeline, so the
Planner, each SQL worker, the Verifier and the Writer all land in one ordered list.

If an `on_event` callback is given, the Recorder also reports progress as it happens (an agent
starting, a tool call finishing, a pipeline stage), which /chat/stream sends to the page live.
"""

import json
import time
from collections import defaultdict
from typing import Callable

from agents import RunHooks

MAX_RESULT_CHARS = 6_000  # per tool call in the response; the agent itself saw the full result


def summarize(name: str, args: dict, result: str) -> str:
    """One readable line for a tool call, for the live progress list and the trace.
    `result` should be the full, untruncated tool output."""
    try:
        data = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        data = None
    if isinstance(data, dict) and data.get("error"):
        return f"Error: {str(data['error'])[:140]}"
    if name == "run_sql":
        rows = data.get("row_count") if isinstance(data, dict) else None
        purpose = args.get("purpose") or "query"
        return f"{purpose} ({rows:,} row{'s' if rows != 1 else ''})" if isinstance(rows, int) else purpose
    if name == "describe_table":
        return f"Read the columns of {args.get('table', 'a table')}"
    if name == "get_column_values":
        target = ".".join(x for x in (args.get("table"), args.get("column")) if x)
        return f"Looked up values in {target}" + (f" matching '{args['search']}'" if args.get("search") else "")
    if name == "get_data_gaps":
        return f"Checked known data gaps for '{args.get('topic', '')}'"
    if name == "list_tables":
        return "Listed the tables"
    if name == "submit_plan":
        plan = args.get("plan") or {}
        action = plan.get("action")
        if action == "clarify":
            return "Asked a clarifying question"
        if action == "out_of_scope":
            return "Found the question is outside the data"
        n = len(plan.get("sub_questions") or [])
        return "Planned the question as one query" if n <= 1 else f"Split the question into {n} parts"
    if name == "submit_finding":
        f = args.get("finding") or {}
        if not f.get("answered", True):
            return "Could not answer its sub-question"
        return f"Answered sub-question {f.get('sub_question_id', '')} ({f.get('confidence', '?')} confidence)".replace(" ()", "")
    if name == "submit_verification":
        v = args.get("verification") or {}
        verdict = {"ok": "all checks passed", "issues": "found issues",
                   "needs_more": "asked for a follow-up check"}.get(v.get("verdict"), v.get("verdict", "done"))
        return f"Verified the findings: {verdict}"
    if name in ("show_table", "show_chart", "show_map"):
        kind = {"show_table": "table", "show_chart": f"{args.get('chart_type', '')} chart".strip(),
                "show_map": "county map"}[name]
        if str(result).startswith("Error"):
            return f"Could not draw a {kind}: {str(result)[7:140]}"
        return f"Drew a {kind}" + (f": {args['title']}" if args.get("title") else "")
    if name == "note":
        return str(result)[:200]
    return name


class Recorder(RunHooks):
    def __init__(self, on_event: Callable[[dict], None] | None = None):
        self.tool_calls: list[dict] = []
        self.query_results: list[str] = []  # full run_sql results: evidence for the numbers check
        self.tokens = defaultdict(lambda: {"input": 0, "output": 0, "requests": 0})
        self._t0 = time.monotonic()
        self._on_event = on_event

    def _elapsed(self) -> float:
        return round(time.monotonic() - self._t0, 2)

    def emit(self, event: dict) -> None:
        """Report progress. A broken listener (e.g. a closed connection) must never stop the pipeline."""
        if self._on_event is None:
            return
        try:
            self._on_event({**event, "t": self._elapsed()})
        except Exception:  # noqa: BLE001
            pass

    def stage(self, stage: str, status: str, **data) -> None:
        """A pipeline step starting or finishing: plan, workers, verify, follow_ups, write, revise."""
        self.emit({"type": "stage", "stage": stage, "status": status, **data})

    def _add(self, agent: str, name: str, args: dict, result: str, summary: str) -> None:
        call = {"seq": len(self.tool_calls) + 1, "agent": agent, "name": name, "args": args,
                "result": result, "summary": summary, "t": self._elapsed()}
        self.tool_calls.append(call)
        self.emit({"type": "tool", **{k: v for k, v in call.items() if k not in ("result", "t")}})

    async def on_agent_start(self, context, agent) -> None:
        self.emit({"type": "agent", "agent": agent.name})

    async def on_tool_end(self, context, agent, tool, result) -> None:
        raw = getattr(context, "tool_arguments", None) or "{}"
        try:
            args = json.loads(raw)
        except json.JSONDecodeError:
            args = {"raw": raw}
        text = str(result)
        if tool.name == "run_sql":
            self.query_results.append(text)
        summary = summarize(tool.name, args, text)
        if len(text) > MAX_RESULT_CHARS:
            text = text[:MAX_RESULT_CHARS] + f"... [truncated, {len(text)} chars]"
        self._add(agent.name, tool.name, args, text, summary)

    async def on_llm_end(self, context, agent, response) -> None:
        usage = getattr(response, "usage", None)
        if usage:
            t = self.tokens[agent.name]
            t["input"] += usage.input_tokens or 0
            t["output"] += usage.output_tokens or 0
            t["requests"] += 1

    def note(self, agent: str, message: str) -> None:
        """A pipeline event that isn't a tool call (a timeout, a skipped step), so it's visible too."""
        self._add(agent, "note", {}, message, message[:200])

    def sql_calls_by(self, agent_name: str) -> list[dict]:
        return [c for c in self.tool_calls if c["agent"] == agent_name and c["name"] == "run_sql"]

    def token_summary(self) -> dict:
        total_in = sum(t["input"] for t in self.tokens.values())
        total_out = sum(t["output"] for t in self.tokens.values())
        return {"by_agent": dict(self.tokens), "input": total_in, "output": total_out}
