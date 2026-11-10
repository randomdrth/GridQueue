"""Replay mode: answer from saved smoke runs instead of calling the model.

For building and testing the page without model cost or waiting:

    GRIDQUEUE_REPLAY=runs uv run uvicorn app:app --port 8000

GRIDQUEUE_REPLAY is a smoke-run JSON file or a folder of them (later files win for the same
question). A message is matched to the closest saved question. The saved tool calls are played
back as live progress events, compressed to about GRIDQUEUE_REPLAY_SECONDS (default 10).

Charts and maps are rebuilt with the current visuals code from each saved finding's SQL, so a
change to the visuals shows up without re-running the model. Tables are already in the saved text.
"""

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Callable

from gridqueue import visuals
from gridqueue.pipeline import PipelineResult
from gridqueue.recorder import summarize
from gridqueue.schemas import Finding

MIN_MATCH = 0.5  # share of the question's words the saved question must share


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def load(source: str | None = None) -> dict[str, dict]:
    """Saved answers keyed by question text."""
    path = Path(source or os.environ.get("GRIDQUEUE_REPLAY", "runs"))
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    saved: dict[str, dict] = {}
    for f in files:
        try:
            report = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for item in report.get("results", []):
            if item.get("response") and not item.get("error"):
                saved[item["q"]] = item
    return saved


def best_match(message: str, saved: dict[str, dict]) -> dict | None:
    asked = _words(message)
    if not asked:
        return None
    score, item = max(((len(asked & _words(q)) / len(asked | _words(q)), it) for q, it in saved.items()),
                      default=(0, None), key=lambda x: x[0])
    return item if score >= MIN_MATCH else None


def _rebuild_visuals(calls: list[dict]) -> None:
    """Re-run each show_* call with the current visuals code, in the original order (so markers match)."""
    findings = {}
    for c in calls:
        if c["name"] == "submit_finding":
            try:
                f = Finding.model_validate(c["args"]["finding"])
                findings[f.sub_question_id] = f
            except Exception:  # noqa: BLE001 -- an old run's shape: keep its saved visuals
                pass
    ctx = visuals.WriterContext(findings=findings)
    builders = {"show_table": visuals._table, "show_chart": visuals._chart, "show_map": visuals._map}
    for c in calls:
        build = builders.get(c["name"])
        if not build:
            continue
        args = dict(c["args"])
        try:
            result = build(ctx, **args)
        except Exception:  # noqa: BLE001
            continue
        v = ctx.visuals[-1]
        if "visual" in c and v["marker"] == c["visual"].get("marker"):
            c["visual"] = {k: val for k, val in v.items() if k != "markdown"}
            c["result"] = result


def _stage_for(agent: str) -> str:
    if agent.startswith("SQL Worker"):
        return "workers"
    return {"Planner": "plan", "Verifier": "verify", "Writer": "write"}.get(agent, "")


async def replay_pipeline(message: str, history: list[dict] | None = None, team=None,
                          on_event: Callable[[dict], None] | None = None) -> PipelineResult:
    item = best_match(message, load())
    emit = on_event or (lambda e: None)
    if item is None:
        return PipelineResult(response="Replay mode has no saved answer close to that question. "
                                       "Ask one of the smoke-test questions.", tool_calls=[], meta={})

    calls = [dict(c) for c in item.get("tool_calls", [])]
    for c in calls:
        c.setdefault("summary", summarize(c["name"], c.get("args", {}), str(c.get("result", ""))))
    _rebuild_visuals(calls)

    meta = dict(item.get("meta") or {})
    plan_call = next((c for c in calls if c["name"] == "submit_plan"), None)
    plan = (plan_call or {}).get("args", {}).get("plan", {})
    meta.setdefault("plan", {"action": plan.get("action", meta.get("action", "answer")),
                             "interpretation": plan.get("interpretation", ""),
                             "sub_questions": [sq.get("question", "") for sq in plan.get("sub_questions", [])]})

    total = max((c.get("t", 0) for c in calls), default=0)
    speed = max(1.0, total / float(os.environ.get("GRIDQUEUE_REPLAY_SECONDS", 10)))
    clock, stage, seen_agents = 0.0, None, set()

    async def move_to(new_stage: str) -> None:
        nonlocal stage
        if new_stage == stage or not new_stage:
            return
        if stage:
            done = {"plan": meta["plan"], "verify": {"verdict": meta.get("verdict")}}.get(stage, {})
            emit({"type": "stage", "stage": stage, "status": "done", **done, "t": clock})
        stage = new_stage
        extra = {"count": len(meta["plan"]["sub_questions"])} if new_stage == "workers" else {}
        emit({"type": "stage", "stage": stage, "status": "start", **extra, "t": clock})

    await move_to("plan")
    for c in calls:
        wait = (c.get("t", clock) - clock) / speed
        if wait > 0:
            await asyncio.sleep(wait)
        clock = c.get("t", clock)
        await move_to(_stage_for(c["agent"]) or stage)
        if c["agent"] not in seen_agents:
            seen_agents.add(c["agent"])
            emit({"type": "agent", "agent": c["agent"], "t": clock})
        emit({"type": "tool", **{k: v for k, v in c.items() if k not in ("result", "t", "visual")}, "t": clock})
    await move_to("write")
    emit({"type": "stage", "stage": "write", "status": "done", "t": clock})
    return PipelineResult(response=item["response"], tool_calls=calls, meta=meta)
