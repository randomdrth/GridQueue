"""Collect the README comparisons (Gemini on Vertex; needs gcloud application-default credentials).

    uv run python scripts/compare.py              # everything
    uv run python scripts/compare.py samples      # just the 3 README sample queries
    uv run python scripts/compare.py failures     # just the plain-agent candidates
    uv run python scripts/compare.py battery      # candidates whose question contains "battery"

Samples: each question goes to the model with no tools and no system prompt, and to GridQueue.
Failures: each candidate goes to a plain run_sql agent and to GridQueue, with both traces kept.
Writes runs/compare-<timestamp>.json after every question, so partial runs are kept.
"""

import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gridqueue.baseline import ask_no_tools, ask_plain_agent  # noqa: E402
from gridqueue.pipeline import run_pipeline  # noqa: E402

SAMPLES = [
    "How many new interconnection requests were filed each year since 2010, by resource type?",
    "Which counties have the most active queue capacity relative to the generation already operating there?",
    "Do projects that withdrew face higher interconnection costs than projects that got built? Compare by region.",
]

# Questions where a plain agent is likely to slip: guessed category spellings, a ratio over a tiny
# base, hybrids vs standalone storage, cost data that doesn't cover a region, cohort choice, joins.
FAILURE_CANDIDATES = [
    "How many projects in PJM are currently operating, and how much capacity do they add up to?",
    "How much battery storage capacity is waiting in the interconnection queue?",
    "Which county has the highest ratio of active queue capacity to existing generation?",
    "What is the median interconnection cost per kW for projects in ERCOT?",
    "What share of solar projects that requested interconnection actually got built?",
    "How much generation capacity is already operating in Texas, and how much is waiting in the queue there?",
]


def trimmed(calls: list[dict]) -> list[dict]:
    return [{k: v for k, v in c.items() if k != "visual"} for c in calls]


async def ours(question: str) -> dict:
    t0 = time.monotonic()
    try:
        r = await run_pipeline(question)
        return {"answer": r.response, "tool_calls": trimmed(r.tool_calls), "meta": r.meta,
                "seconds": round(time.monotonic() - t0, 1)}
    except Exception as e:  # noqa: BLE001
        return {"answer": f"[failed: {type(e).__name__}: {str(e)[:200]}]", "tool_calls": [], "meta": {}}


async def main(which: set[str]) -> None:
    words = {w for w in which if w not in ("samples", "failures")}
    if words:  # rerun only the matching failure candidates
        which = {"failures"}
    out = Path(__file__).resolve().parent.parent / "runs"
    out.mkdir(exist_ok=True)
    path = out / f"compare-{datetime.now():%Y%m%d-%H%M%S}.json"
    report = {"started": datetime.now().isoformat(timespec="seconds"), "samples": [], "failures": []}

    def save():
        path.write_text(json.dumps(report, indent=1, default=str))

    if not which or "samples" in which:
        for q in SAMPLES:
            print(f"\n[sample] {q}", flush=True)
            entry = {"question": q}
            try:
                entry["no_tools"] = await ask_no_tools(q)
            except Exception as e:  # noqa: BLE001
                entry["no_tools"] = f"[failed: {type(e).__name__}: {str(e)[:200]}]"
            print("  no tools done", flush=True)
            entry["gridqueue"] = await ours(q)
            print(f"  GridQueue done in {entry['gridqueue'].get('seconds')} s", flush=True)
            report["samples"].append(entry)
            save()

    if not which or "failures" in which:
        for q in FAILURE_CANDIDATES:
            if words and not any(w.lower() in q.lower() for w in words):
                continue
            print(f"\n[candidate] {q}", flush=True)
            entry = {"question": q, "plain": await ask_plain_agent(q)}
            print(f"  plain agent: {len([c for c in entry['plain']['tool_calls'] if c['name'] == 'run_sql'])} queries",
                  flush=True)
            entry["gridqueue"] = await ours(q)
            print(f"  GridQueue done in {entry['gridqueue'].get('seconds')} s", flush=True)
            report["failures"].append(entry)
            save()

    print(f"\nSaved {path}")


if __name__ == "__main__":
    asyncio.run(main(set(sys.argv[1:])))
