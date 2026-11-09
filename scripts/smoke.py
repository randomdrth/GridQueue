"""Run test questions through the real pipeline (Gemini on Vertex) and save a report.

    uv run python scripts/smoke.py            # all questions
    uv run python scripts/smoke.py 1 3 7      # just these

Writes runs/smoke-<timestamp>.json after every question, so partial runs are kept.
Needs Google credentials: gcloud auth application-default login.
"""

import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gridqueue.pipeline import run_pipeline  # noqa: E402
from gridqueue.roles import ROLES, model_name  # noqa: E402

# (id, question, what it tests). A question with follows=N continues question N's conversation.
QUESTIONS = [
    dict(id=1, kind="simple", q="How many projects are currently active in the PJM queue, and how many GW do they add up to?"),
    dict(id=2, kind="follow-up", follows=1, q="What about MISO?"),
    dict(id=3, kind="completion rates", q="What share of solar, wind, battery and gas projects that entered "
                                          "interconnection queues between 2000 and 2020 actually got built?"),
    dict(id=4, kind="multi-source", q="Do projects that withdrew face higher interconnection costs than projects "
                                      "that got built? Compare by region."),
    dict(id=5, kind="timing", q="How long does it typically take from interconnection request to commercial "
                                "operation in ERCOT compared with PJM?"),
    dict(id=6, kind="congestion", q="Which substations in the West have the most active interconnection requests?"),
    dict(id=7, kind="siting", q="Which counties have the most active queue capacity relative to the generation "
                                "already operating there?"),
    dict(id=8, kind="vague", q="Which region is best?"),
    dict(id=9, kind="out of scope", q="How much electricity will data centers in Virginia use in 2030?"),
]


async def main(selected: set[int]) -> None:
    out_dir = Path(__file__).resolve().parent.parent / "runs"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"smoke-{datetime.now():%Y%m%d-%H%M%S}.json"
    report = {"started": datetime.now().isoformat(timespec="seconds"),
              "models": {r: model_name(r) for r in ROLES}, "results": []}
    histories: dict[int, list[dict]] = {}

    for item in QUESTIONS:
        if selected and item["id"] not in selected:
            continue
        history = histories.get(item.get("follows"), [])
        print(f"\n[{item['id']}] ({item['kind']}) {item['q']}", flush=True)
        t0 = time.monotonic()
        entry = {**item}
        try:
            result = await run_pipeline(item["q"], history=history)
            entry.update(response=result.response, meta=result.meta, tool_calls=result.tool_calls)
            histories[item["id"]] = history + [{"role": "user", "content": item["q"]},
                                               {"role": "assistant", "content": result.response}]
            m = result.meta
            print(f"    {m['action']}/{m['complexity']}, {m['sub_questions']} sub-q, verdict={m['verdict']}, "
                  f"{len(result.tool_calls)} calls, {m['seconds']}s, "
                  f"{m['tokens']['input']:,} in / {m['tokens']['output']:,} out tokens")
            nums = m.get("unsupported_numbers")
            print(f"    visuals={m.get('visuals', [])}, numbers check: "
                  + (f"first draft {nums['first_draft'] or 'ok'} -> final {nums['final'] or 'ok'}" if nums else "n/a"))
            print("    " + result.response[:300].replace("\n", " ") + ("..." if len(result.response) > 300 else ""))
        except Exception as e:
            entry.update(error=f"{type(e).__name__}: {e}", seconds=round(time.monotonic() - t0, 1))
            print(f"    ERROR {entry['error'][:300]}")
        report["results"].append(entry)
        path.write_text(json.dumps(report, indent=1, default=str))

    print(f"\nSaved {path}")


if __name__ == "__main__":
    chosen = {int(a) for a in sys.argv[1:]}
    # A follow-up needs its parent's conversation, so run the parent too.
    chosen |= {q["follows"] for q in QUESTIONS if q["id"] in chosen and q.get("follows")}
    asyncio.run(main(chosen))
