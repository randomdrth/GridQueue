"""Replay mode: saved runs played back as live progress, with visuals rebuilt from their SQL."""

import asyncio
import json

from gridqueue import replay

SQL = "SELECT fips_code, county, state, existing_mw, active_queue_mw FROM county_summary WHERE existing_mw >= 50"
FINDING = {"sub_question_id": 1, "answered": True, "answer": "a", "table_columns": [], "table_rows": [],
           "final_sql": SQL, "sample_size": "", "caveats": [], "confidence": "high"}


def saved_run(tmp_path):
    calls = [
        {"seq": 1, "agent": "Planner", "name": "submit_plan", "t": 1.0, "result": "{}",
         "args": {"plan": {"action": "answer", "interpretation": "i",
                           "sub_questions": [{"id": 1, "question": "Which counties?", "hint": ""}]}}},
        {"seq": 2, "agent": "SQL Worker 1", "name": "run_sql", "t": 2.0, "result": '{"row_count": 3}',
         "args": {"query": SQL, "purpose": "counties"}},
        {"seq": 3, "agent": "SQL Worker 1", "name": "submit_finding", "t": 3.0, "result": "{}",
         "args": {"finding": FINDING}},
        {"seq": 4, "agent": "Verifier", "name": "submit_verification", "t": 4.0, "result": "{}",
         "args": {"verification": {"verdict": "ok"}}},
        {"seq": 5, "agent": "Writer", "name": "show_map", "t": 5.0, "result": "ok",
         "args": {"finding_id": 1, "value_column": "active_queue_mw", "title": "Queue", "value_label": "MW"},
         "visual": {"kind": "map", "marker": "<!--gq:map:1-->", "spec": {"old": True}}},
    ]
    report = {"results": [{"id": 7, "q": "Which counties have the most active queue capacity?",
                           "response": "Deschutes leads.\n\n<!--gq:map:1-->", "meta": {"verdict": "ok"},
                           "tool_calls": calls}]}
    path = tmp_path / "smoke.json"
    path.write_text(json.dumps(report))
    return path


def test_match_needs_a_close_question(tmp_path):
    saved = replay.load(str(saved_run(tmp_path)))
    assert replay.best_match("which counties have the most active queue capacity", saved)["id"] == 7
    assert replay.best_match("how much does solar cost in Spain", saved) is None


def test_replay_streams_progress_and_rebuilds_visuals(tmp_path, monkeypatch):
    monkeypatch.setenv("GRIDQUEUE_REPLAY", str(saved_run(tmp_path)))
    monkeypatch.setenv("GRIDQUEUE_REPLAY_SECONDS", "0.05")
    events = []
    r = asyncio.run(replay.replay_pipeline("Which counties have the most active queue capacity?", on_event=events.append))
    assert r.response.startswith("Deschutes")
    stages = [(e["stage"], e["status"]) for e in events if e["type"] == "stage"]
    assert stages[0] == ("plan", "start") and stages[-1] == ("write", "done")
    assert ("verify", "done") in stages
    assert [e["name"] for e in events if e["type"] == "tool"] == [c["name"] for c in r.tool_calls]
    v = r.tool_calls[-1]["visual"]
    assert "old" not in v["spec"] and [m["column"] for m in v["metrics"]] == ["active_queue_mw", "existing_mw"]
    assert r.meta["plan"]["sub_questions"] == ["Which counties?"]


def test_replay_without_a_match_says_so(tmp_path, monkeypatch):
    monkeypatch.setenv("GRIDQUEUE_REPLAY", str(saved_run(tmp_path)))
    r = asyncio.run(replay.replay_pipeline("What is the capital of France?"))
    assert "no saved answer" in r.response and r.tool_calls == []
