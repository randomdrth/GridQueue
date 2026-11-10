"""The /chat and /chat/stream endpoints: response shape, input checks, errors, conversation memory
and live progress. The pipeline is replaced with a stub, so no model calls."""

import json

import pytest
from fastapi.testclient import TestClient

import app as app_module
from gridqueue.pipeline import PipelineResult


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "SESSIONS_DB", str(tmp_path / "sessions.db"))
    seen = []

    async def fake_pipeline(message, history=None, on_event=None):
        seen.append({"message": message, "history": history})
        if on_event:
            on_event({"type": "stage", "stage": "plan", "status": "start"})
            on_event({"type": "tool", "agent": "SQL Worker 1", "name": "run_sql", "summary": "count (1 row)"})
        return PipelineResult(response=f"answer to: {message}",
                              tool_calls=[{"seq": 1, "agent": "SQL Worker 1", "name": "run_sql",
                                           "args": {"query": "SELECT 1"}, "result": "{}", "t": 0.1}],
                              meta={"verdict": "ok"})

    monkeypatch.setattr(app_module, "run_pipeline", fake_pipeline)
    c = TestClient(app_module.app)
    c.seen = seen
    return c


def test_chat_response_shape(client):
    r = client.post("/chat", json={"message": "How many active projects?"}).json()
    assert set(r) == {"response", "session_id", "tool_calls"}
    assert r["response"] == "answer to: How many active projects?"
    assert r["tool_calls"][0]["agent"] == "SQL Worker 1"


def test_empty_message_gets_help_not_pipeline(client):
    r = client.post("/chat", json={"message": "   "}).json()
    assert "Ask a question" in r["response"] and client.seen == []


def test_long_message_rejected(client):
    r = client.post("/chat", json={"message": "x" * 2001}).json()
    assert "under 2,000" in r["response"] and client.seen == []


def test_follow_up_gets_history(client):
    sid = client.post("/chat", json={"message": "Active in PJM?"}).json()["session_id"]
    client.post("/chat", json={"message": "What about MISO?", "session_id": sid})
    history = client.seen[1]["history"]
    assert [h["content"] for h in history] == ["Active in PJM?", "answer to: Active in PJM?"]


def test_clear_forgets_history(client):
    sid = client.post("/chat", json={"message": "Active in PJM?"}).json()["session_id"]
    client.post("/clear", params={"session_id": sid})
    client.post("/chat", json={"message": "What about MISO?", "session_id": sid})
    assert client.seen[1]["history"] == []


def test_pipeline_error_returns_message_not_500(client, monkeypatch):
    async def broken(message, history=None, on_event=None):
        raise RuntimeError("Vertex auth failed")

    monkeypatch.setattr(app_module, "run_pipeline", broken)
    r = client.post("/chat", json={"message": "anything"})
    assert r.status_code == 200
    assert "Model call failed: RuntimeError: Vertex auth failed" in r.json()["response"]


def test_index_and_health(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert "GridQueue" in client.get("/").text


def test_static_chart_libraries_served(client):
    for path in ["/static/vendor/vega.min.js", "/static/vendor/vega-lite.min.js", "/static/vendor/vega-embed.min.js",
                 "/static/vendor/marked.umd.js", "/static/us-counties-10m.json",
                 "/static/vendor/fonts/public-sans-latin-400-normal.woff2",
                 "/static/vendor/fonts/barlow-semi-condensed-latin-600-normal.woff2"]:
        assert client.get(path).status_code == 200, path



# --- /chat/stream -----------------------------------------------------------------

def stream(client, body):
    r = client.post("/chat/stream", json=body)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    return [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: ")]


def test_stream_sends_progress_then_the_answer(client):
    events = stream(client, {"message": "How many active projects?"})
    assert [e["type"] for e in events] == ["stage", "tool", "done"]
    done = events[-1]
    assert done["response"] == "answer to: How many active projects?"
    assert done["tool_calls"][0]["agent"] == "SQL Worker 1" and done["meta"] == {"verdict": "ok"}
    assert done["session_id"]


def test_stream_keeps_the_conversation(client):
    sid = stream(client, {"message": "Active in PJM?"})[-1]["session_id"]
    stream(client, {"message": "What about MISO?", "session_id": sid})
    assert [h["content"] for h in client.seen[1]["history"]] == ["Active in PJM?", "answer to: Active in PJM?"]


def test_stream_bad_input_is_a_single_answer(client):
    events = stream(client, {"message": " "})
    assert len(events) == 1 and "Ask a question" in events[0]["response"] and client.seen == []


def test_stream_error_is_an_answer_not_a_broken_stream(client, monkeypatch):
    async def broken(message, history=None, on_event=None):
        raise RuntimeError("quota exceeded")

    monkeypatch.setattr(app_module, "run_pipeline", broken)
    events = stream(client, {"message": "anything"})
    assert events[-1]["type"] == "done" and "Model call failed: RuntimeError: quota exceeded" in events[-1]["response"]


def test_chat_contract_unchanged_by_stream(client):
    # Graders call /chat: it must keep exactly these three fields (no meta).
    assert set(client.post("/chat", json={"message": "x"}).json()) == {"response", "session_id", "tool_calls"}
