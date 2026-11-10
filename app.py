"""GridQueue web app: the /chat endpoint and the page.

Run locally:  uv run app.py   then open http://localhost:8000
Cloud Run:    uvicorn app:app --host 0.0.0.0 --port $PORT

Endpoints:
  POST /chat         {message, session_id?} -> {response, session_id, tool_calls}
  POST /chat/stream  same request; Server-Sent Events with live progress, then the same answer
  POST /clear        forget a session's history
  GET  /health

Set GRIDQUEUE_REPLAY=runs to answer from saved smoke runs instead of the model (see gridqueue/replay.py).
"""

import asyncio
import json
import os
import tempfile
import uuid
from pathlib import Path

import uvicorn
from agents import SQLiteSession
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

if os.environ.get("GRIDQUEUE_REPLAY"):
    from gridqueue.replay import replay_pipeline as run_pipeline
else:
    from gridqueue.pipeline import run_pipeline

MAX_MESSAGE_CHARS = 2_000
HISTORY_ITEMS = 8  # last 4 exchanges are enough for follow-ups like "now just PJM"
KEEPALIVE_S = 15   # comment line on a quiet stream, so proxies don't close it

# Only user messages and final answers are stored: the planner needs the conversation, workers don't.
# Cloud Run's filesystem is per-instance and resets on restart, which is fine for chat history.
SESSIONS_DB = os.environ.get("SESSIONS_DB", str(Path(tempfile.gettempdir()) / "gridqueue_sessions.db"))

app = FastAPI(title="GridQueue")
# Chart libraries and county shapes are served from the repo, not a CDN, so the deployed app
# doesn't depend on an outside site being up.
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    tool_calls: list[dict]


def _input_problem(message: str) -> str | None:
    """A reply for input that shouldn't reach the pipeline, or None if it's fine."""
    if not message:
        return "Ask a question about US grid interconnection queues, costs or existing power plants."
    if len(message) > MAX_MESSAGE_CHARS:
        return f"That message is {len(message):,} characters. Please keep questions under {MAX_MESSAGE_CHARS:,}."
    return None


async def _answer(message: str, session_id: str, on_event=None) -> tuple[str, list[dict], dict]:
    """Run the pipeline with the session's history and save the exchange. Never raises."""
    session = SQLiteSession(session_id, SESSIONS_DB)
    try:
        history = await session.get_items(limit=HISTORY_ITEMS)
        kwargs = {"on_event": on_event} if on_event else {}
        result = await run_pipeline(message, history=history, **kwargs)
        await session.add_items([{"role": "user", "content": message},
                                 {"role": "assistant", "content": result.response}])
        return result.response, result.tool_calls, result.meta
    except Exception as e:  # auth, quota, a model that is down: show it in the chat, not as a 500
        return f"Model call failed: {type(e).__name__}: {str(e)[:300]}", [], {"error": True}


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())
    message = request.message.strip()
    problem = _input_problem(message)
    if problem:
        return ChatResponse(response=problem, session_id=session_id, tool_calls=[])
    response, tool_calls, _ = await _answer(message, session_id)
    return ChatResponse(response=response, session_id=session_id, tool_calls=tool_calls)


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, default=str)}\n\n"


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Live progress for the page. Events (one JSON object per `data:` line):
      {"type": "stage", "stage": "plan|workers|verify|follow_ups|write|revise", "status": "start|done", ...}
      {"type": "agent", "agent": "SQL Worker 1"}
      {"type": "tool",  "agent", "name", "args", "summary", "seq"}
      {"type": "done",  "response", "session_id", "tool_calls", "meta"}   always last
    /chat stays the graded contract; this endpoint only adds the progress and the step summary (meta)."""
    session_id = request.session_id or str(uuid.uuid4())
    message = request.message.strip()
    problem = _input_problem(message)

    if problem:
        async def single():
            yield _sse({"type": "done", "response": problem, "session_id": session_id, "tool_calls": [], "meta": {}})
        return StreamingResponse(single(), media_type="text/event-stream")

    queue: asyncio.Queue = asyncio.Queue()

    async def work():
        try:
            response, tool_calls, meta = await _answer(message, session_id, on_event=queue.put_nowait)
            queue.put_nowait({"type": "done", "response": response, "session_id": session_id,
                              "tool_calls": tool_calls, "meta": meta})
        finally:
            queue.put_nowait(None)

    task = asyncio.create_task(work())

    async def events():
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=KEEPALIVE_S)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                if event is None:
                    break
                yield _sse(event)
        finally:
            if not task.done():  # the page went away: stop spending model calls on it
                task.cancel()

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/clear")
async def clear(session_id: str | None = None):
    if session_id:
        await SQLiteSession(session_id, SESSIONS_DB).clear_session()
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))
