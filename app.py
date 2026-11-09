"""GridQueue web app: the /chat endpoint and the page.

Run locally:  uv run app.py   then open http://localhost:8000
Cloud Run:    uvicorn app:app --host 0.0.0.0 --port $PORT
"""

import os
import tempfile
import uuid
from pathlib import Path

import uvicorn
from agents import SQLiteSession
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from gridqueue.pipeline import run_pipeline

MAX_MESSAGE_CHARS = 2_000
HISTORY_ITEMS = 8  # last 4 exchanges are enough for follow-ups like "now just PJM"

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

    if not message:
        return ChatResponse(response="Ask a question about US grid interconnection queues, costs or "
                                     "existing power plants.", session_id=session_id, tool_calls=[])
    if len(message) > MAX_MESSAGE_CHARS:
        return ChatResponse(response=f"That message is {len(message):,} characters. Please keep questions "
                                     f"under {MAX_MESSAGE_CHARS:,}.", session_id=session_id, tool_calls=[])

    session = SQLiteSession(session_id, SESSIONS_DB)
    try:
        history = await session.get_items(limit=HISTORY_ITEMS)
        result = await run_pipeline(message, history=history)
        response, tool_calls = result.response, result.tool_calls
        await session.add_items([{"role": "user", "content": message},
                                 {"role": "assistant", "content": response}])
    except Exception as e:  # auth, quota, a model that is down: show it in the chat, not as a 500
        response, tool_calls = f"Model call failed: {type(e).__name__}: {str(e)[:300]}", []

    return ChatResponse(response=response, session_id=session_id, tool_calls=tool_calls)


@app.post("/clear")
async def clear(session_id: str | None = None):
    if session_id:
        await SQLiteSession(session_id, SESSIONS_DB).clear_session()
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))
