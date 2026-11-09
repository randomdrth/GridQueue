"""Records every tool call, with the agent that made it, for the /chat tool_calls list.

One Recorder per request, passed as hooks to every Runner.run in the pipeline, so the
Planner, each SQL worker, the Verifier and the Writer all land in one ordered list.
"""

import json
import time
from collections import defaultdict

from agents import RunHooks

MAX_RESULT_CHARS = 6_000  # per tool call in the response; the agent itself saw the full result


class Recorder(RunHooks):
    def __init__(self):
        self.tool_calls: list[dict] = []
        self.query_results: list[str] = []  # full run_sql results: evidence for the numbers check
        self.tokens = defaultdict(lambda: {"input": 0, "output": 0, "requests": 0})
        self._t0 = time.monotonic()

    def _elapsed(self) -> float:
        return round(time.monotonic() - self._t0, 2)

    async def on_tool_end(self, context, agent, tool, result) -> None:
        raw = getattr(context, "tool_arguments", None) or "{}"
        try:
            args = json.loads(raw)
        except json.JSONDecodeError:
            args = {"raw": raw}
        text = str(result)
        if tool.name == "run_sql":
            self.query_results.append(text)
        if len(text) > MAX_RESULT_CHARS:
            text = text[:MAX_RESULT_CHARS] + f"... [truncated, {len(text)} chars]"
        self.tool_calls.append({"seq": len(self.tool_calls) + 1, "agent": agent.name, "name": tool.name,
                                "args": args, "result": text, "t": self._elapsed()})

    async def on_llm_end(self, context, agent, response) -> None:
        usage = getattr(response, "usage", None)
        if usage:
            t = self.tokens[agent.name]
            t["input"] += usage.input_tokens or 0
            t["output"] += usage.output_tokens or 0
            t["requests"] += 1

    def note(self, agent: str, message: str) -> None:
        """A pipeline event that isn't a tool call (a timeout, a skipped step), so it's visible too."""
        self.tool_calls.append({"seq": len(self.tool_calls) + 1, "agent": agent, "name": "note",
                                "args": {}, "result": message, "t": self._elapsed()})

    def sql_calls_by(self, agent_name: str) -> list[dict]:
        return [c for c in self.tool_calls if c["agent"] == agent_name and c["name"] == "run_sql"]

    def token_summary(self) -> dict:
        total_in = sum(t["input"] for t in self.tokens.values())
        total_out = sum(t["output"] for t in self.tokens.values())
        return {"by_agent": dict(self.tokens), "input": total_in, "output": total_out}
