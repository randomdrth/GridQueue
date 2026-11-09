"""A scripted stand-in for Gemini, so the pipeline can be tested offline.

Each role gets a script: a function (turn, input) -> list of tool calls, or a string for a
final text answer. The turn number is how many tool results the agent has seen so far.
Real tools still run against the real database; only the model's decisions are scripted.
"""

import asyncio
import json
import uuid
from collections.abc import Callable

from agents.items import ModelResponse
from agents.models.interface import Model
from agents.usage import Usage
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText

# A step is tool calls, final text, or (text, tool calls) in the same turn.
Step = list[tuple[str, dict]] | str | tuple[str, list[tuple[str, dict]]]
Script = Callable[[int, object], Step]

ROLE_MARKERS = {"planner": "Your role: Planner", "worker": "Your role: SQL worker",
                "verifier": "Your role: Verifier", "presenter": "Your role: Writer"}


class FakeModel(Model):
    def __init__(self, role: str, script: Script, delay_s: float = 0.0):
        self.role, self.script, self.delay_s = role, script, delay_s
        self.calls = 0

    async def get_response(self, system_instructions, input, model_settings, tools, output_schema, handoffs,
                           tracing, *, previous_response_id=None, conversation_id=None, prompt=None):
        assert ROLE_MARKERS[self.role] in (system_instructions or ""), "agent got the wrong instructions"
        self.calls += 1
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        turn = sum(1 for i in input if isinstance(i, dict) and i.get("type") == "function_call_output") \
            if isinstance(input, list) else 0
        step = self.script(turn, input)
        text, calls = (step, []) if isinstance(step, str) else step if isinstance(step, tuple) else ("", step)
        output = []
        if text:
            output.append(ResponseOutputMessage(id=f"msg_{uuid.uuid4().hex[:8]}", role="assistant",
                                                status="completed", type="message",
                                                content=[ResponseOutputText(text=text, type="output_text",
                                                                            annotations=[])]))
        output += [ResponseFunctionToolCall(arguments=json.dumps(args), call_id=f"call_{uuid.uuid4().hex[:8]}",
                                            name=name, type="function_call", id=f"fc_{uuid.uuid4().hex[:8]}",
                                            status="completed") for name, args in calls]
        return ModelResponse(output=output, usage=Usage(requests=1, input_tokens=100, output_tokens=20,
                                                        total_tokens=120), response_id=None)

    def stream_response(self, *args, **kwargs):
        raise NotImplementedError


def input_text(inp) -> str:
    """Everything the agent was given, as one string (to read the sub-question in a script)."""
    return inp if isinstance(inp, str) else json.dumps(inp, default=str)
