"""The four agents: Planner, SQL worker, Verifier, Writer (Presenter).

Each structured agent ends by calling its submit_* tool. tool_choice="required" makes the
model call a tool every turn, and stop_at_tool_names ends the run at the submit call, with
the submit tool's return value (JSON) as the final output.

Models come from environment variables so any agent can be swapped without code changes:
  GRIDQUEUE_MODEL            default for all agents (vertex_ai/gemini-3.5-flash)
  GRIDQUEUE_MODEL_PLANNER, _WORKER, _VERIFIER, _PRESENTER   per-agent overrides
"""

import asyncio
import os
from collections.abc import Callable

import litellm
from agents import Agent, ModelSettings, function_tool, set_tracing_disabled
from agents.extensions.models.litellm_model import LitellmModel
from agents.models.interface import Model
from agents.retry import ModelRetryBackoffSettings, ModelRetrySettings, retry_policies

from gridqueue import prompts
from gridqueue.schemas import Finding, Plan, Verification
from gridqueue.tools import describe_table_tool, get_column_values_tool, get_data_gaps_tool, run_sql_tool
from gridqueue.visuals import WRITER_TOOLS

# Traces upload to OpenAI by default, which needs an OpenAI key. We're on Gemini.
set_tracing_disabled(True)
# LiteLLM prints a "Give Feedback / Get Help" banner on every retried error. The SDK handles retries.
litellm.suppress_debug_info = True

DEFAULT_MODEL = "vertex_ai/gemini-3.5-flash"
ROLES = ("planner", "worker", "verifier", "presenter")


def model_name(role: str) -> str:
    return os.environ.get(f"GRIDQUEUE_MODEL_{role.upper()}") or os.environ.get("GRIDQUEUE_MODEL") or DEFAULT_MODEL


# --- Rate limits -------------------------------------------------------------
# Gemini 3.5 Flash on Vertex's global endpoint uses a shared capacity pool: when it's busy, or when we
# burst (parallel workers + verifier), it answers 429 "Resource exhausted" and expects us to back off.
# Two defences: the SDK retries transient errors with exponential backoff + jitter (2, 4, 8, 16s...),
# and a per-process cap on concurrent model calls so one request (or several graders) can't burst.
MAX_CONCURRENT_MODEL_CALLS = int(os.environ.get("MAX_CONCURRENT_MODEL_CALLS", 4))
RETRY_MAX = 5
RETRY_INITIAL_DELAY_S = 2.0
RETRY_MAX_DELAY_S = 20.0
RETRY_STATUSES = (408, 429, 500, 502, 503, 504)


def retry_settings() -> ModelRetrySettings:
    return ModelRetrySettings(
        max_retries=RETRY_MAX,
        backoff=ModelRetryBackoffSettings(initial_delay=RETRY_INITIAL_DELAY_S, max_delay=RETRY_MAX_DELAY_S,
                                          multiplier=2.0, jitter=True),
        policy=retry_policies.any(retry_policies.http_status(RETRY_STATUSES), retry_policies.network_error(),
                                  retry_policies.retry_after()),
    )


_semaphores: dict[int, asyncio.Semaphore] = {}


def _model_call_slot() -> asyncio.Semaphore:
    """One semaphore per event loop (tests run several loops; the server runs one)."""
    loop = asyncio.get_running_loop()
    return _semaphores.setdefault(id(loop), asyncio.Semaphore(MAX_CONCURRENT_MODEL_CALLS))


class ConcurrencyLimitedModel(Model):
    """Wraps a model so at most MAX_CONCURRENT_MODEL_CALLS calls are in flight per process."""

    def __init__(self, inner: Model):
        self.inner = inner

    async def get_response(self, *args, **kwargs):
        async with _model_call_slot():
            return await self.inner.get_response(*args, **kwargs)

    def stream_response(self, *args, **kwargs):
        return self.inner.stream_response(*args, **kwargs)


def litellm_model(role: str) -> Model:
    # api_key is unused: Vertex AI authenticates with gcloud credentials / the Cloud Run service account.
    return ConcurrencyLimitedModel(LitellmModel(model=model_name(role), api_key="unused"))


# A single model call normally takes 2-10 s. Smoke run 5 had one Worker call hang for ~130 s and
# eat the whole worker budget. Timing out a call lets the SDK retry it (timeouts count as network
# errors in retry_settings) instead of waiting on a stalled request.
MODEL_CALL_TIMEOUT_S = float(os.environ.get("GRIDQUEUE_CALL_TIMEOUT_S", "45"))


def _settings(tool_choice: str) -> ModelSettings:
    # Gemini 3.5 models are served only from Vertex's global endpoint. num_retries=0: retries are the
    # SDK's job (retry_settings), so LiteLLM doesn't stack its own quick retries on top.
    return ModelSettings(
        tool_choice=tool_choice,
        retry=retry_settings(),
        extra_args={"vertex_location": os.environ.get("VERTEX_LOCATION", "global"), "num_retries": 0,
                    "timeout": MODEL_CALL_TIMEOUT_S},
    )


# ---------------------------------------------------------------------------
# Submit tools: the structured output of each agent
# ---------------------------------------------------------------------------

@function_tool
def submit_plan(plan: Plan) -> str:
    """Submit your plan. Call exactly once, as your last step."""
    return plan.model_dump_json()


@function_tool
def submit_finding(finding: Finding) -> str:
    """Submit your finding for the sub-question. Call exactly once, as your last step."""
    return finding.model_dump_json()


@function_tool
def submit_verification(verification: Verification) -> str:
    """Submit your verification. Call exactly once, as your last step."""
    return verification.model_dump_json()


# ---------------------------------------------------------------------------
# Agent factory
#
# The planner and workers don't get list_tables: their instructions already describe every table,
# and the first smoke run showed it cost each of them a turn for nothing.
# ---------------------------------------------------------------------------

SUBMIT_TOOL = {"planner": "submit_plan", "worker": "submit_finding", "verifier": "submit_verification"}


def build_agents(model_for: Callable[[str], Model] = litellm_model) -> dict[str, Agent]:
    """Build the four agents. Tests pass a fake model_for; production uses LiteLLM -> Gemini."""
    required = _settings("required")
    return {
        "planner": Agent(
            name="Planner",
            instructions=prompts.PLANNER,
            model=model_for("planner"),
            model_settings=required,
            tools=[describe_table_tool, get_data_gaps_tool, submit_plan],
            tool_use_behavior={"stop_at_tool_names": ["submit_plan"]},
            reset_tool_choice=False,
        ),
        "worker": Agent(
            name="SQL Worker",
            instructions=prompts.WORKER,
            model=model_for("worker"),
            model_settings=required,
            tools=[describe_table_tool, get_column_values_tool, get_data_gaps_tool, run_sql_tool, submit_finding],
            tool_use_behavior={"stop_at_tool_names": ["submit_finding"]},
            reset_tool_choice=False,
        ),
        "verifier": Agent(
            name="Verifier",
            instructions=prompts.VERIFIER,
            model=model_for("verifier"),
            model_settings=required,
            tools=[run_sql_tool, get_data_gaps_tool, submit_verification],
            tool_use_behavior={"stop_at_tool_names": ["submit_verification"]},
            reset_tool_choice=False,
        ),
        "presenter": Agent(
            name="Writer",
            instructions=prompts.PRESENTER,
            model=model_for("presenter"),
            model_settings=_settings("auto"),
            tools=WRITER_TOOLS,
        ),
    }
