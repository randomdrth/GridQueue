"""The two baselines the README compares GridQueue against.

1. No tools, no system prompt: the same Gemini model, asked the question cold.
2. A plain run_sql agent: the same model and the same read-only database, told the table and
   column names, with one tool that runs SQL. No context files, no category values, no data-gap
   notes, no guardrails, no verifier and no numbers check. This is the starter-style agent that
   GridQueue's offline build step and multi-agent pipeline are meant to beat.
"""

import json
import os

import duckdb
import litellm
from agents import Agent, ModelSettings, Runner, function_tool

from gridqueue.db import cursor
from gridqueue.recorder import Recorder
from gridqueue.roles import MODEL_CALL_TIMEOUT_S, litellm_model, model_name, retry_settings

PLAIN_MAX_TURNS = 15
PLAIN_MAX_ROWS = 100


def schema_listing() -> str:
    """Table and column names with types, read from the database itself."""
    with cursor() as cur:
        rows = cur.execute("SELECT table_name, column_name, data_type FROM information_schema.columns "
                           "ORDER BY table_name, ordinal_position").fetchall()
    tables: dict[str, list[str]] = {}
    for table, column, dtype in rows:
        tables.setdefault(table, []).append(f"{column} {dtype}")
    return "\n".join(f"- {t}({', '.join(cols)})" for t, cols in tables.items())


def plain_run_sql(query: str) -> str:
    with cursor() as cur:
        try:
            cur.execute(query)
            rows = cur.fetchmany(PLAIN_MAX_ROWS + 1)
        except duckdb.Error as e:
            return json.dumps({"error": str(e)})
        columns = [c[0] for c in cur.description or []]
    return json.dumps({"columns": columns, "rows": rows[:PLAIN_MAX_ROWS], "truncated": len(rows) > PLAIN_MAX_ROWS},
                      default=str)


@function_tool(name_override="run_sql")
def run_sql_tool(query: str) -> str:
    """Run a DuckDB SQL query against the database and return the results.

    Args:
        query: A DuckDB SQL query.
    """
    return plain_run_sql(query)


def plain_agent() -> Agent:
    return Agent(
        name="Plain SQL agent",
        instructions=("You are a data analyst. Answer questions about US power grid interconnection queues "
                      "using the DuckDB database. Use run_sql to query it. Answer in plain English.\n\n"
                      "Tables:\n" + schema_listing()),
        tools=[run_sql_tool],
        model=litellm_model("worker"),
        model_settings=ModelSettings(retry=retry_settings(), extra_args={
            "vertex_location": os.environ.get("VERTEX_LOCATION", "global"), "num_retries": 0,
            "timeout": MODEL_CALL_TIMEOUT_S}),
    )


async def ask_plain_agent(question: str) -> dict:
    rec = Recorder()
    try:
        r = await Runner.run(plain_agent(), question, max_turns=PLAIN_MAX_TURNS, hooks=rec)
        answer = str(r.final_output)
    except Exception as e:  # noqa: BLE001 -- a baseline failure is a result worth recording
        answer = f"[failed: {type(e).__name__}: {str(e)[:200]}]"
    return {"answer": answer, "tool_calls": rec.tool_calls}


async def ask_no_tools(question: str) -> str:
    """The model alone: no system prompt, no tools."""
    r = await litellm.acompletion(model=model_name("presenter"), messages=[{"role": "user", "content": question}],
                                  vertex_location=os.environ.get("VERTEX_LOCATION", "global"),
                                  timeout=120, num_retries=3)
    return r.choices[0].message.content
