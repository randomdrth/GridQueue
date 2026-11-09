"""End-to-end pipeline tests with a scripted model: real tools, real database, no Gemini.

Each test scripts what the model "decides" and checks the pipeline's flow: branching,
parallel workers, agent-tagged tool calls, follow-up rounds, and graceful failure.
"""

import asyncio
import re
import time

import pytest

from gridqueue import pipeline
from gridqueue.roles import build_agents
from tests.fake_model import FakeModel, input_text

SUBQ = {
    1: ("How many active projects are in PJM?", "SELECT count(*) AS n FROM projects WHERE region='PJM' AND q_status='active'"),
    2: ("How many active projects are in MISO?", "SELECT count(*) AS n FROM projects WHERE region='MISO' AND q_status='active'"),
    3: ("How many withdrawn projects are in PJM?", "SELECT count(*) AS n FROM projects WHERE region='PJM' AND q_status='withdrawn'"),
}


def plan_step(action="answer", complexity="complex", ids=(1, 2), clarify="", oos=""):
    return [("submit_plan", {"plan": {
        "action": action, "complexity": complexity, "interpretation": "Active = q_status active.",
        "clarifying_question": clarify, "out_of_scope_reason": oos,
        "sub_questions": [{"id": i, "question": SUBQ[i][0], "hint": "projects"} for i in ids]}})]


def finding(sq_id, n):
    return {"finding": {"sub_question_id": sq_id, "answered": True, "answer": f"There are {n} projects.",
                        "table_columns": ["n"], "table_rows": [[str(n)]], "final_sql": SUBQ[sq_id][1],
                        "sample_size": "", "caveats": [], "confidence": "high"}}


def worker_script(turn, inp):
    """Turn 0: run the sub-question's query. Turn 1: submit what the query returned."""
    text = input_text(inp)
    sq_id = int(re.search(r"Your sub-question \(#(\d+)\)", text).group(1))
    if turn == 0:
        return [("run_sql", {"query": SUBQ[sq_id][1], "purpose": SUBQ[sq_id][0]})]
    n = int(re.search(r'\\"rows\\": \[\[(\d+)\]\]', text).group(1))
    return [("submit_finding", finding(sq_id, n))]


def verification(verdict="ok", follow_ups=()):
    return [("submit_verification", {"verification": {
        "verdict": verdict, "checks": ["numbers match run_sql results"], "issues": [], "corrections": [],
        "follow_up_questions": [{"id": 99, "question": SUBQ[i][0], "hint": "projects"} for i in follow_ups]}})]


def presenter_script(turn, inp):
    text = input_text(inp)
    if "do not appear in the findings" in text:  # a revision request
        return "PJM has 1,134 active projects."
    if turn == 0:
        return [("show_table", {"finding_id": 1, "columns": ["n"], "labels": ["Active projects"], "max_rows": 5,
                                "title": ""})]
    miso = " and MISO has 1,646" if "## Finding 2" in text else ""
    return f"PJM has 1,134 active projects{miso}.\n\n<!--gq:table:1-->"


def team(planner, worker=worker_script, verifier=lambda t, i: verification(), presenter=presenter_script,
         worker_delay=0.0, presenter_delay=0.0):
    scripts = {"planner": planner, "worker": worker, "verifier": verifier, "presenter": presenter}
    delays = {"worker": worker_delay, "presenter": presenter_delay}
    return build_agents(lambda role: FakeModel(role, scripts[role], delays.get(role, 0.0)))


def run(t, message="How many active projects are in PJM and MISO?", history=None):
    return asyncio.run(pipeline.run_pipeline(message, history=history, team=t))


# ---------------------------------------------------------------------------

def test_complex_question_full_pipeline():
    r = run(team(lambda t, i: plan_step()))
    assert r.response.startswith("PJM has 1,134 active projects and MISO has 1,646.")
    assert "| Active projects |" in r.response and "| 1,134 |" in r.response  # table rendered by code
    agents = {c["agent"] for c in r.tool_calls}
    assert agents == {"Planner", "SQL Worker 1", "SQL Worker 2", "Verifier", "Writer"}
    # Real queries ran against the real database.
    sql = [c for c in r.tool_calls if c["name"] == "run_sql"]
    assert {c["agent"] for c in sql} == {"SQL Worker 1", "SQL Worker 2"}
    assert '"rows": [[1134]]' in next(c["result"] for c in sql if c["agent"] == "SQL Worker 1")
    assert '"rows": [[1646]]' in next(c["result"] for c in sql if c["agent"] == "SQL Worker 2")
    # Plan, findings, verification and chart are all visible in tool_calls, in order.
    names = [c["name"] for c in r.tool_calls]
    assert names[0] == "submit_plan" and names[-1] == "show_table"
    table_call = next(c for c in r.tool_calls if c["name"] == "show_table")
    assert table_call["visual"]["rows"] == [[1134]] and "SELECT count(*)" in table_call["visual"]["sql"]
    assert [c["seq"] for c in r.tool_calls] == list(range(1, len(r.tool_calls) + 1))
    assert r.meta["verdict"] == "ok" and r.meta["complexity"] == "complex"


def test_workers_run_in_parallel():
    t0 = time.monotonic()
    r = run(team(lambda t, i: plan_step(ids=(1, 2, 3)), worker_delay=0.5))
    elapsed = time.monotonic() - t0
    # 3 workers x 2 model calls x 0.5s = 3s if sequential; about 1s in parallel.
    assert elapsed < 2.0, f"workers look sequential ({elapsed:.1f}s)"
    assert {c["agent"] for c in r.tool_calls if c["name"] == "submit_finding"} == \
        {"SQL Worker 1", "SQL Worker 2", "SQL Worker 3"}


def test_clarify_stops_before_any_query():
    r = run(team(lambda t, i: plan_step(action="clarify", ids=(), clarify="Best by what: cost, speed or completion rate?")),
            message="Which region is best?")
    assert r.response == "Best by what: cost, speed or completion rate?"
    assert [c["agent"] for c in r.tool_calls] == ["Planner"]


def test_out_of_scope_goes_to_writer_without_workers():
    r = run(team(lambda t, i: plan_step(action="out_of_scope", ids=(), oos="No load or data center demand data."),
                 presenter=lambda t, i: "The data covers generation, not data center demand."),
            message="How much power will Virginia data centers use?")
    assert "not data center demand" in r.response
    assert not any(c["agent"].startswith("SQL Worker") for c in r.tool_calls)


def test_verifier_follow_up_runs_one_more_worker():
    calls = {"n": 0}

    def verifier(turn, inp):
        calls["n"] += 1
        return verification("needs_more", follow_ups=(3,))

    r = run(team(lambda t, i: plan_step(), verifier=verifier))
    assert calls["n"] == 1, "only one verification round"
    assert "SQL Worker 3" in {c["agent"] for c in r.tool_calls}
    assert r.meta["follow_ups"] == 1


def test_worker_without_submit_is_unanswered_not_fatal():
    def worker(turn, inp):
        return "I think it is about a thousand."  # plain text, no submit_finding

    r = run(team(lambda t, i: plan_step(complexity="simple", ids=(1,)), worker=worker))
    assert any(c["name"] == "note" and "without a valid submit_finding" in c["result"] for c in r.tool_calls)
    assert r.response  # the Writer still answers


def test_worker_turn_limit_salvages_what_it_found():
    def worker(turn, inp):
        if "reached your step limit" in input_text(inp):
            return [("submit_finding", finding(1, 1134))]
        return [("get_data_gaps", {"topic": "PJM"})]  # never submits on its own

    r = run(team(lambda t, i: plan_step(complexity="simple", ids=(1,)), worker=worker))
    assert any(c["name"] == "note" and "step limit" in c["result"] for c in r.tool_calls)
    assert any(c["agent"] == "SQL Worker 1" and c["name"] == "submit_finding" for c in r.tool_calls)
    worker_steps = [c for c in r.tool_calls if c["agent"] == "SQL Worker 1" and c["name"] == "get_data_gaps"]
    assert len(worker_steps) == pipeline.WORKER_TURNS


def test_failed_salvage_is_unanswered_not_fatal():
    r = run(team(lambda t, i: plan_step(complexity="simple", ids=(1,)),
                 worker=lambda t, i: "still thinking" if "step limit" in input_text(i)
                 else [("get_data_gaps", {"topic": "PJM"})]))
    assert any(c["name"] == "note" and "without a valid submit_finding" in c["result"] for c in r.tool_calls)
    assert r.response


def test_writer_text_written_alongside_a_tool_call_is_kept():
    def presenter(turn, inp):
        if turn == 0:  # answer text and a tool call in the SAME turn, as Gemini did in the smoke run
            return ("PJM has 1,134 active projects.", presenter_script(0, inp))
        return "Sources: LBNL Queued Up 2026."

    r = run(team(lambda t, i: plan_step(), presenter=presenter))
    assert "PJM has 1,134 active projects." in r.response
    assert "Sources: LBNL Queued Up 2026." in r.response


def test_writer_timeout_returns_findings(monkeypatch):
    t = team(lambda t, i: plan_step(complexity="simple", ids=(1,)))
    original_run = pipeline.Runner.run

    async def run_with_slow_writer(agent, *args, **kwargs):
        if agent.name == "Writer":
            await asyncio.sleep(10)
        return await original_run(agent, *args, **kwargs)

    monkeypatch.setattr(pipeline.Runner, "run", run_with_slow_writer)
    monkeypatch.setattr(pipeline, "DEADLINE_S", 8)          # Writer gets ~8s minus planning, then times out
    monkeypatch.setattr(pipeline, "PRESENTER_RESERVE_S", 1)
    r = run(t)
    assert "There are 1134 projects." in r.response
    assert any(c["agent"] == "Writer" and c["name"] == "note" for c in r.tool_calls)


def test_all_workers_failing_raises():
    def broken(turn, inp):
        raise RuntimeError("Vertex auth failed")

    with pytest.raises(RuntimeError, match="Vertex auth failed"):
        run(team(lambda t, i: plan_step(), worker=broken))


def test_bad_plan_falls_back_to_single_question():
    r = run(team(lambda t, i: "I'll just answer directly.",  # planner never submits a plan
                 worker=lambda t, i: [("submit_finding", finding(1, 1134))]))
    assert r.meta["action"] == "answer" and r.meta["sub_questions"] == 1


def test_history_reaches_planner_only():
    seen = {}

    def planner(turn, inp):
        seen["planner"] = input_text(inp)
        return plan_step(complexity="simple", ids=(2,))

    def worker(turn, inp):
        seen.setdefault("worker", input_text(inp))
        return worker_script(turn, inp)

    history = [{"role": "user", "content": "How many active projects in PJM?"},
               {"role": "assistant", "content": "1,134."}]
    run(team(planner, worker=worker), message="What about MISO?", history=history)
    assert "1,134." in seen["planner"]
    assert "1,134." not in seen["worker"]


def test_unsupported_number_triggers_one_revision():
    drafts = iter(["PJM has 1,134 active projects, about 3 times MISO's.",   # "3 times" is invented
                   "PJM has 1,134 active projects; MISO has 1,646."])

    def presenter(turn, inp):
        return next(drafts)

    r = run(team(lambda t, i: plan_step(), presenter=presenter))
    assert r.response == "PJM has 1,134 active projects; MISO has 1,646."
    assert r.meta["unsupported_numbers"] == {"first_draft": ["3"], "final": []}
    assert any(c["name"] == "note" and "Asked for one revision" in c["result"] for c in r.tool_calls)


def test_grounded_answer_needs_no_revision():
    r = run(team(lambda t, i: plan_step(), presenter=lambda t, i: "PJM has 1,134 and MISO 1,646 active projects."))
    assert r.meta["unsupported_numbers"] == {"first_draft": [], "final": []}


def test_rate_limit_is_retried_with_backoff(monkeypatch):
    import litellm
    from gridqueue import roles
    monkeypatch.setattr(roles, "RETRY_INITIAL_DELAY_S", 0.05)
    failures = {"n": 0}

    class Flaky(FakeModel):
        async def get_response(self, *a, **k):
            if self.role == "planner" and failures["n"] < 2:
                failures["n"] += 1
                raise litellm.RateLimitError(message="Resource exhausted", llm_provider="vertex_ai", model="m")
            return await super().get_response(*a, **k)

    scripts = {"planner": lambda t, i: plan_step(complexity="simple", ids=(1,)), "worker": worker_script,
               "verifier": lambda t, i: verification(), "presenter": lambda t, i: "PJM has 1,134 active projects."}
    t = build_agents(lambda role: roles.ConcurrencyLimitedModel(Flaky(role, scripts[role])))
    r = run(t)
    assert failures["n"] == 2 and r.response == "PJM has 1,134 active projects."


def test_revision_that_calls_a_tool_keeps_first_draft():
    def presenter(turn, inp):
        if "do not appear in the findings" in input_text(inp):
            return [("show_table", {"finding_id": 1, "columns": ["n"], "labels": ["n"], "max_rows": 1, "title": ""})]
        return "PJM has 1,134 active projects, about 3 times more."

    r = run(team(lambda t, i: plan_step(complexity="simple", ids=(1,)), presenter=presenter))
    assert r.response.startswith("PJM has 1,134 active projects, about 3 times more.")
    assert any("Revision did not finish" in c["result"] for c in r.tool_calls if c["name"] == "note")


def test_duplicate_sub_questions_removed_and_complexity_fixed():
    # A smoke run's plan listed the same sub-question 5 times and called itself "simple".
    plan = plan_step(complexity="simple", ids=(1, 2))
    subqs = plan[0][1]["plan"]["sub_questions"]
    plan[0][1]["plan"]["sub_questions"] = subqs + [dict(subqs[0], id=7), dict(subqs[1], id=8), dict(subqs[0], id=9)]
    r = run(team(lambda t, i: plan))
    assert r.meta["sub_questions"] == 2 and r.meta["complexity"] == "complex"
    assert {c["agent"] for c in r.tool_calls if c["name"] == "submit_finding"} == {"SQL Worker 1", "SQL Worker 2"}


def test_first_draft_kept_when_revised():
    drafts = iter(["PJM has 1,134 projects, about 3 times MISO's.", "PJM has 1,134 projects."])
    r = run(team(lambda t, i: plan_step(), presenter=lambda t, i: next(drafts)))
    assert r.meta["first_draft"] == "PJM has 1,134 projects, about 3 times MISO's."


def test_drop_restart_keeps_the_last_full_answer():
    from gridqueue.pipeline import _drop_restart
    full = "Historically, PJM has connected projects faster than ERCOT, with a median of 1,150 days. More."
    assert _drop_restart(full[:85] + "\n**Data Anomalies:**\n" + full) == full
    assert _drop_restart(full) == full
