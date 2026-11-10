"""The GridQueue pipeline: code decides the flow, LLM agents do each step.

    Planner ──► clarify ───────────────────────────────► return the question
       │
       ├──► out_of_scope ───────────────────────────────► Writer
       │
       └──► answer ──► SQL workers (parallel) ──► Verifier ──► Writer
                              ▲                      │
                              └── needs_more (once) ─┘

Every step has a turn limit and a time budget inside one overall deadline, so a request
finishes well inside the grader's 5 minutes. When a step fails or runs out of time, the
pipeline degrades (records a note, carries on with what it has) instead of crashing.
"""

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Callable

from agents import Agent, ItemHelpers, MaxTurnsExceeded, MessageOutputItem, ModelSettings, RunErrorHandlerInput, RunErrorHandlerResult, Runner
from pydantic import BaseModel, ValidationError

from gridqueue import grounding, visuals
from gridqueue.recorder import Recorder
from gridqueue.roles import SUBMIT_TOOL, build_agents
from gridqueue.schemas import Finding, Plan, SubQuestion, Verification
from gridqueue.tools import fetch_rows

# Turn limits (one turn = one model call, possibly with several tool calls). Gemini usually makes
# one tool call per turn, so a worker needs ~4 turns of lookups plus its queries. Set from the first
# smoke run, where 12 worker turns was too tight for 3 of 9 questions.
PLANNER_TURNS = 8
WORKER_TURNS = 20
VERIFIER_TURNS = 10
PRESENTER_TURNS = 6        # up to 3 visual tool calls, then the answer
REVISION_MIN_S = 25        # only ask the Writer to fix numbers if this much time is left
SALVAGE_TURNS = 2

# Time budget, seconds. The grader waits up to 300.
DEADLINE_S = float(os.environ.get("GRIDQUEUE_DEADLINE_S", 240))
PLANNER_MAX_S = 60
VERIFIER_MAX_S = 75
PRESENTER_RESERVE_S = 45   # always keep this much for the Writer
VERIFIER_RESERVE_S = 40    # and this much for the Verifier while workers run

MAX_SUB_QUESTIONS = 4
MAX_FOLLOW_UPS = 2
MAX_PARALLEL_WORKERS = 4
SQL_EVIDENCE_CHARS = 2_500  # per worker query shown to the Verifier

SKIP_VERIFIER_FOR_SIMPLE = os.environ.get("SKIP_VERIFIER_FOR_SIMPLE", "false").lower() == "true"

_team: dict[str, Agent] | None = None


def default_team() -> dict[str, Agent]:
    global _team
    if _team is None:
        _team = build_agents()
    return _team


@dataclass
class PipelineResult:
    response: str
    tool_calls: list[dict]
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse(model: type[BaseModel], output) -> BaseModel | None:
    if isinstance(output, model):
        return output
    if isinstance(output, str):
        try:
            return model.model_validate_json(output)
        except ValidationError:
            return None
    return None


def _fallback_plan(message: str, why: str) -> Plan:
    return Plan(action="answer", complexity="simple", interpretation=f"Answering the question as asked ({why}).",
                clarifying_question="", out_of_scope_reason="",
                sub_questions=[SubQuestion(id=1, question=message, hint="")])


def _unanswered(sq: SubQuestion, why: str) -> Finding:
    return Finding(sub_question_id=sq.id, answered=False, answer=f"Could not answer: {why}", table_columns=[],
                   table_rows=[], final_sql="", sample_size="", caveats=[], confidence="low")


def _clean_plan(plan: Plan, message: str) -> Plan:
    """Guard against plans the pipeline can't run: no sub-questions, too many, duplicate ids."""
    if plan.action == "clarify" and not plan.clarifying_question.strip():
        plan.action = "answer"
    if plan.action == "answer":
        # Drop repeated sub-questions (a smoke run's plan listed the same one 5 times), then cap.
        seen, unique = set(), []
        for sq in plan.sub_questions:
            key = " ".join(sq.question.lower().split())
            if key and key not in seen:
                seen.add(key)
                unique.append(sq)
        plan.sub_questions = unique[:MAX_SUB_QUESTIONS] or [SubQuestion(id=1, question=message, hint="")]
        for i, sq in enumerate(plan.sub_questions, start=1):
            sq.id = i
        plan.complexity = "simple" if len(plan.sub_questions) == 1 else "complex"
    return plan


def _salvage(role: str, rec: Recorder):
    """Turn-limit handler: instead of losing everything an agent found, give it the full transcript
    and one last chance to call its submit tool with what it has, marking anything unfinished."""
    submit = SUBMIT_TOOL[role]

    async def handler(data: RunErrorHandlerInput) -> RunErrorHandlerResult:
        agent = data.run_data.last_agent
        rec.note(agent.name, "Reached its step limit; asked to submit what it found so far.")
        closer = agent.clone(model_settings=agent.model_settings.resolve(ModelSettings(tool_choice=submit)))
        transcript = list(data.run_data.history) + [{"role": "user", "content": (
            f"You have reached your step limit. Call {submit} now with what your results so far show. "
            "Do not run more queries. Mark anything unfinished (answered=false or lower confidence) "
            "rather than guessing.")}]
        try:
            result = await Runner.run(closer, transcript, max_turns=SALVAGE_TURNS, hooks=rec)
            return RunErrorHandlerResult(final_output=result.final_output, include_in_history=False)
        except Exception as e:  # noqa: BLE001 -- a failed salvage must not sink the request
            rec.note(agent.name, f"Could not submit after the step limit ({type(e).__name__}).")
            return RunErrorHandlerResult(final_output=None, include_in_history=False)

    return handler


# ---------------------------------------------------------------------------
# Inputs for each agent
# ---------------------------------------------------------------------------

def _worker_input(message: str, plan: Plan, sq: SubQuestion) -> str:
    return (f"The user asked: {message}\n"
            f"How the question is being read: {plan.interpretation}\n\n"
            f"Your sub-question (#{sq.id}): {sq.question}\n"
            f"Hint: {sq.hint or 'none'}\n\n"
            "Answer only this sub-question, then call submit_finding.")


def _verifier_input(message: str, plan: Plan, findings: list[Finding], rec: Recorder) -> str:
    blocks = []
    for f in findings:
        calls = rec.sql_calls_by(f"SQL Worker {f.sub_question_id}")[-4:]
        evidence = "\n".join(
            f"- query: {c['args'].get('query', '')}\n  result: {c['result'][:SQL_EVIDENCE_CHARS]}" for c in calls
        ) or "- (no run_sql calls)"
        blocks.append(f"## Finding {f.sub_question_id}\n{f.model_dump_json(indent=1)}\n"
                      f"### run_sql calls this worker made (most recent last)\n{evidence}")
    subqs = "\n".join(f"{sq.id}. {sq.question}" for sq in plan.sub_questions)
    return (f"The user asked: {message}\nInterpretation: {plan.interpretation}\n\nSub-questions:\n{subqs}\n\n"
            + "\n\n".join(blocks) + "\n\nVerify, then call submit_verification.")


def _presenter_input(message: str, plan: Plan, findings: list[Finding], verification: Verification | None,
                     unverified_ids: set[int]) -> str:
    parts = [f"The user asked: {message}"]
    if plan.action == "out_of_scope":
        parts.append(f"This is out of scope for the data. What the data lacks: {plan.out_of_scope_reason}\n"
                     "Explain that plainly and suggest 1-2 related questions the data can answer.")
        return "\n\n".join(parts)
    parts.append(f"Interpretation and assumptions: {plan.interpretation}")
    for f in findings:
        tag = " (follow-up check, not re-verified)" if f.sub_question_id in unverified_ids else ""
        parts.append(f"## Finding {f.sub_question_id}{tag}\n{f.model_dump_json(indent=1)}\n"
                     f"Result columns for show_table / show_chart / show_map: {_result_columns(f)}")
    if verification:
        if verification.corrections:
            parts.append("## Corrections (use these over the findings)\n- " + "\n- ".join(verification.corrections))
        if verification.issues:
            parts.append("## Unresolved issues to mention if they affect the answer\n- "
                         + "\n- ".join(f"Finding {i.sub_question_id}: {i.problem}" for i in verification.issues))
    else:
        parts.append("Note: these findings were not independently verified.")
    parts.append("Write the final answer now.")
    return "\n\n".join(parts)


def _result_columns(f: Finding) -> str:
    """The columns a finding's final_sql actually returns, so the Writer names real columns."""
    if not f.answered or not f.final_sql.strip():
        return "none (no usable query)"
    try:
        cols, _ = fetch_rows(f"SELECT * FROM ({f.final_sql.strip().rstrip(';')}) LIMIT 0", 1)
        return ", ".join(cols)
    except ValueError as e:
        return f"none (its query does not run: {str(e)[:120]})"


def _evidence(findings: list[Finding], verification: Verification | None, rec: Recorder,
              ctx: "visuals.WriterContext") -> list[str]:
    """Everything a number in the answer may legitimately come from."""
    texts = [f.model_dump_json() for f in findings] + rec.query_results
    if verification:
        texts.append(verification.model_dump_json())
    for v in ctx.visuals:
        texts.append(str(v.get("rows", "")))
        texts.append(str(v.get("spec", {}).get("data", {}).get("values", "")))
        texts.append(str(v.get("total_rows", "")))
    return texts


def _fallback_answer(findings: list[Finding]) -> str:
    """Used only if the Writer fails: the findings, plainly."""
    if not findings:
        return "Sorry, I couldn't complete this request in time. Please try again or narrow the question."
    lines = ["I ran out of time writing a full answer. Here is what the data showed:\n"]
    for f in findings:
        lines.append(f"- {f.answer}" + (f" ({f.sample_size})" if f.sample_size else ""))
        lines += [f"  - Caveat: {c}" for c in f.caveats[:2]]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

async def _run_worker(team, sq: SubQuestion, message: str, plan: Plan, rec: Recorder,
                      sem: asyncio.Semaphore) -> Finding:
    agent = team["worker"].clone(name=f"SQL Worker {sq.id}")
    async with sem:
        try:
            result = await Runner.run(agent, _worker_input(message, plan, sq), max_turns=WORKER_TURNS, hooks=rec,
                                      error_handlers={"max_turns": _salvage("worker", rec)})
        except MaxTurnsExceeded:
            rec.note(agent.name, f"Stopped at the {WORKER_TURNS}-turn limit before submitting a finding.")
            return _unanswered(sq, "the analysis did not finish within its step limit")
    finding = _parse(Finding, result.final_output)
    if finding is None:
        rec.note(agent.name, "Ended without a valid submit_finding; treated as unanswered.")
        return _unanswered(sq, "the analysis ended without a usable result")
    finding.sub_question_id = sq.id
    return finding


def _drop_restart(text: str) -> str:
    """If the model started the answer, stopped, and started over, keep only the last start."""
    head = text[:80]
    if len(head) < 40:
        return text
    i = text.rfind(head)
    return text[i:] if i > 0 else text


async def _run_workers(team, sub_questions: list[SubQuestion], message: str, plan: Plan, rec: Recorder,
                       budget_s: float) -> list[Finding]:
    """All sub-questions in parallel, each in a fresh context. Keeps whatever finishes in time."""
    sem = asyncio.Semaphore(MAX_PARALLEL_WORKERS)
    tasks = {asyncio.create_task(_run_worker(team, sq, message, plan, rec, sem)): sq for sq in sub_questions}
    done, pending = await asyncio.wait(tasks, timeout=max(budget_s, 1))
    findings, errors = [], []
    for task, sq in tasks.items():
        if task in pending:
            task.cancel()
            rec.note(f"SQL Worker {sq.id}", f"Ran out of time ({budget_s:.0f}s budget).")
            findings.append(_unanswered(sq, "it ran out of time"))
        elif task.exception():
            errors.append(task.exception())
            rec.note(f"SQL Worker {sq.id}", f"Failed: {type(task.exception()).__name__}: {str(task.exception())[:200]}")
            findings.append(_unanswered(sq, "an error occurred while querying"))
        else:
            findings.append(task.result())
    if errors and len(errors) == len(sub_questions):
        raise errors[0]  # every worker failed the same way (e.g. auth): surface it, don't hide it
    return sorted(findings, key=lambda f: f.sub_question_id)


async def run_pipeline(message: str, history: list[dict] | None = None,
                       team: dict[str, Agent] | None = None,
                       on_event: Callable[[dict], None] | None = None) -> PipelineResult:
    """Answer one message. on_event, if given, receives progress events as they happen."""
    team = team or default_team()
    rec = Recorder(on_event)
    start = time.monotonic()
    timings: dict[str, float] = {}

    def remaining() -> float:
        return DEADLINE_S - (time.monotonic() - start)

    def mark(stage: str, t0: float) -> None:
        timings[stage] = round(time.monotonic() - t0, 1)

    # 1. Plan
    t0 = time.monotonic()
    rec.stage("plan", "start")
    try:
        r = await asyncio.wait_for(
            Runner.run(team["planner"], (history or []) + [{"role": "user", "content": message}],
                       max_turns=PLANNER_TURNS, hooks=rec, error_handlers={"max_turns": _salvage("planner", rec)}),
            timeout=min(PLANNER_MAX_S, remaining() - PRESENTER_RESERVE_S))
        plan = _parse(Plan, r.final_output) or _fallback_plan(message, "planner gave no usable plan")
    except (asyncio.TimeoutError, MaxTurnsExceeded) as e:
        rec.note("Planner", f"Planning did not finish ({type(e).__name__}); answering the question as asked.")
        plan = _fallback_plan(message, "planning did not finish")
    plan = _clean_plan(plan, message)
    mark("plan", t0)
    plan_view = {"action": plan.action, "interpretation": plan.interpretation,
                 "sub_questions": [sq.question for sq in plan.sub_questions] if plan.action == "answer" else []}
    rec.stage("plan", "done", **plan_view)

    meta = {"action": plan.action, "complexity": plan.complexity, "sub_questions": len(plan.sub_questions),
            "verdict": None, "follow_ups": 0, "plan": plan_view}

    def done(response: str) -> PipelineResult:
        meta.update(seconds=round(time.monotonic() - start, 1), timings=timings, tokens=rec.token_summary())
        return PipelineResult(response=response, tool_calls=rec.tool_calls, meta=meta)

    # 2. Clarify: ask and stop.
    if plan.action == "clarify":
        return done(plan.clarifying_question)

    findings: list[Finding] = []
    verification: Verification | None = None
    unverified: set[int] = set()

    if plan.action == "answer":
        # 3. Workers, in parallel.
        t0 = time.monotonic()
        rec.stage("workers", "start", count=len(plan.sub_questions))
        findings = await _run_workers(team, plan.sub_questions, message, plan, rec,
                                      budget_s=remaining() - PRESENTER_RESERVE_S - VERIFIER_RESERVE_S)
        mark("workers", t0)
        rec.stage("workers", "done", answered=sum(f.answered for f in findings), count=len(findings))

        # 4. Verify (and at most one follow-up round).
        if SKIP_VERIFIER_FOR_SIMPLE and plan.complexity == "simple":
            rec.note("Verifier", "Skipped for a simple question (SKIP_VERIFIER_FOR_SIMPLE is on).")
        else:
            t0 = time.monotonic()
            rec.stage("verify", "start")
            try:
                r = await asyncio.wait_for(
                    Runner.run(team["verifier"], _verifier_input(message, plan, findings, rec),
                               max_turns=VERIFIER_TURNS, hooks=rec,
                               error_handlers={"max_turns": _salvage("verifier", rec)}),
                    timeout=min(VERIFIER_MAX_S, remaining() - PRESENTER_RESERVE_S))
                verification = _parse(Verification, r.final_output)
            except (asyncio.TimeoutError, MaxTurnsExceeded) as e:
                rec.note("Verifier", f"Verification did not finish ({type(e).__name__}); findings go unverified.")
            mark("verify", t0)
            meta["verdict"] = verification.verdict if verification else "skipped"
            rec.stage("verify", "done", verdict=meta["verdict"],
                      issues=len(verification.issues) if verification else 0)

            follow_ups = (verification.follow_up_questions[:MAX_FOLLOW_UPS]
                          if verification and verification.verdict == "needs_more" else [])
            budget = remaining() - PRESENTER_RESERVE_S
            if follow_ups and budget > 30:
                next_id = max(f.sub_question_id for f in findings) + 1
                for i, sq in enumerate(follow_ups):
                    sq.id = next_id + i
                plan.sub_questions += follow_ups
                t0 = time.monotonic()
                rec.stage("follow_ups", "start", count=len(follow_ups),
                          sub_questions=[sq.question for sq in follow_ups])
                extra = await _run_workers(team, follow_ups, message, plan, rec, budget_s=budget)
                mark("follow_ups", t0)
                rec.stage("follow_ups", "done", answered=sum(f.answered for f in extra))
                findings += extra
                unverified = {f.sub_question_id for f in extra}
                meta["follow_ups"] = len(extra)
            elif follow_ups:
                rec.note("Verifier", "Follow-up checks skipped: not enough time left.")

    # 5. Write. The Writer picks tables/charts/maps; code builds them from the findings' verified SQL.
    t0 = time.monotonic()
    rec.stage("write", "start")
    ctx = visuals.WriterContext(findings={f.sub_question_id: f for f in findings if f.answered})
    presenter_input = _presenter_input(message, plan, findings, verification, unverified)
    try:
        r = await asyncio.wait_for(
            Runner.run(team["presenter"], presenter_input, context=ctx, max_turns=PRESENTER_TURNS, hooks=rec),
            timeout=max(remaining() - 5, 10))
        # Keep all the Writer's text: if it writes part of the answer in the same turn as a tool call,
        # final_output holds only the text after it (the first smoke run lost a whole answer this way).
        response = _drop_restart(ItemHelpers.text_message_outputs(r.new_items).strip())
    except (asyncio.TimeoutError, MaxTurnsExceeded) as e:
        rec.note("Writer", f"Writing did not finish ({type(e).__name__}); returning the findings directly.")
        response = ""
    mark("write", t0)
    rec.stage("write", "done")

    # 6. Numbers check: every number in the prose must appear in the evidence. One revision if not.
    if response and findings:
        bad = grounding.unsupported_numbers(response, _evidence(findings, verification, rec, ctx))
        meta["unsupported_numbers"] = {"first_draft": bad, "final": bad}
        if bad and remaining() > REVISION_MIN_S:
            rec.note("Writer", f"Numbers not found in the evidence: {', '.join(bad)}. Asked for one revision.")
            t0 = time.monotonic()
            rec.stage("revise", "start", numbers=bad)
            feedback = (f"These numbers in your answer do not appear in the findings or query results: "
                        f"{', '.join(bad)}. Replace each with the exact number from the findings, or remove it. "
                        "Do not compute new numbers (ratios, ranges, sums, differences). Keep every "
                        "<!--gq:...--> marker exactly where it is. Return the complete corrected answer.")
            try:
                r = await asyncio.wait_for(
                    Runner.run(team["presenter"].clone(tools=[]),
                               [{"role": "user", "content": presenter_input},
                                {"role": "assistant", "content": response},
                                {"role": "user", "content": feedback}],
                               context=ctx, max_turns=2, hooks=rec),
                    timeout=max(remaining() - 5, 10))
                # No tools here, so the answer is the last message. Smoke run 6 got a cut-off copy
                # followed by the full answer when every message was joined.
                messages = [i for i in r.new_items if isinstance(i, MessageOutputItem)]
                revised = _drop_restart(ItemHelpers.text_message_output(messages[-1]).strip()) if messages else ""
                if revised:
                    meta["first_draft"] = response  # kept so the catch can be audited
                    response = revised
                    meta["unsupported_numbers"]["final"] = grounding.unsupported_numbers(
                        response, _evidence(findings, verification, rec, ctx))
            except Exception as e:  # noqa: BLE001 -- a failed revision must never sink the answer
                rec.note("Writer", f"Revision did not finish ({type(e).__name__}); kept the first draft.")
            mark("revise", t0)
            rec.stage("revise", "done")
            still = meta["unsupported_numbers"]["final"]
            if still:
                # Smoke run 7: the revision swapped "23%" for "143 projects", another computed number.
                # Remove those sentences outside the opening answer rather than ship them.
                response, dropped = grounding.drop_unsupported_sentences(response, still)
                meta["unsupported_numbers"]["final"] = grounding.unsupported_numbers(
                    response, _evidence(findings, verification, rec, ctx))
                rec.note("Writer", f"Still unsupported after revision: {', '.join(still)}. "
                         f"Removed {dropped} sentence(s) containing them.")

    response = visuals.place_visuals(response, ctx) if response else _fallback_answer(findings)
    visuals.attach_to_tool_calls(rec.tool_calls, ctx)
    meta["visuals"] = [v["kind"] for v in ctx.visuals]
    return done(response)
