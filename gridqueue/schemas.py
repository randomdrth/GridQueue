"""What each agent hands to the next one.

Agents return these by calling a submit_* tool whose arguments are the model below. Function
calling is the most reliable structured-output path across models, and each submit call
shows up in the /chat tool_calls, so the plan, findings and verification are visible.

Kept deliberately simple for Gemini's schema support: strings, ints, enums, flat lists.
"""

from typing import Literal

from pydantic import BaseModel, Field


class SubQuestion(BaseModel):
    id: int = Field(description="1, 2, 3... Unique within the request.")
    question: str = Field(description="Self-contained: names the metric, filters and grouping. "
                                      "No references like 'that' or 'the previous one'.")
    hint: str = Field(description="Tables and columns likely needed, e.g. 'projects: q_status, type_group, q_year'.")


class Plan(BaseModel):
    action: Literal["answer", "clarify", "out_of_scope"]
    complexity: Literal["simple", "complex"] = Field(
        description="simple: one sub-question answers it. complex: needs 2-4 sub-questions.")
    interpretation: str = Field(description="How you read the question, with every assumption made "
                                            "(e.g. 'built = reached operational status').")
    clarifying_question: str = Field(description="The question to ask the user if action is clarify, else ''.")
    out_of_scope_reason: str = Field(description="What the data lacks if action is out_of_scope, else ''.")
    sub_questions: list[SubQuestion] = Field(description="1 for simple, 2-4 for complex. Empty unless action is answer.")


class Finding(BaseModel):
    sub_question_id: int
    answered: bool = Field(description="False if the data could not answer it.")
    answer: str = Field(description="1-3 sentences with the exact numbers from your query results, "
                                    "or why it could not be answered.")
    table_columns: list[str] = Field(description="Column names of the key result table, or [] if none.")
    table_rows: list[list[str]] = Field(description="Up to 20 rows of that table, every cell as text, "
                                                    "copied from run_sql results.")
    final_sql: str = Field(description="The query that produced the answer.")
    sample_size: str = Field(description="How many records the answer rests on, e.g. '3,289 of 4,789 "
                                         "operational projects have an operation date'. '' if not relevant.")
    caveats: list[str] = Field(description="Caveats from run_sql and get_data_gaps that apply to this answer.")
    confidence: Literal["high", "medium", "low"]


class Issue(BaseModel):
    sub_question_id: int
    problem: str


class Verification(BaseModel):
    verdict: Literal["ok", "needs_more"]
    checks: list[str] = Field(description="What you checked, one line each.")
    issues: list[Issue] = Field(description="Problems found. [] if none.")
    corrections: list[str] = Field(description="Facts the writer must use instead of a finding's claim, "
                                               "e.g. 'Finding 2: the count is 412, not 421 (re-ran the query)'.")
    follow_up_questions: list[SubQuestion] = Field(
        description="At most 2 new self-contained sub-questions that would fix a gap. [] if verdict is ok.")
