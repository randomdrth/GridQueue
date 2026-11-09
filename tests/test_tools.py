"""Tests for the agent tools. No model calls: these exercise the plain functions directly.

Run: uv run pytest
"""

import json

import pytest

from gridqueue import guardrails, tools


def call(fn, *args, **kwargs) -> dict:
    return json.loads(fn(*args, **kwargs))


# --- Layer 1: rejected before running ----------------------------------------

def test_invalid_enum_value_rejected_with_valid_values():
    r = call(tools.run_sql, "SELECT count(*) FROM projects WHERE q_status = 'complete'", "t")
    assert "Rejected" in r["error"]
    assert "operational" in r["error"] and "withdrawn" in r["error"]


def test_wrong_case_gets_did_you_mean():
    r = call(tools.run_sql, "SELECT count(*) FROM projects WHERE q_status = 'Withdrawn'", "t")
    assert "Did you mean 'withdrawn'" in r["error"]


def test_invalid_value_inside_in_list_rejected():
    r = call(tools.run_sql, "SELECT count(*) FROM projects WHERE region IN ('PJM', 'New York')", "t")
    assert "New York" in r["error"] and "NYISO" in r["error"]


def test_valid_enum_values_pass():
    assert guardrails.check_sql(
        "SELECT count(*) FROM projects p WHERE p.q_status <> 'withdrawn' AND region IN ('PJM','MISO')") is None


def test_alias_with_enum_name_not_checked():
    # The query defines its own 'region' column, so 'Northeast' is not a source value to validate.
    q = ("WITH t AS (SELECT CASE WHEN region IN ('NYISO','ISO-NE') THEN 'Northeast' ELSE 'Other' END AS region "
         "FROM projects) SELECT count(*) FROM t WHERE region = 'Northeast'")
    assert guardrails.check_sql(q) is None


@pytest.mark.parametrize("q", [
    "SELECT 1; SELECT 2",
    "CREATE TABLE x AS SELECT 1",
    "ATTACH 'other.db'",
    "COPY projects TO 'out.csv'",
    "INSTALL httpfs",
])
def test_non_select_rejected(q):
    assert "Rejected" in call(tools.run_sql, q, "t")["error"]


# --- Running queries -----------------------------------------------------------

def test_known_answer_status_counts():
    r = call(tools.run_sql, "SELECT q_status, count(*) FROM projects GROUP BY 1 ORDER BY 1", "t")
    assert dict(r["rows"]) == {"active": 8513, "operational": 4789, "suspended": 668,
                               "unknown": 10, "withdrawn": 24221}


def test_known_answer_pjm_a03_cost():
    r = call(tools.run_sql, "SELECT total_cost_per_kw_2024 FROM costs WHERE entity='PJM' AND cost_q_id='A03'", "t")
    assert r["rows"] == [[13.72]]


def test_large_result_truncated_with_true_count():
    r = call(tools.run_sql, "SELECT project_uid FROM projects", "t")
    assert r["truncated"] is True
    assert r["row_count"] == 38201
    assert len(r["rows"]) == tools.MAX_ROWS


def test_result_size_capped():
    text = tools.run_sql("SELECT * FROM projects", "t")
    assert len(text) <= tools.MAX_RESULT_CHARS + 500
    assert json.loads(text)["row_count"] == 38201


def test_timeout_stops_slow_query(monkeypatch):
    monkeypatch.setattr(tools, "QUERY_TIMEOUT_S", 0.3)
    r = call(tools.run_sql, "SELECT count(*) FROM range(100000000000)", "t")
    assert "stopped" in r["error"]


def test_sql_error_is_returned_not_raised():
    r = call(tools.run_sql, "SELECT no_such_column FROM projects", "t")
    assert "Binder" in r["error"] and "hint" in r


# --- Layer 2: caveats ----------------------------------------------------------

def test_timing_query_gets_date_caveats():
    r = call(tools.run_sql, "SELECT median(days_ir_to_cod) FROM projects WHERE q_status='operational'", "t")
    text = " ".join(r["caveats"])
    assert "[dates]" in text and "[sparse columns]" in text


def test_cost_query_gets_recency_caveat():
    r = call(tools.run_sql, "SELECT avg(total_cost_per_kw_2024) FROM costs", "t")
    assert any("[cost recency]" in c for c in r["caveats"])


def test_direct_eia_join_gets_fanout_warning():
    q = ("SELECT sum(p.mw_total) FROM projects p JOIN eia_generators g ON p.fips_code = g.fips_code "
         "WHERE p.q_status = 'active'")
    assert any("[fan-out]" in c for c in call(tools.run_sql, q, "t")["caveats"])


def test_simple_query_has_no_noise():
    r = call(tools.run_sql, "SELECT region, count(*) FROM projects GROUP BY 1", "t")
    assert r["caveats"] == []


def test_zero_rows_gets_hint():
    r = call(tools.run_sql, "SELECT * FROM projects WHERE mw_1 > 1e9", "t")
    assert r["row_count"] == 0 and any("[no rows]" in c for c in r["caveats"])


# --- Context tools -------------------------------------------------------------

def test_list_tables():
    assert [t["table"] for t in call(tools.list_tables)["tables"]] == \
        ["projects", "costs", "eia_generators", "county_summary", "poi_summary"]


def test_describe_table_unknown_lists_tables():
    assert "projects" in call(tools.describe_table, "nope")["tables"]


def test_describe_table_includes_joins_and_fill():
    r = call(tools.describe_table, "costs")
    assert any(j["to"] == "projects.project_uid" for j in r["joins"])
    assert all("fill_pct" in c for c in r["columns"])


def test_column_values_exact_list():
    r = call(tools.get_column_values, "projects", "q_status")
    assert r["values"]["withdrawn"] == 24221


def test_column_values_fuzzy_search():
    r = call(tools.get_column_values, "projects", "poi_name_clean", search="hassayampa")
    assert r["matches"][0]["value"].startswith("Hassayampa")


def test_column_values_unknown_column():
    assert "describe_table" in call(tools.get_column_values, "projects", "nope")["error"]


def test_data_gaps_data_centers():
    r = call(tools.get_data_gaps, "data centers")
    assert r["gaps"][0]["topic"] == "data centers / load"


def test_data_gaps_includes_region_fill_for_named_column():
    r = call(tools.get_data_gaps, "developer by region")
    assert "fill_pct_by_region" in r
    assert {row["region"] for row in r["fill_pct_by_region"]} >= {"PJM", "ERCOT"}


def test_tool_wrappers_have_schemas():
    names = {t.name for t in tools.ALL_TOOLS}
    assert names == {"list_tables", "describe_table", "get_column_values", "get_data_gaps", "run_sql"}
    run_sql_params = tools.run_sql_tool.params_json_schema["properties"]
    assert set(run_sql_params) == {"query", "purpose"}
