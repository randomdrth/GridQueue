"""The plain baseline agent: same database, bare run_sql, table and column names only."""

import json

from gridqueue import baseline


def test_schema_listing_names_every_table():
    listing = baseline.schema_listing()
    for table in ("projects", "costs", "eia_generators", "county_summary", "poi_summary"):
        assert f"- {table}(" in listing
    assert "q_status VARCHAR" in listing


def test_plain_run_sql_has_no_guardrails():
    # A misspelled category just returns zero rows, with no hint: that's the point of the baseline.
    out = json.loads(baseline.plain_run_sql("SELECT count(*) FROM projects WHERE q_status = 'Operating'"))
    assert out["rows"] == [[0]]
    assert "error" in json.loads(baseline.plain_run_sql("SELECT nope FROM projects"))


def test_plain_agent_has_one_tool():
    agent = baseline.plain_agent()
    assert [t.name for t in agent.tools] == ["run_sql"]
    assert "county_summary(" in agent.instructions
