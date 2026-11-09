"""Tables, charts and maps built from verified SQL, and the numbers check. No model calls."""

import pytest

from gridqueue import grounding
from gridqueue import visuals as V
from gridqueue.schemas import Finding

COST_SQL = """SELECT p.region, p.q_status, round(median(c.total_cost_per_kw_2024), 2) AS median_cost,
                     round(avg(c.total_cost_per_kw_2024), 2) AS mean_cost
              FROM projects p JOIN costs c USING (project_uid)
              WHERE c.is_primary_estimate AND p.q_status IN ('withdrawn', 'operational')
              GROUP BY ALL ORDER BY 1, 2"""
COUNTY_SQL = "SELECT fips_code, county, state, active_queue_mw FROM county_summary WHERE active_queue_mw > 0"


def finding(i, sql, answered=True):
    return Finding(sub_question_id=i, answered=answered, answer="", table_columns=[], table_rows=[],
                   final_sql=sql, sample_size="", caveats=[], confidence="high")


@pytest.fixture
def ctx():
    return V.WriterContext(findings={1: finding(1, COST_SQL), 2: finding(2, COUNTY_SQL),
                                     3: finding(3, "", answered=False)})


# --- tables -----------------------------------------------------------------

def test_table_numbers_come_from_the_database(ctx):
    # The smoke run's Writer swapped SPP's mean and median. A code-built table cannot.
    V._table(ctx, 1, ["region", "q_status", "median_cost", "mean_cost"], ["Region", "Status", "Median", "Mean"], 30, "")
    spp = next(r for r in ctx.visuals[0]["rows"] if r[:2] == ["SPP", "operational"])
    assert spp[2:] == [30.48, 51.35]
    assert "| SPP | operational | 30.48 | 51.35 |" in ctx.visuals[0]["markdown"]


def test_table_unknown_column_lists_real_ones(ctx):
    with pytest.raises(ValueError, match="Available columns"):
        V._table(ctx, 1, ["median_cost_per_kw"], ["x"], 5, "")


def test_unanswered_finding_cannot_be_drawn(ctx):
    with pytest.raises(ValueError, match="no usable query"):
        V._table(ctx, 3, ["x"], ["x"], 5, "")


def test_tool_errors_are_returned_as_text(ctx):
    assert V._safe(V._table, ctx, 9, ["x"], ["x"], 5, "").startswith("Error: No finding 9")


# --- charts and maps ----------------------------------------------------------

def test_grouped_bar_chart(ctx):
    V._chart(ctx, 1, "bar", "region", ["median_cost"], "q_status", "t", "Region", "$/kW")
    spec = ctx.visuals[0]["spec"]
    assert spec["mark"]["type"] == "bar" and spec["encoding"]["xOffset"] == {"field": "series"}
    assert spec["params"][0]["bind"] == "legend"            # click legend to highlight
    assert {"category": "SPP", "value": 30.48, "series": "operational", "label": "SPP"} in spec["data"]["values"]


def test_chart_rejects_non_numeric_y(ctx):
    with pytest.raises(ValueError, match="not numeric"):
        V._chart(ctx, 1, "bar", "region", ["q_status"], "", "t", "x", "y")


def test_line_chart_zooms(ctx):
    ctx.findings[4] = finding(4, "SELECT q_year, count(*) AS n FROM projects WHERE q_year >= 2010 GROUP BY 1 ORDER BY 1")
    V._chart(ctx, 4, "line", "q_year", ["n"], "", "t", "Year", "Requests")
    assert ctx.visuals[0]["spec"]["params"][0] == {"name": "zoom", "select": "interval", "bind": "scales"}


def test_map_needs_fips(ctx):
    with pytest.raises(ValueError, match="fips_code"):
        V._map(ctx, 1, "median_cost", "t", "v")


def test_county_map(ctx):
    V._map(ctx, 2, "active_queue_mw", "Active queue", "MW")
    v = ctx.visuals[0]
    assert v["counties"] > 1500
    data = v["spec"]["layer"][1]["data"]["values"]
    assert all(len(d["fips"]) == 5 for d in data)
    assert v["spec"]["layer"][1]["encoding"]["color"]["scale"]["type"] == "threshold"


def test_class_breaks_are_rounded_and_increasing():
    breaks = V.class_breaks([1, 5, 12, 140, 145, 290, 580, 1234, 5678, 9000])
    assert breaks == sorted(breaks) and all(b == V._round_sig(b) for b in breaks)


# --- placing visuals -------------------------------------------------------------

def test_place_visuals(ctx):
    V._table(ctx, 1, ["region", "median_cost"], ["Region", "Median"], 20, "Costs")
    V._map(ctx, 2, "active_queue_mw", "Active queue", "MW")
    out = V.place_visuals("Answer.\n\n<!--gq:table:1-->\n\n**How I got this**\n- data", ctx)
    assert "**Costs**" in out and "| Region | Median |" in out
    assert out.index("<!--gq:map:2-->") < out.index("**How I got this**")   # forgotten map placed anyway
    assert "<!--gq:table:1-->" in out                                       # kept for the page's CSV/SQL buttons


def test_attach_visuals_to_tool_calls(ctx):
    result = V._table(ctx, 1, ["region"], ["Region"], 20, "")
    calls = [{"name": "show_table", "result": result}, {"name": "run_sql", "result": "{}"}]
    V.attach_to_tool_calls(calls, ctx)
    assert calls[0]["visual"]["kind"] == "table" and "visual" not in calls[1]


# --- numbers check -------------------------------------------------------------

EVIDENCE = ['{"rows": [["PJM", 1134, 130610.82], ["SPP", 51.35, 30.48]], "ratio_max": 6.17, "ratio_min": 1.77}',
            '{"share_operational": 0.304}']


@pytest.mark.parametrize("prose", [
    "PJM has 1,134 active projects.",
    "That is 130.61 GW (130,610.8 MW).",          # MW -> GW, rounding
    "30.4% of gas projects were built.",          # share -> percent
    "Withdrawn costs are 2 to 6 times higher.",   # honest rounding of 1.77 and 6.17
    "In 2023 the queue grew.",                    # years are skipped
    "| 999 | 888 |",                              # code-rendered table lines are skipped
])
def test_supported_numbers_pass(prose):
    assert grounding.unsupported_numbers(prose, EVIDENCE) == []


@pytest.mark.parametrize("prose,bad", [
    ("PJM has 1,143 active projects.", ["1,143"]),            # transposed digits
    ("Costs are about 3 times higher.", ["3"]),               # invented ratio
    ("SPP's median is $41.20.", ["41.20"]),                    # invented value
])
def test_unsupported_numbers_flagged(prose, bad):
    assert grounding.unsupported_numbers(prose, EVIDENCE) == bad


RATE_SQL = """SELECT type_group, round(count(*) FILTER (WHERE q_status='operational') / count(*), 3) AS completion_rate
              FROM projects WHERE q_year BETWEEN 2000 AND 2020 AND type_group IN ('Gas','Solar','Wind')
              GROUP BY 1 ORDER BY 2 DESC"""


def test_rates_shown_as_percents(ctx):
    ctx.findings[5] = finding(5, RATE_SQL)
    V._table(ctx, 5, ["type_group", "completion_rate"], ["Type", "Completion rate"], 5, "")
    assert "| Gas | 30.4% |" in ctx.visuals[0]["markdown"]
    V._chart(ctx, 5, "bar", "type_group", ["completion_rate"], "", "t", "Type", "Completion rate")
    spec = ctx.visuals[1]["spec"]
    assert spec["encoding"]["y"]["axis"]["format"] == ".0%"


def test_plain_numbers_not_turned_into_percents(ctx):
    V._table(ctx, 1, ["region", "median_cost"], ["Region", "Median"], 20, "")
    assert "%" not in ctx.visuals[0]["markdown"]


# --- per-column formatting (smoke run 5) ---------------------------------------------------------

def test_column_formatting_is_consistent():
    from gridqueue.visuals import render_markdown_table
    md = render_markdown_table(
        ["Type", "Projects", "Capacity (MW)", "Year", "Cost"],
        [["Solar", 8616, 734375.9, 2020, 13.95], ["Gas+Battery", 18, 7218.0, 2021, 12.0]],
        names=["type", "n_projects", "mw", "on_year", "cost"])
    rows = md.splitlines()[2:]
    assert "8,616" in rows[0] and "2020" in rows[0] and "2,020" not in rows[0]
    assert "7,218.0" in rows[1] and "12.00" in rows[1]


def test_percent_point_columns_get_a_percent_sign():
    from gridqueue.visuals import render_markdown_table
    md = render_markdown_table(["Type", "Projects Built (%)", "Total MW"],
                               [["Solar", 13.35, 734375.9], ["Gas", 30.44, 846966.2]],
                               names=["type", "pct_built", "mw_total"])
    assert "13.3%" in md or "13.4%" in md
    assert "30.4%" in md and "734,375.9" in md


def test_model_calls_have_a_timeout():
    from gridqueue.roles import _settings, MODEL_CALL_TIMEOUT_S
    assert _settings("auto").extra_args["timeout"] == MODEL_CALL_TIMEOUT_S <= 60


def test_table_refuses_a_cut_that_hides_a_whole_group():
    from gridqueue.visuals import _groups_cut_off
    rows = [["ERCOT", y, 1] for y in range(2005, 2026)] + [["PJM", y, 1] for y in range(2005, 2026)]
    assert _groups_cut_off(rows, 0, 16) == ["PJM"]
    assert _groups_cut_off(rows, 0, 42) == []
    ranking = [[f"County {i}", i] for i in range(40)]
    assert _groups_cut_off(ranking, 0, 10) == []  # a ranking may be cut to the top N


def test_dataset_names_are_not_numbers():
    from gridqueue.grounding import unsupported_numbers
    assert unsupported_numbers("Sources: LBNL Queued Up 2026, EIA-860 2025.", ["1,134"]) == []


def test_drop_unsupported_sentences_keeps_the_lead():
    from gridqueue.grounding import drop_unsupported_sentences
    prose = ("PJM's median is 1,150 days and ERCOT's is 1,350.\n\n**Limits**\n"
             "* **Data completeness**: Dates are missing for some projects. That leaves 143 in ERCOT.\n"
             "* **Timing bias**: Only finished projects count.\n"
             "* **Gaps**: 11 projects lack dates.")
    out, n = drop_unsupported_sentences(prose, ["143", "11"])
    assert n == 2 and "143" not in out and "11 projects" not in out
    assert "Dates are missing for some projects." in out and "Timing bias" in out
    assert out.startswith("PJM's median is 1,150 days")
    assert "**Gaps**" not in out
