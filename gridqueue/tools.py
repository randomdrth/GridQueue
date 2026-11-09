"""The five tools the GridQueue agents use.

Each is a plain function (testable without a model) plus a function_tool wrapper at the
bottom. All return JSON strings: the model reads text, and an error it can read is better
than an exception it can't see.
"""

import datetime as dt
import decimal
import json
import re
import threading

import duckdb
from agents import function_tool

from gridqueue import guardrails
from gridqueue.db import COLUMNS, COVERAGE, GAPS, JOINS, SCHEMA, TABLES, cursor

MAX_ROWS = 200
MAX_RESULT_CHARS = 20_000
QUERY_TIMEOUT_S = 15
MAX_VALUE_LIST = 60


def _dumps(obj) -> str:
    return json.dumps(obj, default=str)


def _error(message: str, **extra) -> str:
    return _dumps({"error": message, **extra})


def _clean(v):
    """JSON-friendly cell: round floats, ISO dates, decimals to floats."""
    if isinstance(v, decimal.Decimal):
        v = float(v)
    if isinstance(v, float):
        return round(v, 3) if abs(v) < 1e7 else round(v)
    if isinstance(v, (dt.date, dt.datetime)):
        return v.isoformat()
    return v


def _run_with_timeout(cur: duckdb.DuckDBPyConnection, sql: str, params=None, fetch: int | None = None):
    """Execute with an interrupt after QUERY_TIMEOUT_S, so one slow query can't eat the request."""
    timer = threading.Timer(QUERY_TIMEOUT_S, cur.interrupt)
    timer.start()
    try:
        cur.execute(sql, params or [])
        return cur.fetchmany(fetch) if fetch else cur.fetchall()
    finally:
        timer.cancel()


# ---------------------------------------------------------------------------
# 1. list_tables
# ---------------------------------------------------------------------------

def list_tables() -> str:
    """List the database's tables: what each row is, row count, primary key and source.

    Call this first. Then call describe_table for the tables you need.
    """
    return _dumps({
        "dollar_basis": SCHEMA["dollar_basis"],
        "tables": [{"table": t, "description": i["description"], "grain": i["grain"],
                    "rows": i["rows"], "primary_key": i["primary_key"], "source": i["source"]}
                   for t, i in TABLES.items()],
    })


# ---------------------------------------------------------------------------
# 2. describe_table
# ---------------------------------------------------------------------------

def describe_table(table: str) -> str:
    """Describe one table's columns and how it joins to the others.

    Each column has its type, a description, fill_pct (share of rows that are not null) and
    has_value_list (true if it takes a fixed set of values: call get_column_values for them).

    Args:
        table: Table name from list_tables, e.g. 'projects'.
    """
    if table not in TABLES:
        return _error(f"No table '{table}'.", tables=list(TABLES))
    cols = [{"name": c["name"], "type": c["type"], "description": c["description"],
             "fill_pct": c["fill_pct"], "has_value_list": "values" in c and c["type"] == "VARCHAR",
             **({"min": c["min"], "max": c["max"]} if "min" in c else {})}
            for c in TABLES[table]["columns"]]
    joins = [j for j in JOINS if re.search(rf"\b{re.escape(table)}\.", j["from"] + " " + j["to"])]
    return _dumps({"table": table, "grain": TABLES[table]["grain"], "rows": TABLES[table]["rows"],
                   "columns": cols, "joins": joins})


# ---------------------------------------------------------------------------
# 3. get_column_values
# ---------------------------------------------------------------------------

def get_column_values(table: str, column: str, search: str | None = None) -> str:
    """Get the exact values a column takes, or find values matching a search term.

    Use before filtering on any text column: values are case-sensitive and must match exactly.
    - Without search: every value with its row count (fixed-list columns), or the most common
      values (free-text columns like poi_name_clean, county, developer).
    - With search: the closest matches by name, for free text. E.g. search='hassayampa' on
      projects.poi_name_clean finds 'Hassayampa 500 kV'.

    Args:
        table: Table name, e.g. 'projects'.
        column: Column name, e.g. 'q_status'.
        search: Optional text to match against the column's values.
    """
    if table not in COLUMNS:
        return _error(f"No table '{table}'.", tables=list(TABLES))
    if column not in COLUMNS[table]:
        return _error(f"No column '{column}' in {table}. Call describe_table('{table}').")
    info = COLUMNS[table][column]
    # table and column are checked against the schema above, so quoting them is safe.
    tbl, col = f'"{table}"', f'"{column}"'

    if info["type"] != "VARCHAR":
        return _dumps({"table": table, "column": column, "type": info["type"],
                       "fill_pct": info["fill_pct"], "min": info.get("min"), "max": info.get("max"),
                       "note": "Not a text column: filter with numeric or date comparisons."})

    with cursor() as cur:
        if search:
            rows = _run_with_timeout(cur, f"""
                SELECT {col} AS value, count(*) AS n,
                       greatest(jaro_winkler_similarity(lower({col}), lower($1)),
                                CASE WHEN contains(lower({col}), lower($1)) THEN 1.0 ELSE 0 END) AS score
                FROM {tbl} WHERE {col} IS NOT NULL
                GROUP BY 1 ORDER BY score DESC, n DESC LIMIT 15""", [search])
            matches = [{"value": v, "rows": n, "score": round(s, 2)} for v, n, s in rows if s >= 0.7]
            return _dumps({"table": table, "column": column, "search": search, "matches": matches,
                           **({} if matches else {"note": "No close matches. Try a shorter or different term."})})

        total = info["distinct_values"]
        rows = _run_with_timeout(cur, f"""
            SELECT {col}, count(*) FROM {tbl} WHERE {col} IS NOT NULL
            GROUP BY 1 ORDER BY 2 DESC LIMIT {MAX_VALUE_LIST}""")
    out = {"table": table, "column": column, "fill_pct": info["fill_pct"], "distinct_values": total,
           "values": {v: n for v, n in rows}}
    if total > MAX_VALUE_LIST:
        out["note"] = (f"Free-text column with {total} distinct values; showing the {MAX_VALUE_LIST} most "
                       f"common. Use search= to find a specific one.")
    return _dumps(out)


# ---------------------------------------------------------------------------
# 4. get_data_gaps
# ---------------------------------------------------------------------------

def get_data_gaps(topic: str) -> str:
    """Find what the data does NOT cover, or covers only partly, for a topic or columns.

    Call before answering anything about timing, developers, costs, data centers, hybrids,
    locations, or a specific region, and whenever a result seems surprisingly small.
    Returns matching caveats and, for date and text columns, fill rates by region.

    Args:
        topic: Words or column names, e.g. 'on_date PJM', 'costs ERCOT', 'data centers'.
    """
    words = {w.lower() for w in re.findall(r"[A-Za-z_][A-Za-z_0-9-]+", topic) if len(w) > 2}
    def score(g):
        text = f"{g['topic']} {g['applies_to']} {g['gap']}".lower()
        return sum(w in text for w in words)
    matched = sorted((g for g in GAPS if score(g) > 0), key=score, reverse=True)

    fill = []
    region_rows = COVERAGE["column_fill_pct_by_region"]
    cols = [c for c in region_rows[0] if c not in ("region", "n") and c.lower() in words]
    if cols:
        fill = [{"region": r["region"], "projects": r["n"], **{c: r[c] for c in cols}} for r in region_rows]
    status_dates = COVERAGE["status_date_fill_by_region"] if words & {"on_date", "wd_date", "timing",
                                                                        "operational", "withdrawn", "days_ir_to_cod",
                                                                        "days_ir_to_wd", "time", "long"} else []
    return _dumps({
        "topic": topic,
        "gaps": [{"topic": g["topic"], "applies_to": g["applies_to"], "gap": g["gap"]} for g in matched]
                or "No specific gap matched. All gap topics: " + ", ".join(g["topic"] for g in GAPS),
        **({"fill_pct_by_region": fill} if fill else {}),
        **({"date_fill_by_status_and_region": status_dates} if status_dates else {}),
    })


# ---------------------------------------------------------------------------
# 5. run_sql
# ---------------------------------------------------------------------------

def run_sql(query: str, purpose: str) -> str:
    """Run one read-only DuckDB SELECT query and return the rows.

    Returns columns, rows (up to 200), row_count, truncated, and caveats. Read every caveat:
    they say when a result needs a sample size, a date range, or a warning in the answer.
    Queries that compare a fixed-list column to a value that doesn't exist are rejected with
    the valid values listed.

    Args:
        query: One DuckDB SELECT (or WITH ... SELECT) statement.
        purpose: One short line on what this query answers, e.g. 'withdrawal rate by region'.
    """
    reason = guardrails.check_sql(query)
    if reason:
        return _error(reason, purpose=purpose)

    with cursor() as cur:
        try:
            rows = _run_with_timeout(cur, query, fetch=MAX_ROWS + 1)
            columns = [d[0] for d in cur.description or []]
            truncated = len(rows) > MAX_ROWS
            rows = rows[:MAX_ROWS]
            row_count = len(rows)
            if truncated:
                row_count = _run_with_timeout(cur, f"SELECT count(*) FROM ({query.rstrip().rstrip(';')})")[0][0]
        except duckdb.InterruptException:
            return _error(f"Query stopped after {QUERY_TIMEOUT_S}s. Aggregate more or filter first.", purpose=purpose)
        except duckdb.Error as e:
            return _error(f"{type(e).__name__}: {e}", purpose=purpose,
                          hint="Check names with describe_table and values with get_column_values.")

    result = {"purpose": purpose, "columns": columns,
              "rows": [[_clean(v) for v in r] for r in rows],
              "row_count": row_count, "truncated": truncated,
              "caveats": guardrails.annotate(query, row_count)}
    text = _dumps(result)
    while len(text) > MAX_RESULT_CHARS and len(result["rows"]) > 10:
        result["rows"] = result["rows"][: len(result["rows"]) // 2]
        result["truncated"] = True
        text = _dumps(result)
    if result["truncated"] and len(result["rows"]) < row_count:
        result["note"] = (f"Showing {len(result['rows'])} of {row_count} rows. "
                          "Aggregate, filter or ORDER BY ... LIMIT to see what matters.")
        text = _dumps(result)
    return text


# ---------------------------------------------------------------------------
# Agent tool wrappers
# ---------------------------------------------------------------------------

list_tables_tool = function_tool(list_tables)
describe_table_tool = function_tool(describe_table)
get_column_values_tool = function_tool(get_column_values)
get_data_gaps_tool = function_tool(get_data_gaps)
run_sql_tool = function_tool(run_sql)

ALL_TOOLS = [list_tables_tool, describe_table_tool, get_column_values_tool, get_data_gaps_tool, run_sql_tool]
