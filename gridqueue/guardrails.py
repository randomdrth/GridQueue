"""Checks around every query the agent runs.

Layer 1, check_sql: reject queries that are objectively wrong, before they run. The agent
reads the reason instead of a result and tries again.
  - anything other than one SELECT statement
  - a fixed-list column compared to a value that doesn't exist (q_status = 'complete')

Layer 2, annotate: the query ran, but the answer needs a caveat. Attached to the result.
  - data gaps whose applies_to names a column the query uses (from data_gaps.json)
  - sparse columns (under 50% filled)
  - projects joined straight to eia_generators (row fan-out)
  - zero rows returned

Neither layer is the security boundary; the read-only connection in db.py is.
"""

import re

import duckdb

from gridqueue.db import COLUMNS, ENUM_VALUES, GAPS, TABLES, con

MAX_VALUES_IN_MESSAGE = 30
SPARSE_FILL_PCT = 50

_STRING = r"'((?:[^']|'')*)'"
_OPS = r"(?:=|!=|<>)"


def _strip_comments(sql: str) -> str:
    sql = re.sub(r"--[^\n]*", " ", sql)
    return re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)


def _unquote(s: str) -> str:
    return s.replace("''", "'")


# ---------------------------------------------------------------------------
# Layer 1
# ---------------------------------------------------------------------------

def _literal_comparisons(sql: str, column: str) -> list[str]:
    """Every string literal the query compares `column` to with =, !=, <>, IN or NOT IN."""
    col = rf"(?<![\w.])(?:\w+\.)?{re.escape(column)}\b"
    found = []
    found += [m.group(1) for m in re.finditer(rf"{col}\s*{_OPS}\s*{_STRING}", sql, re.IGNORECASE)]
    found += [m.group(1) for m in re.finditer(rf"{_STRING}\s*{_OPS}\s*{col}", sql, re.IGNORECASE)]
    for m in re.finditer(rf"{col}\s+(?:NOT\s+)?IN\s*\(([^()]*)\)", sql, re.IGNORECASE):
        found += re.findall(_STRING, m.group(1))
    return [_unquote(v) for v in found]


def _invalid_value_message(sql: str) -> str | None:
    problems = []
    for column, valid in ENUM_VALUES.items():
        # Skip when the query defines its own column with this name (e.g. "... AS region").
        if re.search(rf"\bAS\s+{re.escape(column)}\b", sql, re.IGNORECASE):
            continue
        for value in _literal_comparisons(sql, column):
            if value in valid:
                continue
            near = [v for v in valid if v.lower() == value.lower()]
            hint = f" Did you mean '{near[0]}'? Matching is case-sensitive." if near else ""
            listed = sorted(valid)[:MAX_VALUES_IN_MESSAGE]
            more = f" (+{len(valid) - len(listed)} more)" if len(valid) > len(listed) else ""
            problems.append(f"{column} = '{value}' matches no rows: it is not a value of {column}.{hint} "
                            f"Valid values: {', '.join(listed)}{more}.")
    if not problems:
        return None
    return "Rejected before running. " + " ".join(dict.fromkeys(problems)) + \
        " Use get_column_values to check exact values, then retry."


def check_sql(query: str) -> str | None:
    """None if the query may run, otherwise the reason it was rejected."""
    try:
        statements = con.extract_statements(query)
    except duckdb.Error as e:
        return f"Rejected: the SQL does not parse. {e}"
    if len(statements) != 1:
        return f"Rejected: send exactly one statement per call (got {len(statements)})."
    if statements[0].type != duckdb.StatementType.SELECT:
        return f"Rejected: only read-only SELECT queries are allowed (got {statements[0].type.name})."
    return _invalid_value_message(_strip_comments(query))


# ---------------------------------------------------------------------------
# Layer 2
# ---------------------------------------------------------------------------

def _mentions(sql: str, name: str) -> bool:
    return re.search(rf"\b{re.escape(name)}\b", sql, re.IGNORECASE) is not None


def referenced(sql: str) -> dict[str, set[str]]:
    """{table: columns of that table the query names}, for tables the query names."""
    sql = _strip_comments(sql)
    return {t: {c for c in COLUMNS[t] if _mentions(sql, c)} for t in TABLES if _mentions(sql, t)}


def _parse_applies_to(applies_to: str) -> list[tuple[str, str | None]]:
    """'projects.on_date, wd_date, days_*' -> [(projects, on_date), (projects, wd_date), (projects, days_*)].
    A bare table name gives (table, None): table-wide gaps, surfaced by get_data_gaps only."""
    out, table = [], None
    for part in (p.strip() for p in applies_to.split(",")):
        if "." in part:
            table, col = part.split(".", 1)
            out.append((table, col))
        elif part in TABLES:
            table = part
            out.append((part, None))
        elif table:
            out.append((table, part))
    return out


def _column_matches(pattern: str, columns: set[str]) -> bool:
    if pattern.endswith("*"):
        return any(c.startswith(pattern[:-1]) for c in columns)
    return pattern in columns


def annotate(query: str, row_count: int) -> list[str]:
    refs = referenced(query)
    notes = []

    for gap in GAPS:
        for table, col in _parse_applies_to(gap["applies_to"]):
            if col and table in refs and _column_matches(col, refs[table]):
                notes.append(f"[{gap['topic']}] {gap['gap']}")
                break

    sparse = sorted({f"{t}.{c} ({COLUMNS[t][c]['fill_pct']}% filled)"
                     for t, cols in refs.items() for c in cols
                     if COLUMNS[t][c]["fill_pct"] < SPARSE_FILL_PCT})
    if sparse:
        notes.append("[sparse columns] " + ", ".join(sparse) +
                     ". Report how many rows had the value, not just the result. "
                     "Call get_data_gaps for fill rates by region.")

    if "projects" in refs and "eia_generators" in refs and "county_summary" not in refs:
        notes.append("[fan-out] projects and eia_generators only link by county, many-to-many. Joining them "
                     "row to row multiplies rows and inflates sums. Aggregate each side by fips_code first, "
                     "or use county_summary.")

    if row_count == 0:
        notes.append("[no rows] The query returned nothing. Check filter values with get_column_values "
                     "before concluding the data has no such records.")
    return notes
