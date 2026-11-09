"""The database connection and the context build_db.py wrote. Loaded once, at import.

The security boundary is the connection, not the prompt: read-only, no file or network
access, and configuration locked so a query can't SET its way out of either.
"""

import json
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CONTEXT_DIR = DATA / "context"
DB_PATH = DATA / "grid.duckdb"

try:
    con = duckdb.connect(
        str(DB_PATH),
        read_only=True,
        config={"enable_external_access": False, "lock_configuration": True},
    )
    SCHEMA = json.loads((CONTEXT_DIR / "schema.json").read_text())
    JOINS = json.loads((CONTEXT_DIR / "joins.json").read_text())
    GAPS = json.loads((CONTEXT_DIR / "data_gaps.json").read_text())
    GLOSSARY = json.loads((CONTEXT_DIR / "glossary.json").read_text())
    COVERAGE = json.loads((CONTEXT_DIR / "coverage.json").read_text())
except (FileNotFoundError, duckdb.Error) as e:
    raise SystemExit(f"Database or context missing ({e}). Run `uv run --group build build_db.py` first.")

TABLES: dict[str, dict] = SCHEMA["tables"]

# {table: {column: column_info}} for quick lookups.
COLUMNS: dict[str, dict[str, dict]] = {
    t: {c["name"]: c for c in info["columns"]} for t, info in TABLES.items()
}

# {column: set of valid values}, merged across every table that has the column.
# Only text columns with a fixed list (booleans are skipped: their 'values' are True/False).
ENUM_VALUES: dict[str, set[str]] = {}
for _table, _cols in COLUMNS.items():
    for _name, _info in _cols.items():
        if _info["type"] == "VARCHAR" and "values" in _info:
            ENUM_VALUES.setdefault(_name, set()).update(_info["values"].keys())


def cursor() -> duckdb.DuckDBPyConnection:
    """A fresh cursor per call. The agents SDK runs tools in threads, and a DuckDB
    connection isn't safe to share across threads; cursors are."""
    return con.cursor()
