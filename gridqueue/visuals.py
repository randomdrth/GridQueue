"""Tables, charts and maps for the Writer: the model chooses, the code renders.

The Writer never types a table cell or a chart value. It names a verified finding and the
columns to show; these tools re-run that finding's final_sql (read-only, same guardrails as
run_sql) and build the output from the real rows:

  show_table -> a markdown table, placed into the answer at a marker
  show_chart -> an interactive Vega-Lite chart (bar, line, scatter)
  show_map   -> an interactive US county map (needs a fips_code column)

Each tool returns a marker like <!--gq:chart:2-->. The Writer puts it on its own line where the
visual belongs. The pipeline then swaps table markers for the table, and attaches chart and map
specs to the tool call in /chat's tool_calls, where the page finds and draws them.
"""

import math
import re
from dataclasses import dataclass, field
from typing import Literal

from agents import RunContextWrapper, function_tool

from gridqueue.schemas import Finding
from gridqueue.tools import fetch_rows

VEGA_LITE_SCHEMA = "https://vega.github.io/schema/vega-lite/v6.json"
COUNTIES_TOPOJSON = "/static/us-counties-10m.json"  # us-atlas 3.0.1, ids are 5-digit FIPS strings

MAX_SOURCE_ROWS = 5_000      # enough for every county
MAX_TABLE_ROWS = 30
MAX_BAR_CATEGORIES = 40
MAX_LINE_POINTS = 600
MAX_SCATTER_POINTS = 3_000
MAX_SERIES = 6


@dataclass
class WriterContext:
    """Per-request state the Writer's tools share: the findings they may draw from, and what they built."""
    findings: dict[int, Finding]
    visuals: list[dict] = field(default_factory=list)

    def add(self, kind: str, **visual) -> str:
        marker = f"<!--gq:{kind}:{len(self.visuals) + 1}-->"
        self.visuals.append({"kind": kind, "marker": marker, **visual})
        return marker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _source(ctx: WriterContext, finding_id: int) -> tuple[Finding, list[str], list[list]]:
    f = ctx.findings.get(finding_id)
    if f is None:
        raise ValueError(f"No finding {finding_id}. Available: {sorted(ctx.findings)}.")
    if not f.answered or not f.final_sql.strip():
        raise ValueError(f"Finding {finding_id} has no usable query to draw from.")
    columns, rows = fetch_rows(f.final_sql.strip().rstrip(";"), MAX_SOURCE_ROWS)
    if not rows:
        raise ValueError(f"Finding {finding_id}'s query returns no rows.")
    return f, columns, rows


def _col(columns: list[str], name: str) -> int:
    """Index of a result column, matched case-insensitively. Raises with the available names."""
    for i, c in enumerate(columns):
        if c.lower() == name.strip().lower():
            return i
    raise ValueError(f"Column '{name}' is not in the query result. Available columns: {columns}.")


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


_PERCENT_NAME = re.compile(r"rate|share|pct|percent|fraction|proportion", re.IGNORECASE)


def is_fraction_column(name: str, values: list) -> bool:
    """A rate/share column holding 0-1 fractions: shown as percents (0.304 -> 30.4%)."""
    nums = [v for v in values if _is_number(v)]
    return bool(_PERCENT_NAME.search(name)) and bool(nums) and all(0 <= v <= 1 for v in nums) \
        and any(0 < v < 1 for v in nums)


def is_percent_points_column(name: str, values: list) -> bool:
    """A rate/share column the SQL already multiplied by 100 (13.35 -> 13.4%)."""
    nums = [v for v in values if _is_number(v)]
    named = bool(_PERCENT_NAME.search(name)) or "%" in name
    return named and bool(nums) and all(0 <= v <= 100 for v in nums) and any(v > 1 for v in nums)


def format_percent(v, scale: float = 100) -> str:
    return "–" if v is None else f"{v * scale:.1f}%"


def format_cell(v) -> str:
    """One value on its own. Tables use format_column so a column reads consistently."""
    if v is None:
        return "–"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        if v.is_integer() and abs(v) < 1e15:
            return f"{int(v):,}"
        return f"{v:,.1f}" if abs(v) >= 1_000 else f"{v:,.2f}"
    return str(v).replace("|", "/")


def _is_year_column(name: str) -> bool:
    return "year" in name.lower()


def format_column(name: str, values: list) -> list[str]:
    """Format one column the same way top to bottom: thousands separators (except years) and one
    shared number of decimals, so 12 and 13.95 don't sit in the same column as "12" and "13.95"."""
    nums = [v for v in values if _is_number(v)]
    if nums and _is_year_column(name) and all(float(v).is_integer() for v in nums):
        return ["–" if v is None else str(int(v)) if _is_number(v) else format_cell(v) for v in values]
    floats = [v for v in nums if isinstance(v, float) and not v.is_integer()]
    if not floats:
        return [format_cell(v) for v in values]
    decimals = 1 if max(abs(v) for v in nums) >= 1_000 else 2
    return [f"{v:,.{decimals}f}" if _is_number(v) else format_cell(v) for v in values]


def render_markdown_table(labels: list[str], rows: list[list], percent_cols: set[int] = frozenset(),
                          names: list[str] | None = None) -> str:
    """percent_cols: 0-1 fraction columns. Columns already in percent points are detected here."""
    names = names or labels
    head = "| " + " | ".join(labels) + " |"
    sep = "| " + " | ".join("---:" if rows and _is_number(rows[0][i]) else ":---" for i in range(len(labels))) + " |"
    points = {i for i in range(len(labels)) if i not in percent_cols
              and is_percent_points_column(f"{names[i]} {labels[i]}", [r[i] for r in rows])}
    cols = [[format_percent(r[i]) for r in rows] if i in percent_cols
            else [format_percent(r[i], 1) for r in rows] if i in points
            else format_column(f"{names[i]} {labels[i]}", [r[i] for r in rows])
            for i in range(len(labels))]
    body = ["| " + " | ".join(col[k] for col in cols) + " |" for k in range(len(rows))]
    return "\n".join([head, sep, *body])


def _pretty(name: str) -> str:
    return re.sub(r"_+", " ", name).strip().capitalize()


def _base(title: str, data: list[dict], height: int = 320) -> dict:
    return {"$schema": VEGA_LITE_SCHEMA, "title": title, "width": "container", "height": height,
            "data": {"values": data}, "config": {"view": {"stroke": None}}}


# ---------------------------------------------------------------------------
# Spec builders (pure functions, tested directly)
# ---------------------------------------------------------------------------

def _value_format(percent: bool) -> dict:
    return {"axis": {"format": ".0%"}} if percent else {}


def bar_spec(title: str, x_label: str, y_label: str, data: list[dict], multi: bool, percent: bool = False) -> dict:
    """data: [{category, value, series}]. Horizontal when there are many or long labels.
    Legend click highlights a series; tooltips show exact values."""
    categories = list(dict.fromkeys(d["category"] for d in data))
    horizontal = len(categories) > 8 or max(len(str(c)) for c in categories) > 14
    cat = {"field": "category", "type": "nominal", "sort": categories, "title": x_label,
           **({} if horizontal else {"axis": {"labelAngle": 0}})}
    val = {"field": "value", "type": "quantitative", "title": y_label, **_value_format(percent)}
    tooltip = [{"field": "category", "title": x_label},
               {"field": "value", "title": y_label, "format": ".1%" if percent else ",.4~f"}]
    enc = {"x": val, "y": cat} if horizontal else {"x": cat, "y": val}
    spec = _base(title, data, height=max(240, 22 * len(categories)) if horizontal else 320)
    if multi:
        enc["yOffset" if horizontal else "xOffset"] = {"field": "series"}
        enc["color"] = {"field": "series", "type": "nominal", "title": None}
        enc["opacity"] = {"condition": {"param": "pick", "value": 1}, "value": 0.25}
        tooltip.insert(1, {"field": "series", "title": "Series"})
        spec["params"] = [{"name": "pick", "select": {"type": "point", "fields": ["series"]}, "bind": "legend"}]
    enc["tooltip"] = tooltip
    spec.update(mark={"type": "bar", "cornerRadiusEnd": 2}, encoding=enc)
    return spec


def line_spec(title: str, x_label: str, y_label: str, data: list[dict], multi: bool, percent: bool = False) -> dict:
    """data: [{x, value, series}]. Drag to pan, scroll to zoom, click the legend to highlight."""
    x_type = "quantitative" if all(_is_number(d["x"]) for d in data) else "ordinal"
    enc = {"x": {"field": "x", "type": x_type, "title": x_label,
                 **({"axis": {"format": "d"}, "scale": {"zero": False}} if x_type == "quantitative" else {})},
           "y": {"field": "value", "type": "quantitative", "title": y_label, **_value_format(percent)},
           "tooltip": [{"field": "x", "title": x_label},
                       {"field": "value", "title": y_label, "format": ".1%" if percent else ",.4~f"}]}
    params = [{"name": "zoom", "select": "interval", "bind": "scales"}]
    if multi:
        enc["color"] = {"field": "series", "type": "nominal", "title": None}
        enc["opacity"] = {"condition": {"param": "pick", "value": 1}, "value": 0.2}
        enc["tooltip"].insert(1, {"field": "series", "title": "Series"})
        params.append({"name": "pick", "select": {"type": "point", "fields": ["series"]}, "bind": "legend"})
    spec = _base(title, data)
    spec.update(params=params, mark={"type": "line", "point": True}, encoding=enc)
    return spec


def scatter_spec(title: str, x_label: str, y_label: str, data: list[dict], multi: bool, label_field: str | None) -> dict:
    """data: [{x, value, series, label}]. Drag to pan, scroll to zoom."""
    tooltip = ([{"field": "label", "title": "Item"}] if label_field else []) + \
              [{"field": "x", "title": x_label, "format": ",.4~f"}, {"field": "value", "title": y_label, "format": ",.4~f"}]
    enc = {"x": {"field": "x", "type": "quantitative", "title": x_label, "scale": {"zero": False}},
           "y": {"field": "value", "type": "quantitative", "title": y_label, "scale": {"zero": False}},
           "tooltip": tooltip}
    if multi:
        enc["color"] = {"field": "series", "type": "nominal", "title": None}
        enc["tooltip"].insert(0, {"field": "series", "title": "Series"})
    spec = _base(title, data)
    spec.update(params=[{"name": "zoom", "select": "interval", "bind": "scales"}],
                mark={"type": "point", "filled": True, "opacity": 0.6}, encoding=enc)
    return spec


def _round_sig(x: float, digits: int = 2) -> float:
    if x == 0:
        return 0.0
    return round(x, -int(math.floor(math.log10(abs(x)))) + digits - 1)


def class_breaks(values: list[float], classes: int = 5) -> list[float]:
    """Quantile class breaks, rounded to 2 significant figures so the legend reads cleanly
    (e.g. 120, 450, 1,300, 3,800). Quantiles keep a few huge counties from washing out the rest."""
    vals = sorted(v for v in values if v is not None)
    if len(vals) < 2:
        return []
    raw = [vals[min(len(vals) - 1, int(len(vals) * k / classes))] for k in range(1, classes)]
    return sorted({_round_sig(b) for b in raw if b > vals[0]})


def map_spec(title: str, value_label: str, data: list[dict]) -> dict:
    """US county choropleth. data: [{fips, value, name}]. Counties without data are drawn grey;
    hovering shows the county and its value. Quantile colours keep a few huge counties from
    washing out the rest."""
    return {
        "$schema": VEGA_LITE_SCHEMA, "title": title, "width": "container", "height": 420,
        "projection": {"type": "albersUsa"},
        "config": {"view": {"stroke": None}},
        "layer": [
            {"data": {"url": COUNTIES_TOPOJSON, "format": {"type": "topojson", "feature": "counties"}},
             "mark": {"type": "geoshape", "fill": "#eeeeee", "stroke": "white", "strokeWidth": 0.2}},
            {"data": {"values": data},
             "transform": [{"lookup": "fips", "from": {
                 "data": {"url": COUNTIES_TOPOJSON, "format": {"type": "topojson", "feature": "counties"}},
                 "key": "id"}, "as": "geo"}],
             "mark": {"type": "geoshape", "stroke": "white", "strokeWidth": 0.2},
             "encoding": {"shape": {"field": "geo", "type": "geojson"},
                          "color": _map_color(value_label, data),
                          "tooltip": [{"field": "name", "title": "County"},
                                      {"field": "value", "title": value_label, "format": ",.4~f"}]}},
            {"data": {"url": COUNTIES_TOPOJSON, "format": {"type": "topojson", "mesh": "states"}},
             "mark": {"type": "geoshape", "fill": None, "stroke": "#888", "strokeWidth": 0.6}},
        ],
    }


def _map_color(value_label: str, data: list[dict]) -> dict:
    breaks = class_breaks([d["value"] for d in data])
    if not breaks:
        return {"field": "value", "type": "quantitative", "title": value_label, "scale": {"scheme": "oranges"}}
    return {"field": "value", "type": "quantitative", "title": value_label,
            "scale": {"type": "threshold", "domain": breaks,
                      "scheme": {"name": "oranges", "count": len(breaks) + 1}},
            "legend": {"format": ",~s"}}


# ---------------------------------------------------------------------------
# The Writer's tools
# ---------------------------------------------------------------------------

def _groups_cut_off(rows: list, first_col: int, limit: int) -> list[str]:
    """Groups in the first column (e.g. a region) that a cut at `limit` rows would drop entirely.
    Smoke run 6 showed 16 ERCOT rows of a 42-row ERCOT+PJM table and silently dropped PJM."""
    if len(rows) <= limit:
        return []
    values = [r[first_col] for r in rows]
    if len(set(values)) > len(values) / 2:  # first column is not a grouping (e.g. a ranking)
        return []
    kept = set(values[:limit])
    return [str(v) for v in dict.fromkeys(values) if v not in kept]


def _table(ctx: WriterContext, finding_id: int, columns: list[str], labels: list[str], max_rows: int, title: str) -> str:
    f, cols, rows = _source(ctx, finding_id)
    idx = [_col(cols, c) for c in columns]
    if len(labels) != len(columns):
        labels = [_pretty(cols[i]) for i in idx]
    limit = max(1, min(max_rows, MAX_TABLE_ROWS))
    hidden = _groups_cut_off(rows, idx[0], limit)
    if hidden:
        raise ValueError(
            f"showing {limit} of {len(rows)} rows would hide every row for {', '.join(hidden[:3])}. "
            f"Use show_chart (a line chart if the rows are years), or a max_rows of at least {len(rows)}"
            + (" if that fits." if len(rows) <= MAX_TABLE_ROWS else ". That is too many rows for a table."))
    shown = [[r[i] for i in idx] for r in rows[:limit]]
    percent_cols = {j for j, i in enumerate(idx) if is_fraction_column(cols[i], [r[i] for r in rows])}
    marker = ctx.add("table", title=title, finding_id=finding_id, columns=labels, rows=shown,
                     total_rows=len(rows), sql=f.final_sql,
                     markdown=render_markdown_table(labels, shown, percent_cols, [cols[i] for i in idx]))
    more = f" (first {len(shown)} of {len(rows)} rows)" if len(rows) > len(shown) else ""
    return f"ok: table{more}. Put {marker} on its own line where the table belongs. Do not retype its numbers."


def _chart(ctx: WriterContext, finding_id: int, chart_type: str, x: str, y: list[str], series: str,
           title: str, x_label: str, y_label: str) -> str:
    f, cols, rows = _source(ctx, finding_id)
    if not y:
        raise ValueError("Give at least one y column.")
    if len(y) > 1 and series:
        raise ValueError("Use either several y columns or one series column, not both.")
    xi, yis = _col(cols, x), [_col(cols, c) for c in y]
    si = _col(cols, series) if series else None
    for yi in yis:
        if not any(_is_number(r[yi]) for r in rows):
            raise ValueError(f"y column '{cols[yi]}' is not numeric.")
    data = []
    for r in rows:
        for yi in yis:
            if r[yi] is None or r[xi] is None:
                continue
            data.append({"x" if chart_type != "bar" else "category": r[xi], "value": r[yi],
                         "series": (str(r[si]) if si is not None else _pretty(cols[yi])),
                         "label": str(r[xi])})
    multi = len(yis) > 1 or si is not None
    percent = all(is_fraction_column(cols[yi], [r[yi] for r in rows]) for yi in yis)
    n_series = len({d["series"] for d in data})
    if n_series > MAX_SERIES:
        raise ValueError(f"{n_series} series is too many to read; use at most {MAX_SERIES}.")
    if chart_type == "bar":
        n = len({d["category"] for d in data})
        if n > MAX_BAR_CATEGORIES:
            raise ValueError(f"{n} categories is too many for a bar chart (max {MAX_BAR_CATEGORIES}). "
                             "Use a table, or a finding with fewer rows.")
        spec = bar_spec(title, x_label, y_label, data, multi, percent)
    elif chart_type == "line":
        if len(data) > MAX_LINE_POINTS:
            raise ValueError(f"{len(data)} points is too many for a line chart (max {MAX_LINE_POINTS}).")
        spec = line_spec(title, x_label, y_label, data, multi, percent)
    else:
        if not all(_is_number(d["x"]) for d in data):
            raise ValueError("A scatter chart needs a numeric x column.")
        spec = scatter_spec(title, x_label, y_label, data[:MAX_SCATTER_POINTS], multi, label_field="label")
    marker = ctx.add("chart", title=title, finding_id=finding_id, sql=f.final_sql, spec=spec)
    return f"ok: {chart_type} chart with {len(data)} points. Put {marker} on its own line where it belongs."


def _map(ctx: WriterContext, finding_id: int, value_column: str, title: str, value_label: str) -> str:
    f, cols, rows = _source(ctx, finding_id)
    try:
        fi = _col(cols, "fips_code")
    except ValueError:
        raise ValueError(f"A county map needs a fips_code column; finding {finding_id}'s query has {cols}. "
                         "Skip the map.") from None
    vi = _col(cols, value_column)
    ci = next((i for i, c in enumerate(cols) if c.lower() == "county"), None)
    si = next((i for i, c in enumerate(cols) if c.lower() == "state"), None)
    data = []
    for r in rows:
        if r[fi] is None or not _is_number(r[vi]):
            continue
        fips = str(r[fi]).split(".")[0].zfill(5)
        name = ", ".join(str(r[i]) for i in (ci, si) if i is not None and r[i]) or fips
        data.append({"fips": fips, "value": r[vi], "name": name})
    if not data:
        raise ValueError(f"No rows with both a fips_code and a numeric {value_column}.")
    marker = ctx.add("map", title=title, finding_id=finding_id, sql=f.final_sql,
                     spec=map_spec(title, value_label, data), counties=len(data))
    return f"ok: county map of {len(data)} counties. Put {marker} on its own line where it belongs."


def _safe(fn, *args) -> str:
    try:
        return fn(*args)
    except ValueError as e:
        return f"Error: {e}"


@function_tool
def show_table(ctx: RunContextWrapper[WriterContext], finding_id: int, columns: list[str], labels: list[str],
               max_rows: int, title: str) -> str:
    """Show a table built from a finding's verified query. The numbers come straight from the database,
    so never type a markdown table yourself.

    Args:
        finding_id: The finding to draw from.
        columns: Result columns to show, in order (use the column names listed for that finding).
        labels: Readable header for each column, same order, with units, e.g. 'Median cost ($/kW)'.
        max_rows: Rows to show (1-30). The query's own order is kept.
        title: Short title for the table, or '' for none.
    """
    return _safe(_table, ctx.context, finding_id, columns, labels, max_rows, title)


@function_tool
def show_chart(ctx: RunContextWrapper[WriterContext], finding_id: int, chart_type: Literal["bar", "line", "scatter"],
               x: str, y: list[str], series: str, title: str, x_label: str, y_label: str) -> str:
    """Show an interactive chart built from a finding's verified query (hover for exact values).

    bar: compare categories (up to 40). line: a trend over years. scatter: two numeric measures.

    Args:
        finding_id: The finding to draw from.
        chart_type: bar, line or scatter.
        x: Result column for categories (bar) or the x axis (line, scatter).
        y: One or more numeric result columns. Several columns become several series.
        series: Optional result column that splits rows into series (e.g. q_status), or ''.
        title: Chart title.
        x_label: x-axis label.
        y_label: y-axis label, with units.
    """
    return _safe(_chart, ctx.context, finding_id, chart_type, x, y, series, title, x_label, y_label)


@function_tool
def show_map(ctx: RunContextWrapper[WriterContext], finding_id: int, value_column: str, title: str,
             value_label: str) -> str:
    """Show an interactive US county map from a finding whose query has a fips_code column.
    Use for questions about where: counties, siting, saturation, concentration.

    Args:
        finding_id: The finding to draw from (its query must return fips_code).
        value_column: Numeric result column that colours each county.
        title: Map title.
        value_label: Legend label with units, e.g. 'Active queue (MW)'.
    """
    return _safe(_map, ctx.context, finding_id, value_column, title, value_label)


WRITER_TOOLS = [show_table, show_chart, show_map]


# ---------------------------------------------------------------------------
# After the Writer: put tables in place, attach chart/map specs to their tool calls
# ---------------------------------------------------------------------------

_MARKER = re.compile(r"<!--gq:(table|chart|map):(\d+)-->")


def place_visuals(text: str, ctx: WriterContext) -> str:
    """Swap table markers for the rendered table. Keep chart/map markers (the page draws them).
    A visual the Writer built but forgot to place goes before 'How I got this'."""
    by_marker = {v["marker"]: v for v in ctx.visuals}
    placed = {m.group(0) for m in _MARKER.finditer(text)}
    missing = [v for v in ctx.visuals if v["marker"] not in placed]

    def block(v: dict) -> str:
        if v["kind"] != "table":
            return v["marker"]
        title = f"**{v['title']}**\n\n" if v.get("title") else ""
        note = f"\n\n_First {len(v['rows'])} of {v['total_rows']:,} rows._" if v["total_rows"] > len(v["rows"]) else ""
        # The marker stays after the table (invisible in markdown) so the page can add CSV/SQL buttons.
        return f"{title}{v['markdown']}{note}\n\n{v['marker']}"

    text = _MARKER.sub(lambda m: "\n\n" + block(by_marker[m.group(0)]) + "\n\n" if m.group(0) in by_marker else "", text)
    if missing:
        extra = "\n\n".join(block(v) for v in missing)
        anchor = text.find("**How I got this**")
        text = (text[:anchor] + extra + "\n\n" + text[anchor:]) if anchor >= 0 else (text + "\n\n" + extra)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def attach_to_tool_calls(tool_calls: list[dict], ctx: WriterContext) -> None:
    """Give each show_* tool call the visual it built (not truncated, unlike tool results)."""
    by_marker = {v["marker"]: v for v in ctx.visuals}
    for call in tool_calls:
        if call["name"] in ("show_table", "show_chart", "show_map"):
            m = _MARKER.search(call["result"])
            if m and m.group(0) in by_marker:
                v = by_marker[m.group(0)]
                call["visual"] = {k: v[k] for k in v if k != "markdown"}
