"""Offline build step. Run once, before app.py:  uv run --group build build_db.py

Reads the raw files in data/raw/ and writes:
  data/grid.duckdb        the database the agent queries (read-only at runtime)
  data/context/*.json     the context the agent retrieves: schema, value lists, joins, gaps

Sources
  1. LBNL "Queued Up" 2026 edition: every US generator interconnection request through 2025
  2. EIA-860 2025 (early release): every existing, proposed and retired US generator
  3. LBNL interconnection cost studies: project-level $/kW for MISO, PJM, SPP, ISO-NE, NYISO
     and non-ISO utilities (BPA, Duke, PacifiCorp)

What this step fixes so the agent doesn't have to
  - Queue IDs are only unique per grid operator, mix ints and strings, and 9 were turned into
    dates by Excel. We build a stable project_uid and drop exact duplicate rows.
  - Six cost files use different column names, fuel labels, status words and dollar years.
    We unify them into one table in 2024 dollars and link each row to its queue project.
  - EIA-860 has county names but no FIPS codes. We map them to FIPS so EIA joins the queue.
  - Fill rates differ a lot by region. We measure them and write them into the context, so the
    agent can say "based on N of M projects" instead of silently dropping rows.
"""

import json
import re
import zipfile
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
OUT_DB = ROOT / "data" / "grid.duckdb"
CONTEXT = ROOT / "data" / "context"

QUEUE_FILE = RAW / "lbnl_ix_queue_data_file_thru2025.xlsx"
EIA_ZIP = RAW / "eia8602025.zip"

# BEA GDP implicit price deflator, annual, index 2017=100 (FRED series A191RD3A086NBEA).
# LBNL deflated its cost files with a GDP deflator; we use the same family to move 2022$ to 2024$.
GDP_DEFLATOR = {2022: 118.023, 2024: 125.428}
TARGET_DOLLAR_YEAR = 2024

ISOS = ["CAISO", "ERCOT", "ISO-NE", "MISO", "NYISO", "PJM", "SPP"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_missing(x) -> bool:
    """True for None, NaN, pd.NA and NaT. (pd.NA is not a float, so isinstance checks miss it.)"""
    try:
        return x is None or bool(pd.isna(x))
    except (TypeError, ValueError):
        return False


def fix_mojibake(x):
    """Repair UTF-8 text that was decoded as Mac Roman in the source file: 'APS‚Äôs' -> 'APS's'.
    A few cells were garbled twice ('‚Äö√Ñ√¨' -> '‚Äì' -> '–'), so repeat until stable."""
    for _ in range(3):
        fixed = _fix_mojibake_once(x)
        if fixed == x:
            return x
        x = fixed
    return x


def _fix_mojibake_once(x):
    if not isinstance(x, str) or not any(m in x for m in ("‚Ä", "√")):
        return x
    try:
        return x.encode("mac_roman").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        # The string also holds a character Mac Roman can't encode: repair the common sequences.
        for bad, good in {"‚Äô": "’", "‚Äò": "‘", "‚Äú": "“", "‚Äù": "”",
                          "‚Äì": "–", "‚Äî": "—", "‚Ä¶": "…", "‚Ä¢": "•"}.items():
            x = x.replace(bad, good)
        return x


def first_fips(x) -> str | None:
    """5-digit county FIPS. A few multi-county projects have two codes glued together in the
    source ('3610136003', or '603106019' after Excel dropped the leading zero): keep the first."""
    if is_missing(x):
        return None
    s = re.sub(r"\D", "", str(int(x)) if isinstance(x, float) else str(x))
    if len(s) > 5:
        s = s[:-5]  # drop the trailing second code
    return s.zfill(5) if s else None


def norm_id(x) -> str | None:
    """Comparable form of a queue ID: uppercase, alphanumerics only, no leading zeros."""
    if is_missing(x):
        return None
    s = re.sub(r"[^A-Z0-9]", "", str(x).upper()).lstrip("0")
    return s or None


def norm_county(x) -> str | None:
    """Comparable form of a county name, so 'St. Lucie County' == 'Saint Lucie'."""
    if is_missing(x):
        return None
    s = str(x).upper().strip()
    s = re.sub(r"\b(COUNTY|PARISH|BOROUGH|CENSUS AREA|MUNICIPALITY|CITY AND BOROUGH)\b", "", s)
    s = re.sub(r"\bSAINT\b", "ST", s)
    s = re.sub(r"\bSAINTE\b", "STE", s)
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s or None


def clean_text(series: pd.Series) -> pd.Series:
    """Repair encoding, strip whitespace, and turn empty strings into nulls (ISO-NE stores blanks as '')."""
    s = series.map(fix_mojibake).astype("string").str.strip()
    return s.mask(s == "")


def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace({"": None}), errors="coerce")


def to_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.date


def days_between(start: pd.Series, end: pd.Series) -> pd.Series:
    """Days from start to end. Null if either is missing or end is before start (bad data)."""
    d = (pd.to_datetime(end) - pd.to_datetime(start)).dt.days
    return d.where(d >= 0).astype("Int64")


# One vocabulary for resource type across the queue, the six cost files and EIA-860.
FUEL_GROUP = {
    # cost-file labels
    "solar": "Solar", "wind": "Wind", "wind land": "Wind", "wind onshore": "Wind",
    "wind offshore": "Offshore Wind", "offshore wind": "Offshore Wind",
    "storage": "Battery", "flywheel": "Battery",
    "natural gas": "Gas", "gas": "Gas", "reciprocating engine": "Gas", "diesel/gas": "Gas",
    "coal": "Coal", "nuclear": "Nuclear", "hydro": "Hydro",
    "pumped hydro": "Pumped Storage", "pumped hydro storage": "Pumped Storage",
    "geothermal": "Geothermal", "biomass": "Biomass", "biogas": "Biomass", "oil": "Oil",
    "fuel cell": "Other", "other": "Other",
    "solar hybrid": "Solar+Battery", "wind hybrid": "Wind+Battery",
    "wind hybrid land": "Wind+Battery", "wind hybrid offshore": "Offshore Wind+Battery",
    "gas hybrid": "Gas+Battery", "hybrid": "Other Hybrid",
}

EIA_TECH_GROUP = {
    "Solar Photovoltaic": "Solar", "Solar Thermal without Energy Storage": "Solar",
    "Solar Thermal with Energy Storage": "Solar",
    "Onshore Wind Turbine": "Wind", "Offshore Wind Turbine": "Offshore Wind",
    "Batteries": "Battery", "Flywheels": "Battery",
    "Hydroelectric Pumped Storage": "Pumped Storage",
    "Conventional Hydroelectric": "Hydro", "Nuclear": "Nuclear", "Geothermal": "Geothermal",
    "Conventional Steam Coal": "Coal", "Coal Integrated Gasification Combined Cycle": "Coal",
    "Petroleum Liquids": "Oil", "Petroleum Coke": "Oil",
    "Wood/Wood Waste Biomass": "Biomass", "Other Waste Biomass": "Biomass",
    "Landfill Gas": "Biomass", "Municipal Solid Waste": "Biomass",
}


def eia_tech_group(tech) -> str:
    if pd.isna(tech):
        return "Other"
    if tech in EIA_TECH_GROUP:
        return EIA_TECH_GROUP[tech]
    if str(tech).startswith("Natural Gas") or tech == "Other Natural Gas":
        return "Gas"
    return "Other"


POI_PLACEHOLDER = re.compile(
    r"^\s*("
    r"unknown|undefined|other_?|n/?a|tbd|none|-|\?|various|multiple"   # explicit unknowns
    r"|[0-9.]+\s*(k?v)?"                                               # just a voltage: '138kV', '34.5'
    r"|dist(ribution)?\.?\s*feeder"                                    # generic feeder, not a location
    r")\s*$", re.IGNORECASE)


def clean_poi(name) -> str | None:
    """POI name usable for grouping, or None when it is a placeholder rather than a location."""
    if is_missing(name):
        return None
    s = re.sub(r";\s*N/A\s*$", "", str(name), flags=re.IGNORECASE).strip()  # 'Chester; N/A' -> 'Chester'
    return None if not s or POI_PLACEHOLDER.match(s) else s


def queue_type_group(type_clean) -> str:
    """Collapse the queue's 49 type_clean combos into the shared vocabulary.
    Keeps the common hybrids; rarer combos become 'Other Hybrid'."""
    if pd.isna(type_clean):
        return "Other"
    keep = {"Solar", "Wind", "Battery", "Gas", "Hydro", "Coal", "Offshore Wind", "Nuclear",
            "Geothermal", "Oil", "Solar+Battery", "Wind+Battery", "Gas+Battery",
            "Offshore Wind+Battery"}
    if type_clean in keep:
        return type_clean
    if type_clean == "Diesel":
        return "Oil"
    if type_clean == "Other Storage":
        return "Battery"
    if type_clean in ("Other", "Hydrogen"):
        return "Other"
    return "Other Hybrid"


# ---------------------------------------------------------------------------
# 1. Interconnection queue (LBNL Queued Up)
# ---------------------------------------------------------------------------

def build_projects() -> pd.DataFrame:
    q = pd.read_excel(QUEUE_FILE, sheet_name="03. Complete Queue Data", header=1)
    n_raw = len(q)

    # Excel turned 9 CLPT queue IDs into dates. Keep them, but readable and flagged.
    is_date_id = q["q_id"].map(lambda x: isinstance(x, (pd.Timestamp, date)))
    q["q_id_note"] = None
    q.loc[is_date_id, "q_id_note"] = "original ID was converted to a date by Excel in the source file"
    q["q_id"] = q["q_id"].map(
        lambda x: x.strftime("%Y-%m-%d") if isinstance(x, (pd.Timestamp, date))
        else (str(int(x)) if isinstance(x, float) and float(x).is_integer() else str(x))
    )

    for col in ["q_status", "IA_phase_raw", "IA_phase_clean", "county", "state", "poi_name",
                "region", "project_name", "utility", "entity", "developer", "cluster",
                "service", "project_type", "type_1", "type_2", "type_3", "type_clean"]:
        q[col] = clean_text(q[col])

    # Exact duplicate rows exist in the source (e.g. Duke 1 and 2 appear twice). Drop them.
    q = q.drop_duplicates(subset=[c for c in q.columns if c != "q_id_note"]).copy()
    n_exact_dupes = n_raw - len(q)

    # Remaining repeats of (entity, q_id) are different records sharing an ID. Keep all,
    # numbered, so project_uid stays unique: "PJM|A03", "Duke|2", "Duke|2#2".
    q["dup_n"] = q.groupby(["entity", "q_id"]).cumcount() + 1
    q["project_uid"] = q["entity"] + "|" + q["q_id"]
    q.loc[q["dup_n"] > 1, "project_uid"] += "#" + q["dup_n"].astype(str)
    q["shares_q_id_with_other_record"] = q.groupby(["entity", "q_id"])["q_id"].transform("size") > 1

    for col in ["q_date", "prop_date", "on_date", "wd_date", "ia_date"]:
        q[col] = to_date(q[col])
    for col in ["mw_1", "mw_2", "mw_3"]:
        q[col] = to_num(q[col])

    q["fips_code"] = q["fips_code"].map(first_fips)
    q["county_norm"] = q["county"].map(norm_county)
    q["poi_name_clean"] = q["poi_name"].map(clean_poi)

    q["type_group"] = q["type_clean"].map(queue_type_group)
    q["is_hybrid"] = q["type_2"].notna()
    q["mw_total"] = q[["mw_1", "mw_2", "mw_3"]].sum(axis=1, min_count=1)
    # 16 requests list negative capacity (derates of existing plants). Keep the raw mw_1,
    # but null mw_total so capacity sums are not silently reduced.
    q["mw_note"] = None
    q.loc[q["mw_1"] < 0, "mw_note"] = "negative capacity in source (likely a derate); excluded from mw_total"
    q.loc[q["mw_1"] < 0, "mw_total"] = None

    q["q_year"] = to_num(q["q_year"]).astype("Int64")
    q["prop_year"] = to_num(q["prop_year"]).astype("Int64")
    q["on_year"] = pd.to_datetime(q["on_date"]).dt.year.astype("Int64")
    q["wd_year"] = pd.to_datetime(q["wd_date"]).dt.year.astype("Int64")
    q["ia_year"] = pd.to_datetime(q["ia_date"]).dt.year.astype("Int64")

    q["days_ir_to_cod"] = days_between(q["q_date"], q["on_date"])
    q["days_ir_to_wd"] = days_between(q["q_date"], q["wd_date"])
    q["days_ir_to_ia"] = days_between(q["q_date"], q["ia_date"])
    q["days_ia_to_cod"] = days_between(q["ia_date"], q["on_date"])

    q["is_iso"] = q["region"].isin(ISOS)

    cols = ["project_uid", "entity", "q_id", "q_id_note", "shares_q_id_with_other_record",
            "region", "is_iso", "utility", "state", "county", "county_norm", "fips_code",
            "poi_name", "poi_name_clean", "project_name", "developer", "cluster", "service", "project_type",
            "q_status", "IA_phase_clean", "IA_phase_raw",
            "type_clean", "type_group", "is_hybrid", "type_1", "type_2", "type_3",
            "mw_1", "mw_2", "mw_3", "mw_total", "mw_note",
            "q_date", "prop_date", "ia_date", "on_date", "wd_date",
            "q_year", "prop_year", "ia_year", "on_year", "wd_year",
            "days_ir_to_ia", "days_ir_to_cod", "days_ia_to_cod", "days_ir_to_wd"]
    q = q[cols].rename(columns={"IA_phase_clean": "ia_phase", "IA_phase_raw": "ia_phase_raw"})
    print(f"projects: {n_raw} raw rows, {n_exact_dupes} exact duplicates dropped, {len(q)} kept")
    return q


# ---------------------------------------------------------------------------
# 2. Interconnection costs (six LBNL files -> one table)
# ---------------------------------------------------------------------------

# Per file: queue entity, ID column, column renames, dollar year, status snapshot description.
COST_FILES = [
    dict(file="miso_costs_2021_clean_data.xlsx", sheet="data", entity="MISO", id_col="Project #",
         dollar_year=2022, data_through="2021",
         cols={"Transmission Owner": "transmission_owner", "Study Type": "study_type",
               "Study Date": "study_date", "Service Type": "service_raw", "Fuel": "fuel_raw",
               "Nameplate MW": "nameplate_mw", "Request Status": "status_raw",
               "Real POI/kW": "poi", "Real Network/kW": "network", "Real Total/kW": "total"}),
    dict(file="pjm_costs_2022_clean_data.xlsx", sheet="data", entity="PJM", id_col="Project #",
         dollar_year=2022, data_through="2022",
         cols={"Transmission Owner": "transmission_owner", "Study Type": "study_type",
               "Study Date": "study_date", "Service Type": "service_raw", "Fuel": "fuel_raw",
               "Nameplate MW": "nameplate_mw", "Request Status": "status_raw",
               "$2022 POI Cost/kW": "poi", "$2022 Network Cost/kW": "network",
               "$2022 Total Cost/kW": "total"}),
    dict(file="spp_costs_2023_clean_data.xlsx", sheet="data", entity="SPP", id_col="Project #",
         dollar_year=2022, data_through="2023",
         cols={"Transmission Owner": "transmission_owner", "Study Type": "study_type",
               "Study Date": "study_date", "Service Type": "service_raw", "Fuel": "fuel_raw",
               "Nameplate MW": "nameplate_mw", "Request Status": "status_raw",
               "$2022 POI Cost/kW": "poi", "$2022 Network Cost/kW": "network",
               "$2022 Total Cost/kW": "total"}),
    dict(file="isone_interconnection_cost_data_publication_vfinal.xlsx", sheet="data",
         entity="ISO-NE", id_col="Queue ID 1", dollar_year=2022, data_through="2021-12-17",
         cols={"Study Type": "study_type", "Study Date": "study_date",
               "Service Type": "service_raw", "Resource Type": "fuel_raw", "County": "county",
               "Nameplate MW": "nameplate_mw", "Request Status": "status_raw",
               "$2022 POI Cost/kW": "poi", "$2022 Network Cost/kW": "network",
               "$2022 Total Cost/kW": "total"}),
    dict(file="nyiso_2022_final_data_cleaned_publication_vfinal.xlsx", sheet="data",
         entity="NYISO", id_col="Queue ID", dollar_year=2022, data_through="2021-12-07",
         cols={"Interconnecting Transmission Owner": "transmission_owner",
               "Study Type": "study_type", "Study Date": "study_date",
               "Resource Type": "fuel_raw", "County": "county", "Nameplate MW": "nameplate_mw",
               "Request Status": "status_raw", "$2022 POI Cost/kW": "poi",
               "$2022 Network Cost/kW": "network", "$2022 Total Cost/kW": "total"}),
    dict(file="ba_costs_2024_clean_data.xlsx", sheet="Project Cost Data", entity=None,
         id_col="Project # in Queued Up", dollar_year=2024, data_through="2024",
         cols={"Balancing Authority": "transmission_owner", "Study Type": "study_type",
               "Study Date": "study_date", "Service Type": "service_raw", "Fuel": "fuel_raw",
               "County": "county", "Nameplate MW": "nameplate_mw",
               "Request Status": "status_raw", "Latitude of POI": "poi_lat",
               "Longitude of POI": "poi_lon", "Transmission Voltage": "voltage_kv",
               "$2024 POI Cost/kW": "poi", "$2024 Network Cost/kW": "network",
               "$2024 Total Cost/kW": "total"}),
]

# Non-ISO cost file uses balancing-authority codes; the queue uses utility entity names.
BA_TO_ENTITY = {"BPA": "BPA", "PAC": "PacifiCorp", "DEC": "Duke", "DEP": "Duke", "DEF": "Duke"}

STATUS_MAP = {"active": "active", "withdrawn": "withdrawn", "complete": "operational",
              "operational": "operational", "suspended": "suspended", "unknown": "unknown"}


def build_costs(projects: pd.DataFrame) -> pd.DataFrame:
    # Lookup from (entity, normalized q_id) to project_uid. Prefer the first record if an ID repeats.
    lookup = (projects.assign(k=projects["q_id"].map(norm_id))
              .sort_values("project_uid")
              .drop_duplicates(["entity", "k"])
              .set_index(["entity", "k"])["project_uid"])

    frames = []
    for spec in COST_FILES:
        raw = pd.read_excel(RAW / spec["file"], sheet_name=spec["sheet"])
        df = raw.rename(columns=spec["cols"]).copy()
        df["source_file"] = spec["file"]
        df["cost_q_id"] = raw[spec["id_col"]].astype("string")
        if spec["entity"]:
            df["entity"] = spec["entity"]
        else:
            df["entity"] = raw["BA"].map(BA_TO_ENTITY)
        df["state"] = clean_text(raw["State"])
        df["queue_date_in_cost_file"] = to_date(raw["Queue Date"]) if "Queue Date" in raw else None
        df["dollar_year_original"] = spec["dollar_year"]
        df["status_data_through"] = spec["data_through"]
        frames.append(df)

    c = pd.concat(frames, ignore_index=True)
    for col in ["transmission_owner", "study_type", "service_raw", "fuel_raw", "county", "status_raw"]:
        c[col] = clean_text(c[col]) if col in c else None
    for col in ["poi", "network", "total", "nameplate_mw", "poi_lat", "poi_lon", "voltage_kv"]:
        c[col] = to_num(c[col]) if col in c else None
    c["study_date"] = to_date(c["study_date"])
    c["study_year"] = pd.to_datetime(c["study_date"]).dt.year.astype("Int64")

    # Convert every cost to 2024 dollars with the GDP deflator.
    factor = c["dollar_year_original"].map(lambda y: GDP_DEFLATOR[TARGET_DOLLAR_YEAR] / GDP_DEFLATOR[y])
    c["poi_cost_per_kw_2024"] = (c["poi"] * factor).round(2)
    c["network_cost_per_kw_2024"] = (c["network"] * factor).round(2)
    c["total_cost_per_kw_2024"] = (c["total"] * factor).round(2)
    # $/kW x MW x 1000 kW/MW = $; report in $ millions.
    c["total_cost_musd_2024"] = (c["total_cost_per_kw_2024"] * c["nameplate_mw"] * 1000 / 1e6).round(2)

    c["fuel_group"] = c["fuel_raw"].str.lower().map(FUEL_GROUP).fillna("Other")
    c["status_in_cost_file"] = c["status_raw"].str.lower().map(STATUS_MAP).fillna("unknown")
    c["county_norm"] = c["county"].map(norm_county)

    # Link to the queue. Non-ISO rows can list two IDs ("G0367, G0368"); match the first.
    first_id = c["cost_q_id"].str.split(",").str[0]
    keys = list(zip(c["entity"], first_id.map(norm_id)))
    c["project_uid"] = [lookup.get(k) for k in keys]
    c["matched_to_queue"] = c["project_uid"].notna()

    # 12 rows are exact duplicates in the source files. Drop them.
    n_before = len(c)
    c = c.drop_duplicates().reset_index(drop=True)
    n_exact_dupes = n_before - len(c)

    c.insert(0, "cost_id", range(1, len(c) + 1))

    # Some projects have several estimates: the non-ISO file gives one per service type (ERIS and
    # NRIS), and some projects were studied more than once. Flag one row per project, by a fixed rule:
    # latest study, then NRIS over ERIS (the fuller service), then the last row. Per-project analyses
    # filter on is_primary_estimate so no project is counted twice.
    svc_rank = c["service_raw"].str.upper().str.contains("NRIS|CAPACITY|CNR", na=False).astype(int)
    order = (c.assign(_svc=svc_rank, _d=pd.to_datetime(c["study_date"]))
             .sort_values(["project_uid", "_d", "_svc", "cost_id"], na_position="first"))
    last = order.drop_duplicates("project_uid", keep="last")["cost_id"]
    c["is_primary_estimate"] = c["project_uid"].isna() | c["cost_id"].isin(last)
    c["estimates_for_project"] = c.groupby("project_uid")["cost_id"].transform("size").where(c["project_uid"].notna(), 1)

    cols = ["cost_id", "project_uid", "matched_to_queue", "is_primary_estimate", "estimates_for_project", "entity", "cost_q_id", "source_file",
            "transmission_owner", "state", "county", "county_norm", "fuel_raw", "fuel_group",
            "nameplate_mw", "service_raw", "study_type", "study_date", "study_year",
            "status_raw", "status_in_cost_file", "status_data_through", "queue_date_in_cost_file",
            "poi_cost_per_kw_2024", "network_cost_per_kw_2024", "total_cost_per_kw_2024",
            "total_cost_musd_2024", "dollar_year_original", "poi_lat", "poi_lon", "voltage_kv"]
    c = c[cols]
    by_file = c.groupby("source_file")["matched_to_queue"].agg(["size", "sum"])
    print(f"costs: {n_before} rows, {n_exact_dupes} exact duplicates dropped, {len(c)} kept, "
          f"{c['is_primary_estimate'].sum()} primary estimates; matched to queue by file:\n" + by_file.to_string())
    return c


# ---------------------------------------------------------------------------
# 3. EIA-860 generators (+ plant attributes)
# ---------------------------------------------------------------------------

def build_eia(projects: pd.DataFrame) -> pd.DataFrame:
    with zipfile.ZipFile(EIA_ZIP) as z:
        with z.open("3_1_Generator_Y2025.xlsx") as f:
            sheets = pd.read_excel(f, sheet_name=None, header=1)
        with z.open("2___Plant_Y2025.xlsx") as f:
            plant = pd.read_excel(f, sheet_name="Plant", header=1)

    def take(df, group, year_col=None, month_col=None):
        out = pd.DataFrame({
            "plant_code": to_num(df["Plant Code"]).astype("Int64"),
            "generator_id": df["Generator ID"].astype("string"),
            "status_group": group,
            "status_code": clean_text(df["Status"]),
            "technology": clean_text(df["Technology"]),
            "prime_mover": clean_text(df["Prime Mover"]),
            "nameplate_mw": to_num(df["Nameplate Capacity (MW)"]),
            "summer_mw": to_num(df["Summer Capacity (MW)"]) if "Summer Capacity (MW)" in df else None,
        })
        out["year"] = to_num(df[year_col]).astype("Int64") if year_col else pd.NA
        out["planned_retirement_year"] = (to_num(df["Planned Retirement Year"]).astype("Int64")
                                          if "Planned Retirement Year" in df else pd.NA)
        return out

    gens = pd.concat([
        take(sheets["Operable"], "operable", "Operating Year"),
        take(sheets["Proposed"], "proposed", "Current Year"),
        take(sheets["Retired and Canceled"], "retired_or_canceled", "Retirement Year"),
    ], ignore_index=True)
    # Drop the trailing footnote rows EIA puts at the bottom of each sheet.
    gens = gens[gens["plant_code"].notna() & gens["generator_id"].notna()].copy()

    p = pd.DataFrame({
        "plant_code": to_num(plant["Plant Code"]).astype("Int64"),
        "plant_name": clean_text(plant["Plant Name"]),
        "utility_name": clean_text(plant["Utility Name"]),
        "state": clean_text(plant["State"]),
        "county": clean_text(plant["County"]),
        "latitude": to_num(plant["Latitude"]),
        "longitude": to_num(plant["Longitude"]),
        "balancing_authority_code": clean_text(plant["Balancing Authority Code"]),
        "nerc_region": clean_text(plant["NERC Region"]),
    }).dropna(subset=["plant_code"])
    g = gens.merge(p, on="plant_code", how="left")
    g["tech_group"] = g["technology"].map(eia_tech_group)
    g["county_norm"] = g["county"].map(norm_county)

    # County name -> FIPS, learned from the queue (which has both). Most common FIPS wins.
    xwalk = (projects.dropna(subset=["state", "county_norm", "fips_code"])
             .groupby(["state", "county_norm"])["fips_code"]
             .agg(lambda s: s.mode().iloc[0]).rename("fips_code").reset_index())
    g = g.merge(xwalk, on=["state", "county_norm"], how="left")

    year_meaning = {"operable": "operating year", "proposed": "planned operating year",
                    "retired_or_canceled": "retirement year"}
    g["year_meaning"] = g["status_group"].map(year_meaning)
    op = g[g["status_group"] == "operable"]
    print(f"eia_generators: {len(g)} rows ({len(op)} operable); "
          f"operable MW with FIPS: {op.loc[op.fips_code.notna(), 'nameplate_mw'].sum() / op['nameplate_mw'].sum():.1%}")
    return g, xwalk


# ---------------------------------------------------------------------------
# 4. Load into DuckDB and build summary tables
# ---------------------------------------------------------------------------

SUMMARY_SQL = """
CREATE TABLE county_summary AS
WITH existing AS (
    SELECT fips_code, any_value(state) AS state, any_value(county) AS county,
           sum(nameplate_mw) AS existing_mw,
           sum(nameplate_mw) FILTER (WHERE tech_group = 'Solar') AS existing_solar_mw,
           sum(nameplate_mw) FILTER (WHERE tech_group IN ('Wind', 'Offshore Wind')) AS existing_wind_mw,
           sum(nameplate_mw) FILTER (WHERE tech_group = 'Battery') AS existing_battery_mw,
           sum(nameplate_mw) FILTER (WHERE tech_group = 'Gas') AS existing_gas_mw,
           count(DISTINCT plant_code) AS existing_plants
    FROM eia_generators WHERE status_group = 'operable' AND fips_code IS NOT NULL
    GROUP BY fips_code
), queue AS (
    SELECT fips_code, any_value(state) AS state, any_value(county) AS county,
           count(*) AS requests_total,
           count(*) FILTER (WHERE q_status = 'active') AS requests_active,
           count(*) FILTER (WHERE q_status = 'withdrawn') AS requests_withdrawn,
           count(*) FILTER (WHERE q_status = 'operational') AS requests_operational,
           sum(mw_total) FILTER (WHERE q_status = 'active') AS active_queue_mw,
           sum(mw_total) FILTER (WHERE q_status = 'active' AND type_group LIKE 'Solar%') AS active_solar_mw,
           sum(mw_total) FILTER (WHERE q_status = 'active' AND type_group = 'Battery') AS active_battery_mw,
           sum(mw_total) FILTER (WHERE q_status = 'active' AND type_group LIKE '%Wind%') AS active_wind_mw,
           sum(mw_total) FILTER (WHERE q_status = 'active' AND type_group LIKE 'Gas%') AS active_gas_mw
    FROM projects WHERE fips_code IS NOT NULL
    GROUP BY fips_code
)
SELECT coalesce(q.fips_code, e.fips_code) AS fips_code,
       coalesce(q.state, e.state) AS state,
       coalesce(q.county, e.county) AS county,
       coalesce(e.existing_mw, 0) AS existing_mw,
       coalesce(e.existing_solar_mw, 0) AS existing_solar_mw,
       coalesce(e.existing_wind_mw, 0) AS existing_wind_mw,
       coalesce(e.existing_battery_mw, 0) AS existing_battery_mw,
       coalesce(e.existing_gas_mw, 0) AS existing_gas_mw,
       coalesce(e.existing_plants, 0) AS existing_plants,
       coalesce(q.requests_total, 0) AS requests_total,
       coalesce(q.requests_active, 0) AS requests_active,
       coalesce(q.requests_withdrawn, 0) AS requests_withdrawn,
       coalesce(q.requests_operational, 0) AS requests_operational,
       coalesce(q.active_queue_mw, 0) AS active_queue_mw,
       coalesce(q.active_solar_mw, 0) AS active_solar_mw,
       coalesce(q.active_battery_mw, 0) AS active_battery_mw,
       coalesce(q.active_wind_mw, 0) AS active_wind_mw,
       coalesce(q.active_gas_mw, 0) AS active_gas_mw,
       round(coalesce(q.active_queue_mw, 0) / nullif(e.existing_mw, 0), 2) AS active_to_existing_ratio
FROM queue q FULL OUTER JOIN existing e USING (fips_code);

CREATE TABLE poi_summary AS
SELECT entity, region, any_value(state) AS state,
       upper(trim(poi_name_clean)) AS poi_name_norm, any_value(poi_name_clean) AS poi_name_example,
       count(*) AS requests_total,
       count(*) FILTER (WHERE q_status = 'active') AS requests_active,
       count(*) FILTER (WHERE q_status = 'withdrawn') AS requests_withdrawn,
       count(*) FILTER (WHERE q_status = 'operational') AS requests_operational,
       coalesce(sum(mw_total) FILTER (WHERE q_status = 'active'), 0) AS active_mw,
       round(count(*) FILTER (WHERE q_status = 'withdrawn') * 1.0 / count(*), 3) AS withdrawn_share,
       min(q_year) AS first_request_year, max(q_year) AS last_request_year
FROM projects WHERE poi_name_clean IS NOT NULL
GROUP BY entity, region, upper(trim(poi_name_clean));
"""


# ---------------------------------------------------------------------------
# 5. Context for the agent
# ---------------------------------------------------------------------------

TABLE_DOCS = {
    "projects": {
        "description": "One row per generator interconnection request in US transmission queues, through end of 2025 (LBNL Queued Up 2026). Generation, storage and hybrids only: load requests (e.g. data centers), distribution-connected and behind-the-meter projects are NOT included.",
        "grain": "one interconnection request", "primary_key": "project_uid",
        "source": "LBNL & GridTracker, Queued Up 2026 Edition, CC BY 4.0",
    },
    "costs": {
        "description": "Project-level interconnection cost estimates from LBNL's cost studies, unified across six files and converted to 2024 dollars. Some projects have more than one estimate (one per service type in the non-ISO file, or several studies). For per-project analysis, filter is_primary_estimate = true so each project counts once.",
        "grain": "one cost estimate (per project, service type and study)", "primary_key": "cost_id",
        "source": "LBNL interconnection cost studies (MISO 2022, PJM 2023, SPP 2023, ISO-NE 2023, NYISO 2023, non-ISO 2026)",
    },
    "eia_generators": {
        "description": "Every US generator of 1 MW or more from EIA-860 2025 early release: operable, proposed, and retired/canceled, with plant location and balancing authority.",
        "grain": "one generator at one plant, in one status group", "primary_key": "(plant_code, generator_id, status_group)",
        "source": "US EIA Form 860, 2025 early release",
    },
    "county_summary": {
        "description": "Precomputed per county: existing operable capacity (EIA-860) next to interconnection queue activity (LBNL). Use for siting and saturation questions.",
        "grain": "one county (5-digit FIPS)", "primary_key": "fips_code",
        "source": "derived from projects + eia_generators",
    },
    "poi_summary": {
        "description": "Precomputed per point of interconnection (substation or line tap, as named in the queue): how many requests target it and what happened to them. Use for congestion questions.",
        "grain": "one POI name within one grid operator", "primary_key": "(entity, poi_name_norm)",
        "source": "derived from projects. POI names are not standardized across operators, so the same substation can appear under slightly different names.",
    },
}

COLUMN_DOCS = {
    "projects": {
        "project_uid": "Unique ID built here: entity|q_id, with #2, #3 added when the source reuses an ID.",
        "entity": "Transmission provider that runs the queue: one of 7 ISOs (CAISO, ERCOT, ISO-NE, MISO, NYISO, PJM, SPP) or a utility.",
        "q_id": "Queue position as published by the operator. Only unique within an entity.",
        "q_id_note": "Set when the source ID is unreliable (9 CLPT IDs were turned into dates by Excel).",
        "shares_q_id_with_other_record": "True when another record in the same entity uses the same q_id.",
        "region": "Standardized region: the 7 ISOs, plus 'West' and 'Southeast' for non-ISO utilities.",
        "is_iso": "True if the region is an ISO/RTO market.",
        "utility": "Utility name (same as entity outside ISOs).",
        "state": "Two-letter state.", "county": "First county listed for the project.",
        "county_norm": "Normalized county name used for joins.",
        "fips_code": "5-digit county FIPS code (text). Joins to county_summary and eia_generators.",
        "poi_name": "Point of interconnection as written by the operator, including placeholders like 'Unknown', 'TBD', '138kV'.",
        "poi_name_clean": "poi_name with placeholders removed (unknown/TBD, bare voltages, generic 'distribution feeder'). Use this for grouping by location.",
        "project_name": "Project name. Missing for most requests.",
        "developer": "Developer name, not standardized. Only populated in ERCOT and NYISO in any volume.",
        "cluster": "Study cluster, where the operator studies requests in groups.",
        "service": "Interconnection service: NRIS, ERIS, NRIS/ERIS or Other. ERCOT is energy-only, so it has no NRIS/ERIS split; NYISO uses its own ERIS/CRIS terms.",
        "project_type": "Generation, Surplus, Upgrade or Replacement.",
        "q_status": "Current status: active, withdrawn, suspended, operational, unknown. 'active' includes projects with a signed IA not yet operating.",
        "ia_phase": "Standardized study phase. LBNL imputes 'IA Executed' for every operational project.",
        "ia_phase_raw": "Study phase as written by the operator.",
        "type_clean": "Resource type as LBNL standardized it, with hybrids combined (e.g. 'Solar+Battery'). 49 distinct values.",
        "type_group": "Resource type collapsed to a short shared list (also used in costs and EIA): Solar, Wind, Offshore Wind, Battery, Gas, Coal, Nuclear, Hydro, Geothermal, Oil, Solar+Battery, Wind+Battery, Gas+Battery, Offshore Wind+Battery, Other Hybrid, Other.",
        "is_hybrid": "True if the request has a second resource type.",
        "type_1": "Primary resource type.", "type_2": "Second resource type (hybrids).", "type_3": "Third resource type.",
        "mw_1": "Capacity of type_1 in MW.",
        "mw_2": "Capacity of type_2 in MW. Often missing for hybrids: LBNL's report imputes it, this public file does not.",
        "mw_3": "Capacity of type_3 in MW.",
        "mw_total": "mw_1 + mw_2 + mw_3. Undercounts hybrid storage where mw_2 is missing. Null for the 16 requests with negative mw_1.",
        "mw_note": "Set when the source capacity is unusable (negative values, likely derates).",
        "q_date": "Date the request entered the queue.",
        "prop_date": "Developer's proposed online date at application. Often revised later.",
        "ia_date": "Date the interconnection agreement was signed.",
        "on_date": "Date the project began commercial operation.",
        "wd_date": "Date the request was withdrawn.",
        "q_year": "Year entered queue.", "prop_year": "Proposed online year.",
        "ia_year": "Year IA signed.", "on_year": "Year operational.", "wd_year": "Year withdrawn.",
        "days_ir_to_ia": "Days from request to signed IA. Null if either date is missing or out of order.",
        "days_ir_to_cod": "Days from request to commercial operation. Null if either date is missing or out of order.",
        "days_ia_to_cod": "Days from signed IA to commercial operation.",
        "days_ir_to_wd": "Days from request to withdrawal.",
    },
    "costs": {
        "cost_id": "Row ID.",
        "project_uid": "Link to projects.project_uid. Null when the cost row could not be matched.",
        "matched_to_queue": "True if linked to a projects row.",
        "is_primary_estimate": "True for exactly one estimate per project (latest study, then NRIS over ERIS). Filter on it for per-project counts, averages and medians.",
        "estimates_for_project": "How many estimates this project has in the table (1 for most).",
        "entity": "Queue entity (MISO, PJM, SPP, ISO-NE, NYISO, BPA, Duke, PacifiCorp).",
        "cost_q_id": "Queue ID as written in the cost file.",
        "source_file": "Which LBNL cost file the row came from.",
        "transmission_owner": "Transmission owner or balancing authority named in the study.",
        "state": "State.", "county": "County (ISO-NE, NYISO and non-ISO only).",
        "county_norm": "Normalized county name.",
        "fuel_raw": "Fuel as written in the cost file.",
        "fuel_group": "Fuel mapped to the shared type_group vocabulary.",
        "nameplate_mw": "Requested capacity in MW at the time of the study.",
        "service_raw": "Service type as written (labels differ by operator).",
        "study_type": "Study the cost comes from (Feasibility, System Impact, Facility, Cluster...). Later studies are more reliable.",
        "study_date": "Date of that study.", "study_year": "Year of that study.",
        "status_raw": "Status as written in the cost file.",
        "status_in_cost_file": "Status at the time the cost file was compiled (NOT current). Use projects.q_status for current status.",
        "status_data_through": "When the cost file's status snapshot was taken.",
        "queue_date_in_cost_file": "Queue entry date as written in the cost file.",
        "poi_cost_per_kw_2024": "Point-of-interconnection cost, 2024 $/kW (substation and line to the grid).",
        "network_cost_per_kw_2024": "Network upgrade cost, 2024 $/kW (upgrades deeper in the grid). Null for some ISO-NE rows.",
        "total_cost_per_kw_2024": "POI + network cost, 2024 $/kW.",
        "total_cost_musd_2024": "total_cost_per_kw_2024 x nameplate_mw, in millions of 2024 dollars.",
        "dollar_year_original": "Dollar year in the source file before conversion (2022 for ISO files, 2024 for non-ISO).",
        "poi_lat": "POI latitude (non-ISO file only).", "poi_lon": "POI longitude (non-ISO file only).",
        "voltage_kv": "Transmission voltage at the POI in kV (non-ISO file only).",
    },
    "eia_generators": {
        "plant_code": "EIA plant ID.", "generator_id": "Generator ID within the plant.",
        "status_group": "operable, proposed, or retired_or_canceled.",
        "status_code": "EIA status code (OP operating, SB standby, OS out of service, OA out of service but expected back; proposed: P, L, T, U, V, TS, OT; retired: RE, CN).",
        "technology": "EIA technology description.",
        "tech_group": "Technology mapped to the shared type_group vocabulary.",
        "prime_mover": "EIA prime mover code.",
        "nameplate_mw": "Nameplate capacity, MW.", "summer_mw": "Summer capacity, MW.",
        "year": "See year_meaning: operating year, planned operating year, or retirement year.",
        "year_meaning": "What the year column means for this row's status_group.",
        "planned_retirement_year": "Planned retirement year if reported.",
        "plant_name": "Plant name.", "utility_name": "Operating utility.",
        "state": "State.", "county": "County.", "county_norm": "Normalized county name.",
        "fips_code": "County FIPS mapped from the queue's county names. Null where no queue project shares the county.",
        "latitude": "Plant latitude.", "longitude": "Plant longitude.",
        "balancing_authority_code": "Balancing authority (e.g. PJM, ERCO, MISO, CISO).",
        "nerc_region": "NERC region.",
    },
}

DATA_GAPS = [
    {"topic": "data centers / load", "applies_to": "projects",
     "gap": "The queue covers generation and storage only. Load requests (data centers, factories) are excluded, so questions about data center demand cannot be answered from this data."},
    {"topic": "hybrid storage", "applies_to": "projects.mw_2, projects.mw_total",
     "gap": "Storage capacity for hybrids is often missing. LBNL imputes it in the published report but not in this file, so storage totals here are lower than the report's."},
    {"topic": "dates", "applies_to": "projects.on_date, wd_date, ia_date, on_year, wd_year, ia_year, days_*",
     "gap": "Dates are incomplete. Always report how many projects had the date (e.g. 'based on N of M operational projects'). IA dates are especially thin in PJM (~3%), SPP (0%), NYISO and the Southeast."},
    {"topic": "timing bias", "applies_to": "projects.days_ir_to_cod, days_ia_to_cod, on_year",
     "gap": "Time-to-operation figures only include projects that already reached operation. Many recent requests are still in the queue and will likely take longer, so these medians understate how long projects entering the queue today should expect to wait, especially for recent cohorts. Say so whenever reporting time to operation."},
    {"topic": "developer", "applies_to": "projects.developer",
     "gap": "Developer names are only populated in volume for ERCOT and NYISO (22% overall). Developer rankings elsewhere are not possible."},
    {"topic": "cost recency", "applies_to": "costs.poi_cost_per_kw_2024, network_cost_per_kw_2024, total_cost_per_kw_2024, total_cost_musd_2024",
     "gap": "Cost studies end in 2021 (MISO, ISO-NE, NYISO), 2022 (PJM), 2023 (SPP) and 2024 (non-ISO). There are no cost data for CAISO, ERCOT, or most non-ISO utilities. Costs for requests submitted after those dates are not covered."},
    {"topic": "cost status", "applies_to": "costs.status_in_cost_file",
     "gap": "The cost files' status is a snapshot from when they were compiled. Join to projects for current status (q_status)."},
    {"topic": "multiple estimates", "applies_to": "costs.total_cost_per_kw_2024, poi_cost_per_kw_2024, network_cost_per_kw_2024, total_cost_musd_2024",
     "gap": "Some projects have several cost estimates (per service type or per study). Filter is_primary_estimate = true for per-project statistics, or the project is counted more than once."},
    {"topic": "cost estimates", "applies_to": "costs.study_type, total_cost_per_kw_2024, total_cost_musd_2024",
     "gap": "Costs are study estimates, not final paid costs. Early-stage studies (Feasibility) are less reliable than Facility studies."},
    {"topic": "location", "applies_to": "projects",
     "gap": "The queue has county, not coordinates. Multi-county projects list only the first county."},
    {"topic": "county join", "applies_to": "eia_generators.fips_code, county_summary.existing_mw, active_to_existing_ratio",
     "gap": "EIA counties are mapped to FIPS via the queue's county names. Counties with no queue project get no FIPS, so a small share of existing capacity is missing from county_summary."},
    {"topic": "POI names", "applies_to": "projects.poi_name, poi_name_clean, poi_summary.poi_name_norm, poi_name_example",
     "gap": "POI names are free text per operator. The same substation can appear under different spellings, so POI counts are approximate."},
    {"topic": "duplicate queue IDs", "applies_to": "projects.project_uid",
     "gap": "135 of 38,201 records reuse an (entity, q_id) already in the file, usually a sparser copy of the same project; their project_uid ends in '#2', '#3'. Including them changes totals by under 0.4%. Counts in this app include them, matching LBNL's published totals. No further investigation is needed."},
    {"topic": "date order", "applies_to": "projects.days_ir_to_cod, days_ia_to_cod",
     "gap": "About 50 projects have an operation date before their queue date (usually uprates or re-filed requests). Their durations are set to null rather than negative."},
    {"topic": "negative capacity", "applies_to": "projects.mw_1",
     "gap": "16 requests list negative MW in the source (likely derates of existing plants). mw_1 keeps the raw value; mw_total is null for them and mw_note explains why."},
    {"topic": "ISO-NE timing", "applies_to": "projects.on_date, days_ir_to_cod, days_ia_to_cod",
     "gap": "ISO-NE has no operation dates in this file, so time-to-operation cannot be computed for ISO-NE."},
    {"topic": "upgrade costs and reasons", "applies_to": "all",
     "gap": "Why a project withdrew is not recorded anywhere in this data. Cost can be correlated with withdrawal but not shown to cause it."},
    {"topic": "EIA early release", "applies_to": "eia_generators",
     "gap": "EIA-860 2025 is an early release and may change slightly in the final version."},
]

JOINS = [
    {"from": "costs.project_uid", "to": "projects.project_uid", "type": "many-to-one",
     "note": "Project-level. 99-100% of ISO cost rows match; non-ISO rows match where the file lists a Queued Up ID. A project can have several estimates: join with costs.is_primary_estimate = true to get one row per project."},
    {"from": "projects.fips_code", "to": "county_summary.fips_code", "type": "many-to-one", "note": "County level."},
    {"from": "eia_generators.fips_code", "to": "county_summary.fips_code", "type": "many-to-one", "note": "County level."},
    {"from": "projects.fips_code", "to": "eia_generators.fips_code", "type": "many-to-many",
     "note": "County level only. There is no project-level link between queue requests and EIA generators; aggregate both sides by county before joining."},
    {"from": "projects.(entity, upper(trim(poi_name_clean)))", "to": "poi_summary.(entity, poi_name_norm)", "type": "many-to-one", "note": "POI level."},
]

GLOSSARY = {
    "interconnection queue": "The line of generation projects waiting for grid operator studies and approval to connect to the transmission grid.",
    "IR": "Interconnection request: when a project enters the queue (q_date).",
    "IA": "Interconnection agreement: the signed contract to connect (ia_date). Usually the last major step before construction.",
    "COD": "Commercial operation date: when the project started operating (on_date).",
    "withdrawn": "The developer left the queue. Most requests end this way.",
    "completion rate": "Share of requests (or MW) from a given entry-year cohort that reached operational status. Use old cohorts (e.g. entered 2000-2020) because recent requests have not had time to finish.",
    "POI": "Point of interconnection: the substation or line where the project connects.",
    "POI cost": "Cost of facilities to connect at the POI.",
    "network upgrade cost": "Cost of upgrades deeper in the grid triggered by the project. The main driver of recent cost increases.",
    "NRIS / ERIS": "Network vs. energy-only interconnection service. NRIS allows the plant to count toward capacity; ERIS is energy only.",
    "ISO / RTO": "Independent system operator / regional transmission organization: the 7 market operators (CAISO, ERCOT, ISO-NE, MISO, NYISO, PJM, SPP).",
    "hybrid": "A project combining resource types at one site, usually solar or wind plus battery.",
    "FIPS": "5-digit county code: 2-digit state + 3-digit county.",
}


def profile_columns(con, table: str) -> list[dict]:
    """Type, fill rate, and either the full value list (<= 60 values) or min/max for each column."""
    out = []
    total = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    for name, dtype, *_ in con.execute(f"DESCRIBE {table}").fetchall():
        nonnull, distinct = con.execute(
            f'SELECT count("{name}"), count(DISTINCT "{name}") FROM {table}').fetchone()
        col = {"name": name, "type": dtype,
               "description": COLUMN_DOCS.get(table, {}).get(name, ""),
               "fill_pct": round(100 * nonnull / total, 1) if total else 0,
               "distinct_values": distinct}
        if dtype in ("VARCHAR", "BOOLEAN") and 0 < distinct <= 60:
            rows = con.execute(f'SELECT "{name}", count(*) FROM {table} WHERE "{name}" IS NOT NULL '
                               f'GROUP BY 1 ORDER BY 2 DESC').fetchall()
            col["values"] = {str(v): n for v, n in rows}
        elif dtype not in ("VARCHAR", "BOOLEAN") and nonnull:
            lo, hi = con.execute(f'SELECT min("{name}"), max("{name}") FROM {table}').fetchone()
            col["min"], col["max"] = str(lo), str(hi)
        out.append(col)
    return out


def fill_by_region(con) -> dict:
    """Fill rates for the patchy projects columns, per region, so the agent can warn per region."""
    cols = ["prop_date", "ia_date", "on_date", "wd_date", "ia_phase", "county", "poi_name",
            "developer", "service", "project_name"]
    sel = ", ".join(f'round(100.0 * count("{c}") / count(*), 0) AS "{c}"' for c in cols)
    df = con.execute(f"SELECT region, count(*) AS n, {sel} FROM projects GROUP BY region ORDER BY region").df()
    # Dates that only exist for some statuses: measure them against the right denominator.
    status_based = con.execute("""
        SELECT region,
               count(*) FILTER (WHERE q_status='operational') AS operational_n,
               round(100.0 * count(on_date) FILTER (WHERE q_status='operational') / nullif(count(*) FILTER (WHERE q_status='operational'),0), 0) AS on_date_pct_of_operational,
               count(*) FILTER (WHERE q_status='withdrawn') AS withdrawn_n,
               round(100.0 * count(wd_date) FILTER (WHERE q_status='withdrawn') / nullif(count(*) FILTER (WHERE q_status='withdrawn'),0), 0) AS wd_date_pct_of_withdrawn
        FROM projects GROUP BY region ORDER BY region""").df()
    return {"column_fill_pct_by_region": df.to_dict(orient="records"),
            "status_date_fill_by_region": status_based.to_dict(orient="records")}


def write_context(con) -> None:
    CONTEXT.mkdir(parents=True, exist_ok=True)
    schema = {"database": "grid.duckdb", "built": date.today().isoformat(),
              "dollar_basis": f"All costs in {TARGET_DOLLAR_YEAR} dollars (BEA GDP deflator, FRED A191RD3A086NBEA: 2022={GDP_DEFLATOR[2022]}, 2024={GDP_DEFLATOR[2024]}).",
              "tables": {}}
    for t, doc in TABLE_DOCS.items():
        rows = con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        schema["tables"][t] = {**doc, "rows": rows, "columns": profile_columns(con, t)}
    (CONTEXT / "schema.json").write_text(json.dumps(schema, indent=2, default=str))
    (CONTEXT / "joins.json").write_text(json.dumps(JOINS, indent=2))
    (CONTEXT / "data_gaps.json").write_text(json.dumps(DATA_GAPS, indent=2))
    (CONTEXT / "glossary.json").write_text(json.dumps(GLOSSARY, indent=2))
    (CONTEXT / "coverage.json").write_text(json.dumps(fill_by_region(con), indent=2, default=str))
    print(f"context: wrote {len(list(CONTEXT.glob('*.json')))} files to {CONTEXT.relative_to(ROOT)}")


# ---------------------------------------------------------------------------

def main() -> None:
    projects = build_projects()
    costs = build_costs(projects)
    eia, _ = build_eia(projects)

    OUT_DB.unlink(missing_ok=True)
    con = duckdb.connect(str(OUT_DB))
    con.register("projects_df", projects)
    con.register("costs_df", costs)
    con.register("eia_df", eia)
    con.execute("CREATE TABLE projects AS SELECT * FROM projects_df")
    con.execute("CREATE TABLE costs AS SELECT * FROM costs_df")
    con.execute("""CREATE TABLE eia_generators AS SELECT plant_code, generator_id, status_group,
                   status_code, technology, tech_group, prime_mover, nameplate_mw, summer_mw, year,
                   year_meaning, planned_retirement_year, plant_name, utility_name, state, county,
                   county_norm, fips_code, latitude, longitude, balancing_authority_code, nerc_region
                   FROM eia_df""")
    con.execute(SUMMARY_SQL)

    # Sanity checks: fail the build loudly rather than ship a broken database.
    assert con.execute("SELECT count(*) = count(DISTINCT project_uid) FROM projects").fetchone()[0], "project_uid not unique"
    assert con.execute("SELECT count(*) FROM costs c LEFT JOIN projects p USING (project_uid) "
                       "WHERE c.project_uid IS NOT NULL AND p.project_uid IS NULL").fetchone()[0] == 0, "dangling cost links"
    assert con.execute("SELECT count(*) FROM county_summary WHERE fips_code IS NULL").fetchone()[0] == 0

    write_context(con)
    con.close()
    print(f"built {OUT_DB.relative_to(ROOT)} ({OUT_DB.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
