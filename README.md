# GridQueue

Ask plain-English questions about the US grid interconnection queue. GridQueue answers with SQL over public LBNL and EIA data, checks its own numbers, and shows every query it ran.

**Live app:** https://gridqueue-git-808237570445.europe-west1.run.app (Columbia sign-in)  
**Path:** Text2SQL  
**Author:** sk5694

## The project

More than 8,500 power projects (1,865 GW) are waiting for permission to connect to the US grid. Most will never be built: of the requests filed from 2000 to 2020, 71% withdrew and 19% reached operation. Where projects queue, what they cost to connect and how long they wait decide where new generation gets built.

**Who it's for:** developers deciding where to site a project, energy analysts tracking the backlog, and policy and ratepayer groups who need numbers they can source. Today they answer these questions by downloading LBNL's spreadsheets and writing pivot tables. GridQueue lets them ask instead, and shows the SQL so they can check it.

**What it can't answer:** electricity demand (the queues list generation and storage, not data centers or other large loads), interconnection costs in CAISO and ERCOT (the LBNL cost studies don't cover them), and final costs (the studies are estimates). It says so instead of guessing.

## How it works

```
question ─► Planner ─► SQL Workers (in parallel) ─► Verifier ─► Writer ─► answer + tables/charts/maps
              │            │                           │           │
          splits it    query DuckDB,              re-runs its    writes the text; code
          into 1-4     one sub-question each      own checks,    draws visuals from the
          parts                                   can send a     verified SQL; a numbers
                                                  worker back    check traces every figure
```

- **Planner** reads the schema and known data gaps, then splits the question into 1 to 4 independent sub-questions (or asks one clarifying question, or says the data can't answer it).
- **SQL Workers** each answer one sub-question with read-only SQL and submit a structured finding: the answer, the final SQL, sample size, caveats and confidence.
- **Verifier** checks every finding with up to 2 queries of its own and can request one follow-up round.
- **Writer** writes the answer and picks visuals. It never types a table: code re-runs the finding's SQL to draw tables, charts (bar, stacked bar, line, scatter) and county maps.
- **Numbers check:** code compares every number in the answer with the query results (allowing for rounding and unit changes). A number it can't trace goes back to the Writer for one revision; if it still can't be traced, the sentence is removed (outside the opening answer, which is flagged instead). It can't catch a ratio written in words ("nearly nine times").

Every model call has a timeout and retries with backoff; each stage has a time budget, so an answer arrives within about 4 minutes even when a call stalls. The page streams progress live (`/chat/stream`); `/chat` returns `{response, session_id, tool_calls}`, with each tool call tagged by the `agent` that made it.

## The data

| Source | What it is | Size |
|---|---|---|
| [LBNL Queued Up 2026](https://emp.lbl.gov/queues) | Every interconnection request in US transmission queues through 2025 | 15.6 MB xlsx |
| [EIA-860 2025](https://www.eia.gov/electricity/data/eia860/) (early release) | Every generator of 1 MW or more: operable, proposed, retired | 23.6 MB zip |
| [LBNL interconnection cost studies](https://emp.lbl.gov/interconnection_costs) | Upgrade cost estimates from PJM, MISO, SPP, NYISO, ISO-NE and non-ISO utilities | 6 xlsx, 13.4 MB |

The raw files are in `data/raw/`. **`build_db.py` runs once, offline,** and writes a 7.9 MB read-only DuckDB file plus the context the agents read at run time:

```bash
uv run --group build build_db.py      # data/raw/* -> data/grid.duckdb + data/context/*.json
```

| Table | Rows | Built from |
|---|---:|---|
| `projects` | 38,201 | Queued Up: cleaned dates, statuses, resource groups, county FIPS codes |
| `eia_generators` | 36,571 | EIA-860: generators with status and technology groups |
| `costs` | 5,485 | Six cost files unified, converted to 2024 dollars, one primary estimate flagged per project |
| `county_summary` | 2,634 | Existing operable MW (EIA) next to queue activity (LBNL), per county |
| `poi_summary` | 24,524 | Requests and outcomes per point of interconnection |

What the build step does that a plain agent can't do at query time:

- **Cleans the raw files:** repairs garbled characters in names, turns "NA" strings into real nulls, drops an exact duplicate cost row, and nulls negative capacities and durations (operation dates before request dates).
- **Normalizes categories:** statuses, regions and resource groups (e.g. `Solar+Battery`), with every valid value stored so a guardrail can reject a misspelled filter and suggest the right one.
- **Joins the sources:** county FIPS codes on both datasets, costs matched to queue projects, and dollars converted to 2024 with the GDP deflator.
- **Writes context files** (`data/context/`): column descriptions, join paths, a glossary, coverage by region and 18 known data gaps (for example: hybrid storage MW is undercounted; medians of finished projects understate today's waits). The relevant gaps are attached to query results automatically.

## Run it locally

```bash
gcloud auth application-default login     # Gemini 3.5 Flash on Vertex AI
uv sync
uv run uvicorn app:app --port 8000        # open http://localhost:8000
uv run pytest                             # 118 tests, no model calls
```

`GRIDQUEUE_REPLAY=runs uv run uvicorn app:app` replays saved runs instead of calling the model, for working on the page.

## Sample queries

Each answer below is checked against a direct SQL query. "No tools" is the same model (Gemini 3.5 Flash) with no tools and no system prompt. Full runs: `uv run python scripts/compare.py samples`.

### 1. How many new interconnection requests were filed each year since 2010, by resource type?

**Expected** (`projects`, grouped by `q_year` and `type_group`): 31,006 requests from 2010 to 2025, rising from 1,266 in 2010 to a peak of 3,435 in 2022. Standalone battery requests grew from 6 (2010) to 1,088 (2023); solar peaked at 1,232 (2020); wind at 375 (2016).

| | Answer |
|---|---|
| **No tools** | A year-by-year table of approximate gigawatts ("~100 GW" in 2010, "~901 GW" in 2023), not request counts, and "more than 2,500 new project requests" in 2023. The data has 3,400. Every figure is marked approximate; none comes from a query. |
| **GridQueue** (56 s) | 31,006 requests. Battery rose from 6 to 1,088 (2010 to 2023), solar+battery from 11 to 831, wind peaked in 2016 (375), solar in 2020 (1,232). Stacked bar by year in the fixed resource colors. All numbers match. |

### 2. Which counties have the most active queue capacity relative to the generation already operating there?

**Expected** (`county_summary`, counties with at least 50 MW operating): Deschutes, OR at 185.7× (9,505 MW queued vs. 51 MW operating), Crook, OR at 97.9× and Colorado, TX at 68.5×. Without the 50 MW floor the top is Colleton, SC at 2,299× on a 2.5 MW base, which says more about the base than the queue.

| | Answer |
|---|---|
| **No tools** | Lists regions and counties from general knowledge (Imperial and Kern in California, Esmeralda and Nye in Nevada, Pecos, Reeves and Webb in Texas) with ratios of "10 to over 100 times". No county-level numbers. Only Nye appears in the actual top 10, at 39.6×. |
| **GridQueue** (40 s) | Deschutes, Crook and Colorado County with the ratios and MW above, a county map (switchable between ratio, operating MW and queued MW) and a top 10 table. States the 50 MW floor and why. |

### 3. Do projects that withdrew face higher interconnection costs than projects that got built? Compare by region.

**Expected** (`costs` primary estimates joined to `projects` on `project_uid`, 2024 $/kW): withdrawn projects have higher median costs in every region except NYISO. PJM $109 vs. $21, MISO $221 vs. $36, West $450 vs. $167. NYISO $96 vs. $105.

| | Answer |
|---|---|
| **No tools** | "Yes", with cost ranges recalled from LBNL reports (PJM completed $15 to $30/kW, withdrawn $150 to $250+). Claims withdrawn NYISO projects face costs "3 to 4 times higher". The data shows the opposite. |
| **GridQueue** (52 s) | Same direction with exact medians and means for 7 regions, and the NYISO exception called out. Grouped bar in the status colors plus a table, using one primary estimate per project so studies aren't double-counted. |

## Where a plain run_sql agent gets it wrong

The baseline (`gridqueue/baseline.py`) is the starter-style agent: the same model and the same read-only database, the table and column names in its instructions, one `run_sql` tool and a 15-turn cap. It has none of the build step's context, category values, data-gap notes or guardrails, and no Verifier. We asked it and GridQueue 6 questions (`uv run python scripts/compare.py failures`). It answered 3 correctly, gave a mix of four different rates for a fourth, and got these two wrong.

### 1. Which county has the highest ratio of active queue capacity to existing generation?

| | Answer |
|---|---|
| **Plain agent** | Colleton County, SC, at **2,299×**: 5,748 MW queued against **2.5 MW** of existing generation. It then lists counties with 0 MW as "infinite" ratios. |
| **GridQueue** | Deschutes County, OR, at **185.7×** (9,505 MW vs. 51 MW), then Crook, OR and Colorado, TX, among the 1,497 counties with at least 50 MW operating. |
| **Why** | The plain agent sorted the ratio column and stopped. A 2.5 MW base turns any queue into an outlier. GridQueue's agents share a domain rule (ratios need a base of at least 50 MW), and the Verifier re-runs the top five and the county count. |

<details><summary>Plain agent trace (5 queries)</summary>

```
1. SELECT fips_code, state, county, existing_mw, active_queue_mw, active_to_existing_ratio
   FROM county_summary ORDER BY active_to_existing_ratio DESC LIMIT 10
   -> Colleton SC 2.5 MW existing, 5,747.79 MW queued, ratio 2299.12; Labette KS 1.0 MW, 1410.0; ...
2. SELECT ... FROM county_summary WHERE existing_mw = 0 AND active_queue_mw > 0 ORDER BY active_queue_mw DESC LIMIT 10
   -> Duval TX 0 MW / 3,833 MW; City of Chesapeake VA 0 MW / 3,034 MW; ...
3. SELECT active_to_existing_ratio, count(*) FROM county_summary WHERE existing_mw = 0 GROUP BY 1
   -> [null, 594]
4. SELECT sum(nameplate_mw) FROM eia_generators WHERE state = 'SC' AND county_norm = 'colleton'
   -> [null]
5. SELECT * FROM county_summary WHERE county = 'Colleton' AND state = 'SC'
   -> existing_mw 2.5, active_queue_mw 5,747.79
=> "The county with the highest defined ratio ... is Colleton County, South Carolina ... Ratio: 2,299.12"
```
</details>

<details><summary>GridQueue trace (15 steps, 5 queries)</summary>

```
[Planner] describe_table county_summary; get_data_gaps 'county_summary'
[Planner] submit_plan: one sub-question, "among counties with at least 50 MW of existing generation"
[SQL Worker 1] describe_table; get_data_gaps 'county active queue existing generation'
[SQL Worker 1] run_sql: SELECT fips_code, county, state, existing_mw, active_queue_mw,
                 active_queue_mw / existing_mw AS active_to_existing_ratio
               FROM county_summary WHERE existing_mw >= 50 ORDER BY active_to_existing_ratio DESC  -> 1,497 rows
[SQL Worker 1] run_sql: recompute the ratio for Deschutes (fips 41017) against the stored column  -> 1 row
[SQL Worker 1] submit_finding (high confidence)
[Verifier] get_data_gaps; run_sql: sample of county_summary  -> 2 rows
[Verifier] run_sql: top 5 counties with existing_mw >= 50, ratio recomputed  -> 5 rows
[Verifier] run_sql: COUNT(*), COUNT(existing_mw >= 50) FROM county_summary  -> 2,634 / 1,497
[Verifier] submit_verification: all checks passed
[Writer] show_map (ratio by county); show_table (top 10)
=> "Deschutes County, Oregon has the highest ratio ... 185.7, with 9,505 MW in the active queue
    compared to 51 MW of existing operating capacity ..."
```
</details>

### 2. How much battery storage capacity is waiting in the interconnection queue?

**Expected** (`projects`, active, summing `mw_n` where `type_n` is Battery or Other Storage): **552.7 GW**. That's 391.5 GW of standalone storage plus 161.2 GW recorded inside hybrids. Many hybrids have no storage MW recorded, so this is an undercount.

| | Answer |
|---|---|
| **Plain agent** | No answer. It ran 15 queries, hit the 15-turn cap and stopped (in both runs we tried). Along the way it computed 922 GW, 960 GW and 544 GW without settling on a definition. |
| **GridQueue** | **552,750 MW** of storage across 2,718 projects with recorded storage MW, with the undercount caveat: 1,266 of the 3,984 active projects that list storage have no storage MW recorded. |
| **Why** | Hybrids store their total in `mw_total`, so "battery capacity" is ambiguous unless you know to sum only the storage components. GridQueue got this wrong too at first. Its first answer was 922,496 MW, because it counted solar+battery hybrids' full capacity. We added one rule to the shared domain context (storage is the sum of `mw_n` where `type_n` is a storage type), and the rerun is above. |

<details><summary>Plain agent trace (15 queries, no answer)</summary>

```
 1. SELECT DISTINCT q_status FROM projects                                   -> 5 values
 2. SELECT DISTINCT type_group FROM projects                                 -> 16 values
 3. SELECT DISTINCT type_1 FROM projects LIMIT 20                            -> 14 values
 4. SELECT type_group, type_1, mw_1, type_2, mw_2, type_3, mw_3, mw_total FROM projects
    WHERE type_group LIKE '%Battery%' LIMIT 10                               -> Solar+Battery rows with mw_2 null
 5. SELECT SUM(active_battery_mw) FROM county_summary                        -> 390,824
 6. SELECT type_group, COUNT(*), SUM(mw_total), SUM(battery mw_n) ... GROUP BY type_group  -> 15 rows
 7. SELECT SUM(mw_n where type_n = 'Battery') FROM projects WHERE q_status = 'active'    -> 544,128
 8. SELECT standalone_battery_mw, hybrid_battery_mw, total_project_capacity_with_battery -> 391,451 / 158,798 / 922,496
 9. SELECT SUM(mw_total), SUM(battery mw_n) ... WHERE type_group = 'Battery'  -> 391,451 / 385,330
10. SELECT SUM(mw_total) ... WHERE type_group LIKE '%Battery%' OR type_n = 'Battery'      -> 959,640
11. SELECT COUNT(*) ... same filter                                          -> 3,984
12. SELECT q_status, COUNT(*), standalone and hybrid battery MW ... GROUP BY q_status     -> 5 rows
13. SELECT type_group, COUNT(*) ... WHERE type_n = 'Other Storage' GROUP BY 1            -> Battery 111, Other Hybrid 13
14. SELECT SUM(mw_total) ... WHERE type_group = 'Battery' AND fips_code IS NULL          -> 628
15. SELECT SUM(mw_total) FROM projects WHERE q_status = 'active'             -> 1,864,963
=> MaxTurnsExceeded: Max turns (15) exceeded
```
</details>

<details><summary>GridQueue trace (17 steps, 7 queries)</summary>

```
[Planner] describe_table projects; get_data_gaps 'active battery storage'
[Planner] submit_plan: sum mw_1..mw_3 only where type_n is Battery or Other Storage; warn hybrid storage is undercounted
[SQL Worker 1] get_data_gaps; describe_table; get_column_values type_1 ~ 'Battery', ~ 'Storage'
[SQL Worker 1] run_sql: total storage MW and project count (storage components only)        -> 1 row
[SQL Worker 1] run_sql: active projects by storage type with and without recorded MW        -> 1 row
[SQL Worker 1] run_sql: projects listing storage vs. projects with storage MW recorded       -> 1 row
[SQL Worker 1] run_sql: final total and count                                               -> 1 row
[SQL Worker 1] run_sql: missing storage MW by type_group                                    -> 5 rows
[SQL Worker 1] run_sql: final rounded figures                                               -> 1 row
[SQL Worker 1] submit_finding (high confidence)
[Verifier] get_data_gaps; run_sql: recompute the storage total and counts                   -> 1 row
[Verifier] submit_verification: all checks passed
=> "There is a total of 552,750 MW (about 552.7 GW) of battery storage capacity currently active
    in US interconnection queues, spread across 2,718 projects with recorded storage capacity ..."
```
</details>

## Repo layout

```
app.py                  FastAPI: /, /chat, /chat/stream, /clear, /health
build_db.py             offline build: raw files -> DuckDB + context JSON
gridqueue/
  pipeline.py           Planner -> Workers -> Verifier -> Writer, budgets, numbers check
  roles.py prompts.py   agents, retries, prompts
  tools.py guardrails.py  read-only SQL tools, category and fan-out checks, gap notes
  visuals.py            tables, Vega-Lite charts and county maps from verified SQL
  grounding.py          the numbers check
  baseline.py           the plain run_sql agent and the no-tools baseline
index.html static/      the page, vendored chart libraries, fonts and county shapes
scripts/                smoke tests and the README comparisons
tests/                  unit tests with a fake model
```
