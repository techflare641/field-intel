# field-intel

Satellite imagery -> per-field stress signals -> a governed AI assistant for fresh-produce growers.

A small, production-shaped Python service that shows how I like to build AI features on top of
geospatial data: a deterministic, idempotent data pipeline; a FastAPI surface with role-based
access; and a tool-calling LLM agent whose blast radius is bounded by design (allowlisted typed
tools, row-level provenance on every answer, and a human-approval gate on anything that looks like
an action).

```
uv sync && make demo        # synthetic Sentinel-2-like scenes -> NDVI pipeline -> ask the agent
```

No API keys or satellite credentials are needed for the demo; a deterministic `fake` LLM
provider drives the real tool loop. Set `FIELD_INTEL_LLM_PROVIDERS=anthropic,openai,fake` with
keys to use real models with fallback.

## What it does

1. **Ingests field boundaries** (GeoJSON, any CRS) with GeoPandas/Shapely: validates geometry,
   computes hectares in an equal-area CRS, upserts.
2. **Reduces multispectral scenes to per-field NDVI statistics** with Rasterio: mean / p10 / p50 /
   p90, valid-pixel fraction (clouds and nodata count as coverage loss), and a stress flag.
3. **Loads observations idempotently** into Postgres or SQLite. Scene id is a content hash of the
   raster bytes plus acquisition date; `(field_id, scene_id)` is unique; reruns are audited no-ops.
4. **Exposes an API** for fields, observations, pipeline runs, agent Q&A, and an action queue.
5. **Answers grower questions with an LLM agent** that can only see the observations table
   through four typed tools, cites observation ids, and *proposes* actions that an operator
   approves or rejects.

## Architecture

```mermaid
flowchart LR
  GeoJSON[fields.geojson] --> Ingest[pipeline.ingest_fields]
  GeoTIFF[Sentinel-2-like GeoTIFF] --> Zonal[pipeline.zonal]
  Ingest --> DB[(fields / scenes / field_observations / action_requests / audit_events)]
  Zonal --> Run[pipeline.run idempotent on field_id + scene_id]
  Run --> DB
  DB --> Tools[agent.tools allowlisted, pydantic-validated, provenance-carrying]
  Tools --> Loop[agent.loop bounded ReAct]
  LLM[agent.llm Anthropic / OpenAI / Fake behind a fallback gateway] --> Loop
  Loop --> API[FastAPI /ask]
  API --> Queue[/actions pending]
  Queue --> Operator[operator approve / reject]
```

## Quickstart

```bash
uv sync                      # Python 3.12 pinned in .python-version
make demo                    # synth -> run (x3, third is a no-op) -> ask
make serve                   # http://127.0.0.1:8000/docs
```

Talk to the API (keys from `.env.example`; growers can ask, only operators can approve):

```bash
curl -s -H 'X-API-Key: dev-grower-key' localhost:8000/fields | jq '.[0]'

curl -s -H 'X-API-Key: dev-grower-key' -H 'content-type: application/json' \
  -d '{"question":"Which fields look stressed this week, and should we scout any?"}' \
  localhost:8000/ask | jq '{text, citations, pending_actions}'

curl -s -H 'X-API-Key: dev-grower-key' 'localhost:8000/actions?status_filter=pending' | jq
curl -s -X POST -H 'X-API-Key: dev-operator-key' localhost:8000/actions/1/approve
```

Sample agent output from the demo dataset:

```
Fields currently flagged as stressed (latest scene):
- F-104 Hill Piece (iceberg lettuce): mean NDVI 0.20, p10 0.15 [obs 9]
- F-103 Well Road (broccoli): mean NDVI 0.28, p10 0.23 [obs 8]
- F-105 Slough Corner is healthy on average but has a low-NDVI patch (p10 0.19) worth a look [obs 10]
Proposed action #1: scout on F-104 - pending operator approval.
```

## Governance: how the agent is kept honest

| Control | Where |
| --- | --- |
| LLM never touches rasters or SQL; only four typed tools over pre-computed rows | `agent/tools.py` |
| Every tool result returns the `observation_id`s it used; the answer aggregates them as citations | `agent/tools.py`, `agent/loop.py` |
| Tool arguments validated with pydantic (`extra="forbid"`); bad calls return a tool error, never raise | `agent/tools.py::execute` |
| `propose_action` writes a *pending* `ActionRequest`; evidence ids must belong to that field | `agent/tools.py::propose_action` |
| Only `operator` role can approve/reject; decisions are final (409 on re-decide) | `api/auth.py`, `api/main.py` |
| Inbound PII redaction (email / phone / SSN) before any provider call | `agent/guardrails.py` |
| Tool output sanitised for prompt-injection phrasing and size-capped | `agent/guardrails.py` |
| Hard step limit; loop ends with an explicit incomplete answer | `agent/loop.py` |
| Pipeline runs, agent turns, proposals, and decisions all land in `audit_events` | `db.py::record_audit` |
| `POST /pipeline/run` only reads paths under `FIELD_INTEL_DATA_ROOT` | `api/main.py::_safe_path` |
| API keys compared with `hmac.compare_digest`; logged as non-reversible labels | `api/auth.py` |

## API

| Method | Path | Role | Purpose |
| --- | --- | --- | --- |
| GET | `/health` | public | liveness + configured providers |
| GET | `/fields` | grower | fields with latest observation |
| GET | `/fields/{id}/observations` | grower | NDVI time series for one field |
| POST | `/pipeline/run` | operator | ingest one scene (idempotent) |
| POST | `/ask` | grower | agent Q&A with citations and any proposed actions |
| GET | `/actions` | grower | action queue, optional `status_filter` |
| POST | `/actions/{id}/approve` `/reject` | operator | human-in-the-loop decision |

## Decisions worth discussing

**The agent reads a table, not the world.** The LLM has no access to rasters, the filesystem,
or SQL. The pipeline reduces imagery to `field_observations`, and the agent reaches it through
four allowlisted tools whose results carry the row ids they were built from. That gives every
answer row-level provenance (which scene, which pixels) and makes the model's blast radius the
size of a `SELECT`. It also means a real Claude/GPT call and the offline `FakeClient` exercise
exactly the same tool path, so the tests cover the governance, not the model.

**Idempotency by content hash, not by filename or date.** `scene_id = sha256(raster bytes ||
acquisition date)[:16]`. Re-running the same file is a no-op; a re-delivered, re-processed scene
for the same day is a new scene rather than a silent overwrite. Reruns are recorded in
`audit_events` with `skipped_existing` counts so "did the nightly job actually do anything" is a
query, not a guess.

**Actions are proposals.** `propose_action` is the only tool with side effects, and its side
effect is a `pending` row. An operator approves via the API; that decision is attributed and
final. This is the same pattern I have used for money movement and underwriting: let the model
draft, let a human sign.

**Coverage is measured against the polygon, not against the pixels rasterio happened to keep.**
An early version computed `valid_pixel_frac` from rasterio's combined mask, which silently folds
nodata into "outside the polygon" and reports a fully clouded field as 100 percent valid. The
fix (`geometry_mask` for the footprint, dataset mask for validity) is small, but it is the
difference between "no data for F-106 this week" and a confident wrong number.

## What I'd do differently

- Push zonal statistics into PostGIS (`ST_SummaryStats` over `raster` tiles) or a dbt model
  earlier instead of computing in Python; that is where the analytics would eventually live and
  it makes backfills a SQL job.
- Use a real cloud mask (Sentinel-2 SCL band) rather than treating nodata as the only invalid
  case, and store per-field cloud fraction separately from valid fraction.
- Replace the API-key scheme with Supabase JWTs plus row-level security so growers only ever see
  their own fields at the database layer, not just the route layer.
- Stream the agent turn (SSE) so an operator console can render tool calls as they happen.

## Next steps

- Next.js / Tailwind operator console: action queue, field map (Mapbox GL), per-field NDVI trend.
- `SentinelHubSource`: real Process API fetch with SCL cloud masking (interface already in
  `pipeline/sources.py`).
- Prompt versioning and an offline eval set for the agent; LangSmith or OpenTelemetry traces per
  tool call.

## Layout

```
src/field_intel/
  config.py           pydantic-settings, env-driven, safe defaults
  db.py               SQLAlchemy 2.0 models + Database wrapper
  pipeline/
    synthetic.py      deterministic demo dataset (6 fields, 2 scenes, cloud strip)
    sources.py        SceneSource protocol; local GeoTIFF + Sentinel Hub seam
    ingest_fields.py  GeoJSON -> validated GeoDataFrame -> upsert
    zonal.py          rasterio.mask + NDVI + stats, nodata/coverage aware
    run.py            idempotent orchestrator + audit
  agent/
    llm.py            Message/ToolCall model, Anthropic + OpenAI adapters, FakeClient, Gateway
    tools.py          allowlisted typed tools with provenance
    guardrails.py     PII redaction, injection sanitising, truncation
    loop.py           bounded ReAct loop -> Answer
  api/                FastAPI app factory, API-key RBAC, schemas
  cli.py              synth | run | ask | actions | serve
tests/                35 tests: NDVI math, idempotency, agent loop, guardrails, wire formats, API
```

## Tooling

uv, ruff, mypy (strict), pytest, Docker (multi-stage, non-root), GitHub Actions (lint, typecheck,
tests, offline demo, container smoke test).

## License

MIT
