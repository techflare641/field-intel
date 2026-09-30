"""Allowlisted, typed tools the agent may call.

Rules that make the agent auditable:

* Tools read only from ``field_observations`` / ``fields``. No raster access, no raw SQL.
* Arguments are validated with pydantic *before* execution; bad args become a tool error
  the model can recover from, never an exception.
* Every result carries ``provenance``: the ``FieldObservation.id`` rows it was built from,
  so an answer can be traced back to specific scenes and pixels.
* ``propose_action`` writes a *pending* ``ActionRequest``. Nothing executes.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from itertools import pairwise
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError
from pydantic import Field as PField
from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from field_intel.agent.llm import ToolSpec
from field_intel.db import ActionRequest, ActionStatus, Field, FieldObservation, record_audit


@dataclass(slots=True)
class ToolContext:
    session: Session
    principal: str
    stress_threshold: float


@dataclass(slots=True)
class ToolResult:
    data: Any
    provenance: list[int] = field(default_factory=list)
    error: str | None = None
    created_action_ids: list[int] = field(default_factory=list)

    def to_json(self) -> str:
        body: dict[str, Any] = {"data": self.data, "provenance": self.provenance}
        if self.error:
            body = {"error": self.error}
        return json.dumps(body, default=str)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ListFieldsArgs(_StrictModel):
    grower: str | None = PField(default=None, description="Filter by grower name (exact).")


class FieldTrendArgs(_StrictModel):
    field_id: str = PField(description="Field id, e.g. 'F-103'.")
    days: int = PField(default=30, ge=1, le=365, description="Look-back window in days.")


class RankStressedArgs(_StrictModel):
    top_n: int = PField(default=5, ge=1, le=50)
    threshold: float | None = PField(
        default=None, ge=-1.0, le=1.0, description="Override the configured NDVI stress threshold."
    )


ActionKind = Literal["scout", "irrigation_check", "agronomist_review"]


class ProposeActionArgs(_StrictModel):
    field_id: str
    action: ActionKind
    reason: str = PField(min_length=10, max_length=1000)
    evidence_observation_ids: list[int] = PField(min_length=1, max_length=50)


Handler = Callable[[ToolContext, Any], ToolResult]


@dataclass(frozen=True, slots=True)
class Tool:
    spec: ToolSpec
    args_model: type[BaseModel]
    handler: Handler


def observation_row(o: FieldObservation, fld: Field | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "observation_id": o.id,
        "field_id": o.field_id,
        "scene_id": o.scene_id,
        "acquired_on": o.acquired_on.isoformat(),
        "ndvi_mean": round(o.ndvi_mean, 3),
        "ndvi_p10": round(o.ndvi_p10, 3),
        "ndvi_p50": round(o.ndvi_p50, 3),
        "ndvi_p90": round(o.ndvi_p90, 3),
        "valid_pixel_frac": round(o.valid_pixel_frac, 3),
        "pixel_count": o.pixel_count,
        "stress_flag": o.stress_flag,
    }
    if fld is not None:
        row.update({"name": fld.name, "crop": fld.crop, "grower": fld.grower})
    return row


def latest_observations(session: Session) -> list[tuple[FieldObservation, Field]]:
    latest = (
        select(
            FieldObservation.field_id.label("field_id"),
            func.max(FieldObservation.acquired_on).label("acquired_on"),
        )
        .group_by(FieldObservation.field_id)
        .subquery()
    )
    stmt = (
        select(FieldObservation, Field)
        .join(Field, Field.id == FieldObservation.field_id)
        .join(
            latest,
            and_(
                FieldObservation.field_id == latest.c.field_id,
                FieldObservation.acquired_on == latest.c.acquired_on,
            ),
        )
        .order_by(FieldObservation.field_id, FieldObservation.id.desc())
    )
    seen: set[str] = set()
    out: list[tuple[FieldObservation, Field]] = []
    for obs, fld in session.execute(stmt).all():
        if obs.field_id in seen:
            continue
        seen.add(obs.field_id)
        out.append((obs, fld))
    return out


# ---------------------------------------------------------------------------- tools


def list_fields(ctx: ToolContext, args: ListFieldsArgs) -> ToolResult:
    latest = {obs.field_id: obs for obs, _ in latest_observations(ctx.session)}
    stmt = select(Field).order_by(Field.id)
    if args.grower:
        stmt = stmt.where(Field.grower == args.grower)
    rows: list[dict[str, Any]] = []
    provenance: list[int] = []
    for fld in ctx.session.scalars(stmt).all():
        obs = latest.get(fld.id)
        row: dict[str, Any] = {
            "field_id": fld.id,
            "name": fld.name,
            "crop": fld.crop,
            "grower": fld.grower,
            "area_ha": round(fld.area_ha, 2),
            "latest": observation_row(obs) if obs else None,
        }
        if obs:
            provenance.append(obs.id)
        rows.append(row)
    return ToolResult(data=rows, provenance=provenance)


def get_field_trend(ctx: ToolContext, args: FieldTrendArgs) -> ToolResult:
    fld = ctx.session.get(Field, args.field_id)
    if fld is None:
        return ToolResult(data=None, error=f"unknown field_id {args.field_id!r}")
    newest = ctx.session.scalar(
        select(func.max(FieldObservation.acquired_on)).where(
            FieldObservation.field_id == args.field_id
        )
    )
    if newest is None:
        return ToolResult(data=[], provenance=[])
    since = newest - timedelta(days=args.days)
    obs = ctx.session.scalars(
        select(FieldObservation)
        .where(FieldObservation.field_id == args.field_id, FieldObservation.acquired_on >= since)
        .order_by(FieldObservation.acquired_on, FieldObservation.id)
    ).all()
    rows = [observation_row(o, fld) for o in obs]
    for prev, cur in pairwise(rows):
        cur["ndvi_mean_delta"] = round(cur["ndvi_mean"] - prev["ndvi_mean"], 3)
    return ToolResult(data=rows, provenance=[o.id for o in obs])


def rank_stressed_fields(ctx: ToolContext, args: RankStressedArgs) -> ToolResult:
    threshold = ctx.stress_threshold if args.threshold is None else args.threshold
    latest = latest_observations(ctx.session)
    ranked = sorted(latest, key=lambda pair: pair[0].ndvi_mean)[: args.top_n]
    rows = []
    for obs, fld in ranked:
        row = observation_row(obs, fld)
        row["stress_flag"] = obs.ndvi_mean < threshold
        row["threshold"] = threshold
        rows.append(row)
    return ToolResult(data=rows, provenance=[obs.id for obs, _ in ranked])


def propose_action(ctx: ToolContext, args: ProposeActionArgs) -> ToolResult:
    fld = ctx.session.get(Field, args.field_id)
    if fld is None:
        return ToolResult(data=None, error=f"unknown field_id {args.field_id!r}")
    known = set(
        ctx.session.scalars(
            select(FieldObservation.id).where(
                FieldObservation.id.in_(args.evidence_observation_ids),
                FieldObservation.field_id == args.field_id,
            )
        ).all()
    )
    bad = sorted(set(args.evidence_observation_ids) - known)
    if bad:
        return ToolResult(
            data=None,
            error=f"evidence_observation_ids {bad} do not belong to field {args.field_id}",
        )
    req = ActionRequest(
        field_id=args.field_id,
        action=args.action,
        reason=args.reason,
        evidence_json=json.dumps(sorted(known)),
        status=ActionStatus.PENDING,
        proposed_by=f"agent:{ctx.principal}",
    )
    ctx.session.add(req)
    ctx.session.flush()
    record_audit(
        ctx.session,
        "action.proposed",
        ctx.principal,
        {"action_request_id": req.id, "field_id": req.field_id, "action": req.action},
    )
    return ToolResult(
        data={
            "action_request_id": req.id,
            "field_id": req.field_id,
            "action": req.action,
            "status": "pending_operator_approval",
        },
        provenance=sorted(known),
        created_action_ids=[req.id],
    )


def _spec(name: str, description: str, model: type[BaseModel]) -> ToolSpec:
    schema = model.model_json_schema()
    schema.pop("title", None)
    return ToolSpec(name=name, description=description, parameters=schema)


REGISTRY: dict[str, Tool] = {
    "list_fields": Tool(
        _spec(
            "list_fields",
            "List all fields with their latest NDVI observation. Optional grower filter.",
            ListFieldsArgs,
        ),
        ListFieldsArgs,
        list_fields,
    ),
    "get_field_trend": Tool(
        _spec(
            "get_field_trend",
            "Time series of NDVI observations for one field over the last N days, oldest first, "
            "with per-step deltas.",
            FieldTrendArgs,
        ),
        FieldTrendArgs,
        get_field_trend,
    ),
    "rank_stressed_fields": Tool(
        _spec(
            "rank_stressed_fields",
            "Fields ordered from lowest to highest latest mean NDVI, with stress flags. "
            "Use this first when asked which fields need attention.",
            RankStressedArgs,
        ),
        RankStressedArgs,
        rank_stressed_fields,
    ),
    "propose_action": Tool(
        _spec(
            "propose_action",
            "Propose a field action (scout, irrigation_check, agronomist_review). Creates a "
            "PENDING request that a human operator must approve; nothing is executed. "
            "evidence_observation_ids must be observation ids you have already retrieved.",
            ProposeActionArgs,
        ),
        ProposeActionArgs,
        propose_action,
    ),
}


def tool_specs() -> list[ToolSpec]:
    return [t.spec for t in REGISTRY.values()]


def execute(ctx: ToolContext, name: str, raw_args: dict[str, Any]) -> ToolResult:
    """Validate and run one tool. Never raises for model mistakes."""
    tool = REGISTRY.get(name)
    if tool is None:
        return ToolResult(data=None, error=f"unknown tool {name!r}; allowed: {sorted(REGISTRY)}")
    try:
        args = tool.args_model.model_validate(raw_args)
    except ValidationError as exc:
        return ToolResult(data=None, error=f"invalid arguments for {name}: {exc.errors()}")
    return tool.handler(ctx, args)
