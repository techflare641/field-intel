"""FastAPI application.

``create_app`` is a factory so tests can inject their own ``Settings``, ``Database`` and
LLM client. Mutating endpoints write ``AuditEvent`` rows; every response carries an
``X-Request-ID`` for log correlation.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from field_intel import __version__
from field_intel.agent.llm import LLMClient, LLMProviderError, build_gateway
from field_intel.agent.loop import run_agent
from field_intel.agent.tools import latest_observations, observation_row
from field_intel.api import schemas
from field_intel.api.auth import Principal, current_principal, require_operator
from field_intel.config import Settings, get_settings
from field_intel.db import (
    ActionRequest,
    ActionStatus,
    Database,
    Field,
    FieldObservation,
    record_audit,
)
from field_intel.pipeline.ingest_fields import FieldIngestError
from field_intel.pipeline.run import run_pipeline

log = logging.getLogger("field_intel.api")


def create_app(
    settings: Settings | None = None,
    *,
    db: Database | None = None,
    llm: LLMClient | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(
        title="Field Intel",
        version=__version__,
        description="Satellite imagery -> per-field stress signals -> governed AI assistant.",
    )
    app.state.settings = settings
    app.state.db = db or Database(settings.database_url)
    app.state.llm = llm or build_gateway(
        settings.provider_order,
        anthropic_api_key=settings.anthropic_api_key,
        anthropic_model=settings.anthropic_model,
        anthropic_base_url=settings.anthropic_base_url,
        openai_api_key=settings.openai_api_key,
        openai_model=settings.openai_model,
        openai_base_url=settings.openai_base_url,
        timeout=settings.llm_timeout_seconds,
    )

    @app.middleware("http")
    async def request_id_and_timing(
        request: Request, call_next: Callable[[Request], Any]
    ) -> Response:
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
        start = time.perf_counter()
        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        log.info(
            "request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round((time.perf_counter() - start) * 1000, 1),
            },
        )
        return response

    _register_routes(app)
    return app


# ------------------------------------------------------------------------ dependencies


def _db(request: Request) -> Database:
    return request.app.state.db  # type: ignore[no-any-return]


def _settings(request: Request) -> Settings:
    return request.app.state.settings  # type: ignore[no-any-return]


def _llm(request: Request) -> LLMClient:
    return request.app.state.llm  # type: ignore[no-any-return]


def get_session(request: Request) -> Iterator[Session]:
    with _db(request).session() as session:
        yield session


def _action_out(a: ActionRequest) -> schemas.ActionOut:
    return schemas.ActionOut(
        id=a.id,
        field_id=a.field_id,
        action=a.action,
        reason=a.reason,
        evidence=a.evidence,
        status=a.status,
        proposed_by=a.proposed_by,
        decided_by=a.decided_by,
        decision_note=a.decision_note,
        created_at=a.created_at,
        decided_at=a.decided_at,
    )


def _safe_path(raw: str, root: Path) -> Path:
    """Resolve ``raw`` and ensure it lives under ``root`` (no traversal outside data dir)."""
    root_resolved = root.resolve()
    candidate = (
        (root_resolved / raw).resolve() if not Path(raw).is_absolute() else Path(raw).resolve()
    )
    if root_resolved not in candidate.parents and candidate != root_resolved:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"path must be under {root}")
    if not candidate.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"{raw} not found")
    return candidate


# ------------------------------------------------------------------------------ routes


def _register_routes(app: FastAPI) -> None:
    @app.get("/health", response_model=schemas.HealthOut, tags=["ops"])
    def health(settings: Settings = Depends(_settings)) -> schemas.HealthOut:
        return schemas.HealthOut(
            status="ok", version=__version__, llm_providers=settings.provider_order
        )

    @app.get("/fields", response_model=list[schemas.FieldOut], tags=["fields"])
    def list_fields(
        _: Principal = Depends(current_principal),
        session: Session = Depends(get_session),
    ) -> list[schemas.FieldOut]:
        latest = {obs.field_id: obs for obs, _f in latest_observations(session)}
        out = []
        for fld in session.scalars(select(Field).order_by(Field.id)).all():
            obs = latest.get(fld.id)
            out.append(
                schemas.FieldOut(
                    field_id=fld.id,
                    name=fld.name,
                    crop=fld.crop,
                    grower=fld.grower,
                    area_ha=round(fld.area_ha, 2),
                    latest=schemas.ObservationOut(**observation_row(obs)) if obs else None,
                )
            )
        return out

    @app.get(
        "/fields/{field_id}/observations",
        response_model=list[schemas.ObservationOut],
        tags=["fields"],
    )
    def field_observations(
        field_id: str,
        _: Principal = Depends(current_principal),
        session: Session = Depends(get_session),
    ) -> list[schemas.ObservationOut]:
        if session.get(Field, field_id) is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"field {field_id} not found")
        obs = session.scalars(
            select(FieldObservation)
            .where(FieldObservation.field_id == field_id)
            .order_by(FieldObservation.acquired_on, FieldObservation.id)
        ).all()
        return [schemas.ObservationOut(**observation_row(o)) for o in obs]

    @app.post("/pipeline/run", response_model=schemas.PipelineRunOut, tags=["pipeline"])
    def pipeline_run(
        body: schemas.PipelineRunIn,
        principal: Principal = Depends(require_operator),
        settings: Settings = Depends(_settings),
        db: Database = Depends(_db),
    ) -> schemas.PipelineRunOut:
        root = Path(settings.data_root)
        fields_path = _safe_path(body.fields_path, root)
        raster_path = _safe_path(body.raster_path, root)
        try:
            result = run_pipeline(
                db,
                fields_path=fields_path,
                raster_path=raster_path,
                acquired_on=body.acquired_on,
                principal=principal.audit_name,
                stress_threshold=settings.stress_ndvi_threshold,
            )
        except FieldIngestError as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
        return schemas.PipelineRunOut(**result.as_dict())  # type: ignore[arg-type]

    @app.post("/ask", response_model=schemas.AskOut, tags=["agent"])
    def ask(
        body: schemas.AskIn,
        principal: Principal = Depends(current_principal),
        settings: Settings = Depends(_settings),
        llm: LLMClient = Depends(_llm),
        session: Session = Depends(get_session),
    ) -> schemas.AskOut:
        try:
            answer = run_agent(
                body.question,
                session=session,
                llm=llm,
                principal=principal.audit_name,
                stress_threshold=settings.stress_ndvi_threshold,
                max_steps=settings.agent_max_steps,
                tool_result_max_chars=settings.tool_result_max_chars,
            )
        except LLMProviderError as exc:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
        return schemas.AskOut(**answer.as_dict())

    @app.get("/actions", response_model=list[schemas.ActionOut], tags=["actions"])
    def list_actions(
        status_filter: ActionStatus | None = None,
        _: Principal = Depends(current_principal),
        session: Session = Depends(get_session),
    ) -> list[schemas.ActionOut]:
        stmt = select(ActionRequest).order_by(ActionRequest.id.desc())
        if status_filter is not None:
            stmt = stmt.where(ActionRequest.status == status_filter)
        return [_action_out(a) for a in session.scalars(stmt).all()]

    def _decide(
        action_id: int,
        decision: ActionStatus,
        body: schemas.DecisionIn,
        principal: Principal,
        session: Session,
    ) -> schemas.ActionOut:
        req = session.get(ActionRequest, action_id)
        if req is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"action {action_id} not found")
        if req.status != ActionStatus.PENDING:
            raise HTTPException(
                status.HTTP_409_CONFLICT, f"action {action_id} already {req.status}"
            )
        req.status = decision
        req.decided_by = principal.audit_name
        req.decision_note = body.note
        req.decided_at = datetime.now(tz=UTC)
        record_audit(
            session,
            f"action.{decision}",
            principal.audit_name,
            {"action_request_id": req.id, "field_id": req.field_id, "note": body.note},
        )
        session.flush()
        return _action_out(req)

    @app.post("/actions/{action_id}/approve", response_model=schemas.ActionOut, tags=["actions"])
    def approve_action(
        action_id: int,
        body: schemas.DecisionIn | None = None,
        principal: Principal = Depends(require_operator),
        session: Session = Depends(get_session),
    ) -> schemas.ActionOut:
        return _decide(
            action_id, ActionStatus.APPROVED, body or schemas.DecisionIn(), principal, session
        )

    @app.post("/actions/{action_id}/reject", response_model=schemas.ActionOut, tags=["actions"])
    def reject_action(
        action_id: int,
        body: schemas.DecisionIn | None = None,
        principal: Principal = Depends(require_operator),
        session: Session = Depends(get_session),
    ) -> schemas.ActionOut:
        return _decide(
            action_id, ActionStatus.REJECTED, body or schemas.DecisionIn(), principal, session
        )
