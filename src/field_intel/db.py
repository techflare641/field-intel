"""Persistence layer (SQLAlchemy 2.0, typed).

Design notes
------------
* ``FieldObservation`` is the *only* table the agent's tools read from. Raw rasters
  never reach the LLM; the pipeline reduces them to auditable per-field rows.
* ``(field_id, scene_id)`` is unique, which is what makes ``pipeline.run`` idempotent.
  ``scene_id`` is a content hash of the raster + acquisition date, so re-ingesting
  the same file is a no-op and a *different* file for the same date is a new scene.
* ``ActionRequest`` is the human-in-the-loop gate: the agent may *propose*, only an
  operator may *approve*.
* ``AuditEvent`` records pipeline runs, agent turns, and approvals with the principal
  that triggered them.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)


def utcnow() -> datetime:
    return datetime.now(tz=UTC)


class Base(DeclarativeBase):
    pass


class Field(Base):
    """A grower's field/block polygon. Geometry stored as WKT in the field's own CRS."""

    __tablename__ = "fields"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    crop: Mapped[str] = mapped_column(String(100), nullable=False, default="unknown")
    grower: Mapped[str] = mapped_column(String(200), nullable=False, default="unknown")
    area_ha: Mapped[float] = mapped_column(Float, nullable=False)
    crs: Mapped[str] = mapped_column(String(64), nullable=False)
    geometry_wkt: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    observations: Mapped[list[FieldObservation]] = relationship(back_populates="field")


class Scene(Base):
    """One satellite acquisition (a multi-band raster) keyed by content hash."""

    __tablename__ = "scenes"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # sha256 prefix
    source: Mapped[str] = mapped_column(String(64), nullable=False)  # local | sentinel_hub
    uri: Mapped[str] = mapped_column(Text, nullable=False)
    acquired_on: Mapped[date] = mapped_column(Date, nullable=False)
    crs: Mapped[str] = mapped_column(String(64), nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    observations: Mapped[list[FieldObservation]] = relationship(back_populates="scene")


class FieldObservation(Base):
    """Per-field zonal NDVI statistics for one scene. The agent's only data surface."""

    __tablename__ = "field_observations"
    __table_args__ = (UniqueConstraint("field_id", "scene_id", name="uq_field_scene"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    field_id: Mapped[str] = mapped_column(ForeignKey("fields.id"), nullable=False, index=True)
    scene_id: Mapped[str] = mapped_column(ForeignKey("scenes.id"), nullable=False, index=True)
    acquired_on: Mapped[date] = mapped_column(Date, nullable=False, index=True)

    ndvi_mean: Mapped[float] = mapped_column(Float, nullable=False)
    ndvi_p10: Mapped[float] = mapped_column(Float, nullable=False)
    ndvi_p50: Mapped[float] = mapped_column(Float, nullable=False)
    ndvi_p90: Mapped[float] = mapped_column(Float, nullable=False)
    valid_pixel_frac: Mapped[float] = mapped_column(Float, nullable=False)
    pixel_count: Mapped[int] = mapped_column(Integer, nullable=False)
    stress_flag: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    field: Mapped[Field] = relationship(back_populates="observations")
    scene: Mapped[Scene] = relationship(back_populates="observations")


class ActionStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class ActionRequest(Base):
    """An action the agent *proposed*. Nothing executes until an operator approves."""

    __tablename__ = "action_requests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    field_id: Mapped[str] = mapped_column(ForeignKey("fields.id"), nullable=False, index=True)
    action: Mapped[str] = mapped_column(String(64), nullable=False)  # scout | irrigate | ...
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    status: Mapped[str] = mapped_column(String(16), nullable=False, default=ActionStatus.PENDING)
    proposed_by: Mapped[str] = mapped_column(String(200), nullable=False)  # agent:<principal>
    decided_by: Mapped[str | None] = mapped_column(String(200), nullable=True)
    decision_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    @property
    def evidence(self) -> list[int]:
        loaded: Any = json.loads(self.evidence_json or "[]")
        return [int(x) for x in loaded]


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    kind: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    principal: Mapped[str] = mapped_column(String(200), nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    @property
    def payload(self) -> dict[str, Any]:
        loaded: Any = json.loads(self.payload_json or "{}")
        return dict(loaded)


def record_audit(session: Session, kind: str, principal: str, payload: dict[str, Any]) -> None:
    session.add(
        AuditEvent(kind=kind, principal=principal, payload_json=json.dumps(payload, default=str))
    )


# --------------------------------------------------------------------------- engine


def make_engine(database_url: str) -> Engine:
    if database_url.startswith("sqlite:///") and not database_url.endswith(":memory:"):
        db_path = Path(database_url.removeprefix("sqlite:///"))
        db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(database_url, future=True)
    if engine.dialect.name == "sqlite":

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn: Any, _record: Any) -> None:
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


def init_db(engine: Engine) -> None:
    Base.metadata.create_all(engine)


class Database:
    """Small wrapper bundling engine + session factory so callers don't touch globals."""

    def __init__(self, database_url: str) -> None:
        self.engine = make_engine(database_url)
        self._factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        init_db(self.engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        session = self._factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
