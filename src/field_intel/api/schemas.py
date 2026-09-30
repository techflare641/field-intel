"""Pydantic response/request models for the HTTP API."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field


class HealthOut(BaseModel):
    status: str
    version: str
    llm_providers: list[str]


class ObservationOut(BaseModel):
    observation_id: int
    field_id: str
    scene_id: str
    acquired_on: date
    ndvi_mean: float
    ndvi_p10: float
    ndvi_p50: float
    ndvi_p90: float
    valid_pixel_frac: float
    pixel_count: int
    stress_flag: bool


class FieldOut(BaseModel):
    field_id: str
    name: str
    crop: str
    grower: str
    area_ha: float
    latest: ObservationOut | None


class PipelineRunIn(BaseModel):
    fields_path: str = Field(description="Path to a GeoJSON of field polygons (server-local).")
    raster_path: str = Field(description="Path to a multispectral GeoTIFF (server-local).")
    acquired_on: date


class PipelineRunOut(BaseModel):
    scene_id: str
    acquired_on: date
    fields_inserted: int
    fields_updated: int
    written: int
    skipped_existing: int
    skipped_low_coverage: list[str]
    skipped_no_overlap: list[str]


class AskIn(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


class AskOut(BaseModel):
    text: str
    citations: list[int]
    pending_actions: list[int]
    provider: str
    model: str
    steps: int
    complete: bool
    redactions: int
    trace: list[dict[str, Any]]


class ActionOut(BaseModel):
    id: int
    field_id: str
    action: str
    reason: str
    evidence: list[int]
    status: str
    proposed_by: str
    decided_by: str | None
    decision_note: str | None
    created_at: datetime
    decided_at: datetime | None


class DecisionIn(BaseModel):
    note: str | None = Field(default=None, max_length=1000)
