"""Pipeline orchestrator.

``run_pipeline`` is idempotent: the scene id is a content hash of the raster bytes plus
the acquisition date, and ``(field_id, scene_id)`` is unique. Re-running on the same
inputs writes nothing and reports every field as ``skipped_existing``. Each run leaves
an ``AuditEvent`` with the counts so operators can see what happened and why.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import rasterio
from pyproj import Transformer
from shapely import wkt as shapely_wkt
from shapely.ops import transform as shapely_transform
from sqlalchemy import select

from field_intel.db import Database, Field, FieldObservation, Scene, record_audit
from field_intel.pipeline.ingest_fields import load_fields, upsert_fields
from field_intel.pipeline.zonal import compute_zonal_ndvi, intersects_raster


@dataclass(slots=True)
class RunResult:
    scene_id: str
    acquired_on: date
    fields_inserted: int = 0
    fields_updated: int = 0
    written: int = 0
    skipped_existing: int = 0
    skipped_low_coverage: list[str] = field(default_factory=list)
    skipped_no_overlap: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "scene_id": self.scene_id,
            "acquired_on": self.acquired_on.isoformat(),
            "fields_inserted": self.fields_inserted,
            "fields_updated": self.fields_updated,
            "written": self.written,
            "skipped_existing": self.skipped_existing,
            "skipped_low_coverage": list(self.skipped_low_coverage),
            "skipped_no_overlap": list(self.skipped_no_overlap),
        }


def scene_content_id(raster_path: Path, acquired_on: date) -> str:
    """Stable id for a scene: sha256(raster bytes || acquisition date)[:16]."""
    h = hashlib.sha256()
    with raster_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    h.update(acquired_on.isoformat().encode())
    return h.hexdigest()[:16]


def run_pipeline(
    db: Database,
    *,
    fields_path: Path,
    raster_path: Path,
    acquired_on: date,
    principal: str = "system",
    source: str = "local",
    stress_threshold: float = 0.35,
    min_valid_frac: float = 0.2,
) -> RunResult:
    gdf = load_fields(fields_path)
    scene_id = scene_content_id(raster_path, acquired_on)
    result = RunResult(scene_id=scene_id, acquired_on=acquired_on)

    with db.session() as session, rasterio.open(raster_path) as dataset:
        raster_crs = dataset.crs.to_string()

        scene = session.get(Scene, scene_id)
        if scene is None:
            session.add(
                Scene(
                    id=scene_id,
                    source=source,
                    uri=str(raster_path),
                    acquired_on=acquired_on,
                    crs=raster_crs,
                )
            )
            session.flush()

        result.fields_inserted, result.fields_updated = upsert_fields(session, gdf)

        existing_ids: set[str] = set(
            session.scalars(
                select(FieldObservation.field_id).where(FieldObservation.scene_id == scene_id)
            ).all()
        )

        fields = session.scalars(select(Field).where(Field.id.in_(gdf["id"].tolist()))).all()
        for fld in fields:
            if fld.id in existing_ids:
                result.skipped_existing += 1
                continue

            geom = shapely_wkt.loads(fld.geometry_wkt)
            if fld.crs != raster_crs:
                transformer = Transformer.from_crs(fld.crs, raster_crs, always_xy=True)
                geom = shapely_transform(transformer.transform, geom)

            stats = compute_zonal_ndvi(dataset, geom, min_valid_frac=min_valid_frac)
            if stats is None:
                bucket = (
                    result.skipped_low_coverage
                    if intersects_raster(dataset, geom)
                    else result.skipped_no_overlap
                )
                bucket.append(fld.id)
                continue

            session.add(
                FieldObservation(
                    field_id=fld.id,
                    scene_id=scene_id,
                    acquired_on=acquired_on,
                    ndvi_mean=stats.ndvi_mean,
                    ndvi_p10=stats.ndvi_p10,
                    ndvi_p50=stats.ndvi_p50,
                    ndvi_p90=stats.ndvi_p90,
                    valid_pixel_frac=stats.valid_pixel_frac,
                    pixel_count=stats.pixel_count,
                    stress_flag=stats.ndvi_mean < stress_threshold,
                )
            )
            result.written += 1

        record_audit(session, "pipeline.run", principal, result.as_dict())

    return result
