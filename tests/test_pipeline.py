from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from sqlalchemy import func, select

from field_intel.db import AuditEvent, Database, Field, FieldObservation, Scene
from field_intel.pipeline.ingest_fields import FieldIngestError, load_fields
from field_intel.pipeline.run import RunResult, run_pipeline, scene_content_id
from field_intel.pipeline.synthetic import DEMO_DATES


def test_first_run_writes_and_skips_cloudy_field(
    loaded_db: tuple[Database, list[RunResult]],
) -> None:
    _, (first, second) = loaded_db
    assert first.fields_inserted == 6
    assert first.written == 5
    assert first.skipped_low_coverage == ["F-106"]  # under the simulated cloud mask
    assert second.written == 6
    assert second.skipped_low_coverage == []
    assert first.scene_id != second.scene_id


def test_rerun_is_idempotent_and_audited(
    loaded_db: tuple[Database, list[RunResult]], demo_paths: dict[str, Path]
) -> None:
    db, (_, second) = loaded_db
    with db.session() as s:
        before = s.scalar(select(func.count()).select_from(FieldObservation))

    d = DEMO_DATES[1]
    again = run_pipeline(
        db,
        fields_path=demo_paths["fields"],
        raster_path=demo_paths[d.isoformat()],
        acquired_on=d,
        principal="test-rerun",
    )
    assert again.scene_id == second.scene_id
    assert again.written == 0
    assert again.skipped_existing == 6

    with db.session() as s:
        after = s.scalar(select(func.count()).select_from(FieldObservation))
        assert after == before
        assert s.scalar(select(func.count()).select_from(Scene)) == 2
        events = s.scalars(
            select(AuditEvent).where(AuditEvent.kind == "pipeline.run").order_by(AuditEvent.id)
        ).all()
    assert len(events) == 3
    assert events[-1].principal == "test-rerun"
    assert events[-1].payload["skipped_existing"] == 6


def test_scene_id_depends_on_bytes_and_date(demo_paths: dict[str, Path]) -> None:
    p = demo_paths[DEMO_DATES[0].isoformat()]
    a = scene_content_id(p, date(2026, 9, 14))
    b = scene_content_id(p, date(2026, 9, 15))
    c = scene_content_id(demo_paths[DEMO_DATES[1].isoformat()], date(2026, 9, 14))
    assert a != b
    assert a != c
    assert a == scene_content_id(p, date(2026, 9, 14))


def test_stress_flag_and_decline(loaded_db: tuple[Database, list[RunResult]]) -> None:
    db, _ = loaded_db
    with db.session() as s:
        f103 = s.scalars(
            select(FieldObservation)
            .where(FieldObservation.field_id == "F-103")
            .order_by(FieldObservation.acquired_on)
        ).all()
        f104 = s.scalars(select(FieldObservation).where(FieldObservation.field_id == "F-104")).all()
        f105 = s.scalars(select(FieldObservation).where(FieldObservation.field_id == "F-105")).all()
        fields = {f.id: f for f in s.scalars(select(Field)).all()}

    assert [o.stress_flag for o in f103] == [False, True]  # declined below threshold
    assert f103[1].ndvi_mean < f103[0].ndvi_mean - 0.25
    assert all(o.stress_flag for o in f104)
    # patchy field: healthy mean, low p10
    assert all(not o.stress_flag and o.ndvi_p10 < 0.3 for o in f105)
    assert fields["F-101"].area_ha == pytest.approx(35.0, abs=0.5)
    assert fields["F-101"].crs == "EPSG:4326"


def test_load_fields_rejects_bad_input(tmp_path: Path, demo_paths: dict[str, Path]) -> None:
    gdf = load_fields(demo_paths["fields"])
    assert set(gdf["id"]) == {f"F-10{i}" for i in range(1, 7)}

    dup = gdf.copy()
    dup.loc[dup.index[1], "id"] = "F-101"
    dup_path = tmp_path / "dup.geojson"
    dup.to_file(dup_path, driver="GeoJSON")
    with pytest.raises(FieldIngestError, match="duplicate"):
        load_fields(dup_path)

    missing = gdf.drop(columns=["name"])
    missing_path = tmp_path / "missing.geojson"
    missing.to_file(missing_path, driver="GeoJSON")
    with pytest.raises(FieldIngestError, match="missing required"):
        load_fields(missing_path)
