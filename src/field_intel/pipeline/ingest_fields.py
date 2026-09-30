"""Field boundary ingest: GeoJSON -> validated GeoDataFrame -> upserted ``fields`` rows."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
from shapely import make_valid
from shapely.geometry import GeometryCollection
from shapely.geometry.base import BaseGeometry
from sqlalchemy.orm import Session

from field_intel.db import Field

REQUIRED_COLUMNS = ("id", "name")
OPTIONAL_DEFAULTS = {"crop": "unknown", "grower": "unknown"}
# Equal-area CRS used only to compute hectares consistently regardless of input CRS.
AREA_CRS = "EPSG:6933"


class FieldIngestError(ValueError):
    pass


def load_fields(path: Path) -> gpd.GeoDataFrame:
    """Read field polygons and normalize schema + geometry validity."""
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise FieldIngestError(f"no features in {path}")
    if gdf.crs is None:
        raise FieldIngestError(f"{path} has no CRS; GeoJSON should be EPSG:4326")

    missing = [c for c in REQUIRED_COLUMNS if c not in gdf.columns]
    if missing:
        raise FieldIngestError(f"missing required properties {missing} in {path}")
    for col, default in OPTIONAL_DEFAULTS.items():
        if col not in gdf.columns:
            gdf[col] = default
        gdf[col] = gdf[col].fillna(default).astype(str)

    gdf["id"] = gdf["id"].astype(str)
    if gdf["id"].duplicated().any():
        dupes = sorted(gdf.loc[gdf["id"].duplicated(), "id"].unique().tolist())
        raise FieldIngestError(f"duplicate field ids: {dupes}")

    gdf["geometry"] = gdf.geometry.apply(_clean_geometry)
    gdf = gdf[~gdf.geometry.is_empty].copy()
    if gdf.empty:
        raise FieldIngestError("all geometries were empty after validation")

    gdf["area_ha"] = gdf.to_crs(AREA_CRS).geometry.area / 10_000.0
    return gdf


def _clean_geometry(geom: BaseGeometry) -> BaseGeometry:
    if geom is None or geom.is_empty:
        raise FieldIngestError("empty geometry")
    fixed = geom if geom.is_valid else make_valid(geom)
    # make_valid can return GeometryCollections; keep only the polygonal part.
    if isinstance(fixed, GeometryCollection):
        polys = [g for g in fixed.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
        if not polys:
            raise FieldIngestError("geometry has no polygonal component")
        fixed = polys[0] if len(polys) == 1 else gpd.GeoSeries(polys).union_all()
    if fixed.geom_type not in ("Polygon", "MultiPolygon"):
        raise FieldIngestError(f"unsupported geometry type {fixed.geom_type}")
    return fixed


def upsert_fields(session: Session, gdf: gpd.GeoDataFrame) -> tuple[int, int]:
    """Insert new fields / update existing ones. Returns (inserted, updated)."""
    inserted = updated = 0
    crs = gdf.crs.to_string()
    for row in gdf.itertuples(index=False):
        existing = session.get(Field, row.id)
        values = {
            "name": row.name,
            "crop": row.crop,
            "grower": row.grower,
            "area_ha": float(row.area_ha),
            "crs": crs,
            "geometry_wkt": row.geometry.wkt,
        }
        if existing is None:
            session.add(Field(id=row.id, **values))
            inserted += 1
        else:
            changed = any(getattr(existing, k) != v for k, v in values.items())
            if changed:
                for k, v in values.items():
                    setattr(existing, k, v)
                updated += 1
    session.flush()
    return inserted, updated
