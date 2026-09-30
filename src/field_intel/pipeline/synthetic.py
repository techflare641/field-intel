"""Deterministic synthetic data so the whole stack runs offline with no Sentinel Hub creds.

Produces a Sentinel-2-like 4-band (B02, B03, B04, B08) 10 m GeoTIFF over a 4 km x 4 km
tile in the Salinas Valley (EPSG:32610) plus a GeoJSON of six field blocks in EPSG:4326.
Each field gets a target NDVI per date so the demo has healthy, declining, stressed,
patchy, and cloud-obscured cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import Polygon, box

RASTER_CRS = "EPSG:32610"  # UTM 10N
PIXEL_M = 10.0
WIDTH = HEIGHT = 400
ORIGIN_X, ORIGIN_Y = 619_000.0, 4_062_000.0  # upper-left corner
NODATA = 0
BAND_DESCRIPTIONS = ("B02", "B03", "B04", "B08")
SOIL_NDVI = 0.15
REFLECTANCE_SUM = 5000.0  # red + nir, Sentinel-2 L2A 0..10000 scale

DEMO_DATES = (date(2026, 9, 14), date(2026, 9, 24))


@dataclass(frozen=True, slots=True)
class SyntheticField:
    id: str
    name: str
    crop: str
    grower: str
    geometry: Polygon
    ndvi_by_date: dict[date, float]
    stressed_patch: Polygon | None = None


def _rect(x0: float, y0: float, w: float, h: float) -> Polygon:
    return box(x0, y0, x0 + w, y0 + h)


def demo_fields() -> list[SyntheticField]:
    d1, d2 = DEMO_DATES
    return [
        SyntheticField(
            "F-101",
            "North Block",
            "romaine",
            "Rio Vista Farms",
            _rect(619_600, 4_060_800, 700, 500),
            {d1: 0.72, d2: 0.74},
        ),
        SyntheticField(
            "F-102",
            "River Strip",
            "strawberry",
            "Rio Vista Farms",
            _rect(620_600, 4_060_700, 900, 350),
            {d1: 0.55, d2: 0.56},
        ),
        SyntheticField(
            "F-103",
            "Well Road",
            "broccoli",
            "Salinas Greens LLC",
            _rect(619_500, 4_059_600, 800, 600),
            {d1: 0.62, d2: 0.28},  # declining
        ),
        SyntheticField(
            "F-104",
            "Hill Piece",
            "iceberg lettuce",
            "Salinas Greens LLC",
            _rect(620_700, 4_059_400, 600, 700),
            {d1: 0.22, d2: 0.20},  # stressed
        ),
        SyntheticField(
            "F-105",
            "Slough Corner",
            "celery",
            "Monterey Coast Growers",
            _rect(621_600, 4_059_600, 900, 900),
            {d1: 0.68, d2: 0.66},
            stressed_patch=_rect(621_600, 4_059_600, 450, 450),  # SW quarter
        ),
        SyntheticField(
            "F-106",
            "Edge Parcel",
            "spinach",
            "Monterey Coast Growers",
            _rect(619_400, 4_061_500, 1_000, 400),
            {d1: 0.60, d2: 0.61},  # under clouds on d1
        ),
    ]


# Rows 0..CLOUD_ROWS are nodata on the first date (simulated cloud mask).
CLOUD_ROWS_DATE1 = 60  # 600 m from the top edge -> covers F-106 (y >= 4_061_400)


def write_fields_geojson(fields: list[SyntheticField], out_path: Path) -> Path:
    gdf = gpd.GeoDataFrame(
        {
            "id": [f.id for f in fields],
            "name": [f.name for f in fields],
            "crop": [f.crop for f in fields],
            "grower": [f.grower for f in fields],
        },
        geometry=[f.geometry for f in fields],
        crs=RASTER_CRS,
    ).to_crs("EPSG:4326")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path, driver="GeoJSON")
    return out_path


def write_scene(
    fields: list[SyntheticField],
    acquired_on: date,
    out_path: Path,
    *,
    seed: int = 7,
    cloud_rows: int = 0,
    noise_sd: float = 0.04,
) -> Path:
    rng = np.random.default_rng(seed + acquired_on.toordinal())
    transform = from_origin(ORIGIN_X, ORIGIN_Y, PIXEL_M, PIXEL_M)

    target = np.full((HEIGHT, WIDTH), SOIL_NDVI, dtype=np.float64)
    shapes = [(f.geometry, i + 1) for i, f in enumerate(fields)]
    field_ids = rasterize(shapes, out_shape=(HEIGHT, WIDTH), transform=transform, fill=0)
    for i, f in enumerate(fields):
        target[field_ids == i + 1] = f.ndvi_by_date.get(acquired_on, SOIL_NDVI)
        if f.stressed_patch is not None:
            patch = rasterize(
                [(f.stressed_patch, 1)], out_shape=(HEIGHT, WIDTH), transform=transform, fill=0
            )
            target[patch == 1] = 0.20

    target = np.clip(target + rng.normal(0.0, noise_sd, target.shape), -0.2, 0.95)
    nir = REFLECTANCE_SUM * (1.0 + target) / 2.0
    red = REFLECTANCE_SUM - nir
    green = red * 1.15 + rng.normal(0, 40, target.shape)
    blue = red * 0.85 + rng.normal(0, 40, target.shape)

    stack = np.stack([blue, green, red, nir]).clip(1, 10_000).astype(np.uint16)
    if cloud_rows > 0:
        stack[:, :cloud_rows, :] = NODATA

    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "dtype": "uint16",
        "count": 4,
        "width": WIDTH,
        "height": HEIGHT,
        "crs": RASTER_CRS,
        "transform": transform,
        "nodata": NODATA,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(stack)
        for idx, desc in enumerate(BAND_DESCRIPTIONS, start=1):
            dst.set_band_description(idx, desc)
        dst.update_tags(ACQUIRED_ON=acquired_on.isoformat(), SOURCE="synthetic")
    return out_path


def generate_demo(out_dir: Path) -> dict[str, Path]:
    """Write fields.geojson + one scene per demo date. Returns the written paths."""
    fields = demo_fields()
    out: dict[str, Path] = {"fields": write_fields_geojson(fields, out_dir / "fields.geojson")}
    for i, acquired in enumerate(DEMO_DATES):
        out[acquired.isoformat()] = write_scene(
            fields,
            acquired,
            out_dir / f"scene_{acquired.isoformat()}.tif",
            cloud_rows=CLOUD_ROWS_DATE1 if i == 0 else 0,
        )
    return out
