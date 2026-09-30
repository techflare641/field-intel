from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from field_intel.pipeline.zonal import compute_zonal_ndvi, ndvi, resolve_band_indexes


def _write_raster(
    path: Path,
    red: np.ndarray,
    nir: np.ndarray,
    *,
    nodata: int | None = 0,
    describe: bool = True,
) -> Path:
    h, w = red.shape
    transform = from_origin(0, h * 10, 10, 10)  # 10 m pixels, origin at (0, top)
    zeros = np.zeros_like(red)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        dtype="uint16",
        count=4,
        width=w,
        height=h,
        crs="EPSG:32610",
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(np.stack([zeros, zeros, red, nir]).astype(np.uint16))
        if describe:
            for i, d in enumerate(("B02", "B03", "B04", "B08"), start=1):
                dst.set_band_description(i, d)
    return path


def test_ndvi_math_and_zero_denominator() -> None:
    red = np.array([[1000.0, 0.0, 2000.0]])
    nir = np.array([[3000.0, 0.0, 2000.0]])
    out = ndvi(red, nir)
    assert out[0, 0] == pytest.approx(0.5)
    assert np.isnan(out[0, 1])  # 0/0 -> NaN, not an exception
    assert out[0, 2] == pytest.approx(0.0)


def test_zonal_stats_known_values(tmp_path: Path) -> None:
    # 20x20 raster; left half NDVI 0.6, right half NDVI 0.2
    red = np.full((20, 20), 2000.0)
    nir = np.where(np.broadcast_to(np.arange(20)[None, :], (20, 20)) < 10, 8000.0, 3000.0)
    path = _write_raster(tmp_path / "r.tif", red, nir)
    with rasterio.open(path) as ds:
        left = compute_zonal_ndvi(ds, box(0, 0, 100, 200))
        whole = compute_zonal_ndvi(ds, box(0, 0, 200, 200))
    assert left is not None and whole is not None
    assert left.ndvi_mean == pytest.approx(0.6)
    assert left.pixel_count == 200
    assert left.valid_pixel_frac == 1.0
    assert whole.ndvi_mean == pytest.approx(0.4)
    assert whole.ndvi_p10 == pytest.approx(0.2)
    assert whole.ndvi_p90 == pytest.approx(0.6)


def test_nodata_pixels_are_excluded(tmp_path: Path) -> None:
    red = np.full((10, 10), 2000.0)
    nir = np.full((10, 10), 8000.0)
    red[:5, :] = 0  # top half nodata in both bands
    nir[:5, :] = 0
    path = _write_raster(tmp_path / "r.tif", red, nir, nodata=0)
    with rasterio.open(path) as ds:
        stats = compute_zonal_ndvi(ds, box(0, 0, 100, 100), min_valid_frac=0.2)
        too_cloudy = compute_zonal_ndvi(ds, box(0, 0, 100, 100), min_valid_frac=0.6)
    assert stats is not None
    assert stats.valid_pixel_frac == pytest.approx(0.5)
    assert stats.ndvi_mean == pytest.approx(0.6)  # nodata did not drag the mean down
    assert too_cloudy is None  # coverage guard


def test_no_overlap_and_partial_overlap(tmp_path: Path) -> None:
    red = np.full((10, 10), 2000.0)
    nir = np.full((10, 10), 8000.0)
    path = _write_raster(tmp_path / "r.tif", red, nir)
    with rasterio.open(path) as ds:
        assert compute_zonal_ndvi(ds, box(500, 500, 600, 600)) is None
        partial = compute_zonal_ndvi(ds, box(50, 50, 300, 300))  # only 5x5 inside
    assert partial is not None
    assert partial.pixel_count == 25
    assert partial.valid_pixel_frac == 1.0


def test_band_resolution_by_description_and_fallback(tmp_path: Path) -> None:
    red = np.full((4, 4), 1.0)
    nir = np.full((4, 4), 1.0)
    with rasterio.open(_write_raster(tmp_path / "a.tif", red, nir, describe=True)) as ds:
        assert resolve_band_indexes(ds) == (3, 4)
    with rasterio.open(_write_raster(tmp_path / "b.tif", red, nir, describe=False)) as ds:
        assert resolve_band_indexes(ds) == (3, 4)
