"""Zonal NDVI statistics for a polygon over a multispectral raster.

Pure functions: given an open rasterio dataset and a geometry *already in the raster
CRS*, return summary stats or ``None`` when the polygon has no usable pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import rasterio
from rasterio.features import geometry_mask
from rasterio.mask import mask as rio_mask
from shapely.geometry import box, mapping
from shapely.geometry.base import BaseGeometry

# Sentinel-2 style band descriptions we recognise; fall back to positional indexes.
RED_NAMES = ("B04", "red", "RED")
NIR_NAMES = ("B08", "nir", "NIR")
DEFAULT_RED_INDEX = 3
DEFAULT_NIR_INDEX = 4
EPS = 1e-6


@dataclass(frozen=True, slots=True)
class ZonalStats:
    ndvi_mean: float
    ndvi_p10: float
    ndvi_p50: float
    ndvi_p90: float
    valid_pixel_frac: float
    pixel_count: int  # pixels inside the polygon (valid + invalid)

    @property
    def valid_pixels(self) -> int:
        return round(self.pixel_count * self.valid_pixel_frac)


def resolve_band_indexes(dataset: rasterio.DatasetReader) -> tuple[int, int]:
    """Find (red, nir) 1-based band indexes by description, else defaults."""
    descriptions = [d or "" for d in (dataset.descriptions or ())]
    red = nir = None
    for idx, desc in enumerate(descriptions, start=1):
        if desc in RED_NAMES:
            red = idx
        elif desc in NIR_NAMES:
            nir = idx
    if red is None or nir is None:
        if dataset.count < DEFAULT_NIR_INDEX:
            raise ValueError(
                f"raster has {dataset.count} bands and no B04/B08 descriptions; "
                "need at least 4 bands or labelled red/nir bands"
            )
        red, nir = DEFAULT_RED_INDEX, DEFAULT_NIR_INDEX
    return red, nir


def ndvi(
    red: npt.NDArray[np.floating[Any]], nir: npt.NDArray[np.floating[Any]]
) -> npt.NDArray[np.float64]:
    """(NIR - RED) / (NIR + RED); returns NaN where the denominator is ~0."""
    red64 = red.astype(np.float64)
    nir64 = nir.astype(np.float64)
    denom = nir64 + red64
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(np.abs(denom) > EPS, (nir64 - red64) / denom, np.nan)
    return np.asarray(out, dtype=np.float64)


def intersects_raster(dataset: rasterio.DatasetReader, geom: BaseGeometry) -> bool:
    b = dataset.bounds
    return bool(geom.intersects(box(b.left, b.bottom, b.right, b.top)))


def compute_zonal_ndvi(
    dataset: rasterio.DatasetReader,
    geom: BaseGeometry,
    *,
    min_valid_frac: float = 0.2,
) -> ZonalStats | None:
    """NDVI stats for ``geom`` (raster CRS). ``None`` if no overlap or coverage too low.

    Pixels are excluded when: outside the polygon, equal to the dataset nodata value,
    masked by the dataset mask, or the NDVI denominator is zero (e.g. all-zero fill).
    """
    if geom.is_empty or not intersects_raster(dataset, geom):
        return None

    red_idx, nir_idx = resolve_band_indexes(dataset)
    try:
        data, out_transform = rio_mask(
            dataset,
            [mapping(geom)],
            crop=True,
            all_touched=False,
            filled=False,
            indexes=[red_idx, nir_idx],
        )
    except ValueError:
        # rasterio raises when shapes don't overlap after crop (edge-touching cases)
        return None

    red_masked = np.ma.asarray(data[0])
    nir_masked = np.ma.asarray(data[1])

    # Footprint of the polygon itself, independent of nodata: this is the denominator
    # for valid_pixel_frac so cloud/nodata pixels *inside* the field count as coverage loss.
    inside = geometry_mask(
        [mapping(geom)], out_shape=red_masked.shape, transform=out_transform, invert=True
    )
    pixel_count = int(inside.sum())
    if pixel_count == 0:
        return None

    # rasterio's mask combines "outside polygon" and "== nodata"; either makes a pixel invalid.
    masked_out = np.ma.getmaskarray(red_masked) | np.ma.getmaskarray(nir_masked)
    values = ndvi(np.ma.filled(red_masked, 0), np.ma.filled(nir_masked, 0))
    valid = inside & ~masked_out & np.isfinite(values)
    valid_pixels = int(valid.sum())
    valid_frac = valid_pixels / pixel_count
    if valid_pixels == 0 or valid_frac < min_valid_frac:
        return None

    sample = values[valid]
    p10, p50, p90 = np.percentile(sample, [10, 50, 90])
    return ZonalStats(
        ndvi_mean=float(sample.mean()),
        ndvi_p10=float(p10),
        ndvi_p50=float(p50),
        ndvi_p90=float(p90),
        valid_pixel_frac=float(valid_frac),
        pixel_count=pixel_count,
    )
