"""Scene sources.

The pipeline only ever sees a local GeoTIFF path plus an acquisition date. Where that
file comes from is a ``SceneSource``. ``LocalGeoTIFFSource`` is what the demo and tests
use; ``SentinelHubSource`` is the production seam (Sentinel Hub Process API -> L2A
B02/B03/B04/B08 at 10 m) and is intentionally left as a typed, documented stub so the
integration point is obvious without shipping credentials or network calls here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Protocol

from shapely.geometry.base import BaseGeometry


class NotConfiguredError(RuntimeError):
    """Raised when a source needs credentials or setup that is not present."""


@dataclass(frozen=True, slots=True)
class SceneRef:
    """A fetched scene ready for the pipeline."""

    path: Path
    acquired_on: date
    source: str  # "local" | "sentinel_hub"


class SceneSource(Protocol):
    name: str

    def fetch(self, aoi: BaseGeometry, acquired_on: date) -> SceneRef:
        """Return a local multispectral GeoTIFF covering ``aoi`` for ``acquired_on``."""
        ...


@dataclass(frozen=True, slots=True)
class LocalGeoTIFFSource:
    """A raster that already exists on disk."""

    path: Path
    name: str = "local"

    def fetch(self, aoi: BaseGeometry, acquired_on: date) -> SceneRef:
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        return SceneRef(path=self.path, acquired_on=acquired_on, source=self.name)


@dataclass(frozen=True, slots=True)
class SentinelHubSource:
    """Placeholder for the Sentinel Hub Process API integration.

    Production implementation would: OAuth client-credentials -> POST /api/v1/process with
    an evalscript selecting B02,B03,B04,B08 (+SCL for cloud mask) at 10 m over ``aoi``
    bounds for a +/- 2 day window around ``acquired_on`` -> write GeoTIFF to ``cache_dir``.
    """

    client_id: str = ""
    client_secret: str = ""
    cache_dir: Path = Path("data/local/sentinel")
    name: str = "sentinel_hub"

    def fetch(self, aoi: BaseGeometry, acquired_on: date) -> SceneRef:
        if not (self.client_id and self.client_secret):
            raise NotConfiguredError(
                "SentinelHubSource requires client_id/client_secret; "
                "use LocalGeoTIFFSource or `field-intel synth` for offline data."
            )
        raise NotImplementedError("Sentinel Hub fetch is out of scope for this showcase.")
