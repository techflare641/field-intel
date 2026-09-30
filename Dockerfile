# syntax=docker/dockerfile:1.7
# Multi-stage build: resolve deps with uv, ship a slim runtime with a non-root user.

FROM python:3.12-slim-bookworm AS builder
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never

# Install dependencies first (cached unless lockfile changes), then the project itself.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


FROM python:3.12-slim-bookworm AS runtime
# rasterio/geopandas wheels bundle GDAL/GEOS/PROJ; only libexpat-style basics are needed.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app

WORKDIR /app
COPY --from=builder --chown=app:app /app /app
USER app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    FIELD_INTEL_DATABASE_URL=sqlite:////app/data/local/field_intel.db \
    FIELD_INTEL_DATA_ROOT=/app/data
RUN mkdir -p /app/data/local

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["field-intel", "serve", "--host", "0.0.0.0", "--port", "8000"]
