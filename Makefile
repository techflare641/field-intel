.PHONY: install lint typecheck test check demo serve docker clean

DATA_DIR ?= data/local

install:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run mypy

test:
	uv run pytest

check: lint typecheck test

# End-to-end offline demo: synthetic scene -> pipeline -> ask the agent (FakeClient, no API keys).
demo:
	uv run field-intel synth --out-dir $(DATA_DIR)
	uv run field-intel run --fields $(DATA_DIR)/fields.geojson --raster $(DATA_DIR)/scene_2026-09-14.tif --acquired 2026-09-14
	uv run field-intel run --fields $(DATA_DIR)/fields.geojson --raster $(DATA_DIR)/scene_2026-09-24.tif --acquired 2026-09-24
	uv run field-intel run --fields $(DATA_DIR)/fields.geojson --raster $(DATA_DIR)/scene_2026-09-24.tif --acquired 2026-09-24
	uv run field-intel ask "Which fields look stressed this week, and should we scout any of them?"

serve:
	uv run field-intel serve --reload

docker:
	docker build -t field-intel:local .

clean:
	rm -rf $(DATA_DIR) .pytest_cache .mypy_cache .ruff_cache
