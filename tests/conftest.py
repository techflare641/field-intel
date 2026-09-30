from __future__ import annotations

from collections.abc import Iterator
from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from field_intel.agent.llm import FakeClient
from field_intel.api.main import create_app
from field_intel.config import Settings
from field_intel.db import Database
from field_intel.pipeline.run import RunResult, run_pipeline
from field_intel.pipeline.synthetic import DEMO_DATES, generate_demo

GROWER = {"X-API-Key": "test-grower"}
OPERATOR = {"X-API-Key": "test-operator"}


@pytest.fixture(scope="session")
def demo_paths(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Synthetic fields + two scenes, generated once per test session."""
    return generate_demo(tmp_path_factory.mktemp("demo"))


@pytest.fixture
def db(tmp_path: Path) -> Database:
    return Database(f"sqlite:///{tmp_path / 'test.db'}")


@pytest.fixture
def loaded_db(db: Database, demo_paths: dict[str, Path]) -> tuple[Database, list[RunResult]]:
    """Database with both demo scenes ingested."""
    results = [
        run_pipeline(
            db,
            fields_path=demo_paths["fields"],
            raster_path=demo_paths[d.isoformat()],
            acquired_on=d,
            principal="test",
        )
        for d in DEMO_DATES
    ]
    return db, results


@pytest.fixture
def settings(tmp_path: Path, demo_paths: dict[str, Path]) -> Settings:
    return Settings(
        database_url=f"sqlite:///{tmp_path / 'api.db'}",
        llm_providers="fake",
        api_keys="test-grower:grower,test-operator:operator",
        data_root=str(demo_paths["fields"].parent),
        agent_max_steps=6,
    )


@pytest.fixture
def client(settings: Settings, loaded_db: tuple[Database, list[RunResult]]) -> Iterator[TestClient]:
    db, _ = loaded_db
    app = create_app(settings, db=db, llm=FakeClient())
    with TestClient(app) as c:
        yield c


@pytest.fixture
def first_date() -> date:
    return DEMO_DATES[0]
