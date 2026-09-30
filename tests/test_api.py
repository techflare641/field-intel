from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from tests.conftest import GROWER, OPERATOR


def test_health_is_public(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["llm_providers"] == ["fake"]
    assert r.headers["X-Request-ID"]


def test_auth_required_and_validated(client: TestClient) -> None:
    assert client.get("/fields").status_code == 401
    assert client.get("/fields", headers={"X-API-Key": "nope"}).status_code == 401
    assert client.get("/fields", headers=GROWER).status_code == 200


def test_fields_and_observations(client: TestClient) -> None:
    fields = client.get("/fields", headers=GROWER).json()
    assert [f["field_id"] for f in fields] == [f"F-10{i}" for i in range(1, 7)]
    latest = {f["field_id"]: f["latest"] for f in fields}
    assert latest["F-104"]["stress_flag"] is True
    assert latest["F-101"]["stress_flag"] is False
    assert latest["F-106"]["acquired_on"] == "2026-09-24"  # only the cloud-free scene

    obs = client.get("/fields/F-103/observations", headers=GROWER).json()
    assert [o["acquired_on"] for o in obs] == ["2026-09-14", "2026-09-24"]
    assert client.get("/fields/NOPE/observations", headers=GROWER).status_code == 404


def test_pipeline_run_is_operator_only_and_path_guarded(
    client: TestClient, demo_paths: dict[str, Path]
) -> None:
    body = {
        "fields_path": "fields.geojson",
        "raster_path": "scene_2026-09-24.tif",
        "acquired_on": "2026-09-24",
    }
    assert client.post("/pipeline/run", json=body, headers=GROWER).status_code == 403

    r = client.post("/pipeline/run", json=body, headers=OPERATOR)
    assert r.status_code == 200
    assert r.json()["written"] == 0 and r.json()["skipped_existing"] == 6  # idempotent

    evil = dict(body, fields_path="../../../etc/hosts")
    assert client.post("/pipeline/run", json=evil, headers=OPERATOR).status_code == 400
    absolute = dict(body, fields_path="/etc/hosts")
    assert client.post("/pipeline/run", json=absolute, headers=OPERATOR).status_code == 400
    missing = dict(body, raster_path="nope.tif")
    assert client.post("/pipeline/run", json=missing, headers=OPERATOR).status_code == 404


def test_ask_then_hitl_approval_flow(client: TestClient) -> None:
    r = client.post(
        "/ask", json={"question": "Should we scout any stressed fields?"}, headers=GROWER
    )
    assert r.status_code == 200
    body = r.json()
    assert body["complete"] is True
    assert body["citations"]
    assert len(body["pending_actions"]) == 1
    action_id = body["pending_actions"][0]

    pending = client.get("/actions", params={"status_filter": "pending"}, headers=GROWER).json()
    assert [a["id"] for a in pending] == [action_id]
    assert pending[0]["proposed_by"].startswith("agent:grower:")
    assert set(pending[0]["evidence"]) <= set(body["citations"])

    # growers cannot approve
    assert client.post(f"/actions/{action_id}/approve", headers=GROWER).status_code == 403

    r = client.post(f"/actions/{action_id}/approve", json={"note": "go"}, headers=OPERATOR)
    assert r.status_code == 200
    assert r.json()["status"] == "approved"
    assert r.json()["decided_by"].startswith("operator:")
    assert r.json()["decision_note"] == "go"

    # decisions are final
    assert client.post(f"/actions/{action_id}/reject", headers=OPERATOR).status_code == 409
    assert client.post("/actions/9999/approve", headers=OPERATOR).status_code == 404


def test_ask_validates_question_length(client: TestClient) -> None:
    assert client.post("/ask", json={"question": "hi"}, headers=GROWER).status_code == 422
