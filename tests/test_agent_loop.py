from __future__ import annotations

import json
from collections.abc import Sequence

import pytest
from sqlalchemy import select

from field_intel.agent.llm import (
    FakeClient,
    Gateway,
    LLMProviderError,
    LLMResponse,
    Message,
    ToolCall,
    ToolSpec,
)
from field_intel.agent.loop import run_agent
from field_intel.db import ActionRequest, ActionStatus, AuditEvent, Database
from field_intel.pipeline.run import RunResult

THRESHOLD = 0.35


class ScriptedClient:
    """Replays a fixed list of responses; records every request it saw."""

    name = "scripted"

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.requests: list[list[Message]] = []

    def complete(
        self, *, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> LLMResponse:
        del system, tools
        self.requests.append(list(messages))
        if not self._responses:
            return LLMResponse("done", [], self.name, "scripted")
        return self._responses.pop(0)


def _tool(name: str, **args: object) -> LLMResponse:
    return LLMResponse("", [ToolCall(f"c{len(name)}", name, dict(args))], "scripted", "s")


def _text(t: str) -> LLMResponse:
    return LLMResponse(t, [], "scripted", "s")


def _last_tool_payload(client: ScriptedClient) -> dict[str, object]:
    tool_msgs = [m for m in client.requests[-1] if m.role == "tool"]
    return dict(json.loads(tool_msgs[-1].content))


def test_fake_client_end_to_end(loaded_db: tuple[Database, list[RunResult]]) -> None:
    db, _ = loaded_db
    with db.session() as s:
        answer = run_agent(
            "Which fields are stressed and should we scout any?",
            session=s,
            llm=FakeClient(),
            principal="test",
            stress_threshold=THRESHOLD,
        )
        pending = s.scalars(select(ActionRequest)).all()
        audit = s.scalars(select(AuditEvent).where(AuditEvent.kind == "agent.turn")).all()

    assert answer.complete
    assert "F-104" in answer.text and "F-103" in answer.text
    assert answer.citations  # provenance present
    assert all("[obs " in line for line in answer.text.splitlines() if line.startswith("- "))
    assert answer.pending_actions == [pending[0].id]
    assert pending[0].status == ActionStatus.PENDING
    assert pending[0].field_id == "F-104"
    assert set(pending[0].evidence) <= set(answer.citations)
    assert len(audit) == 1
    assert audit[0].payload["trace"][0]["tool"] == "rank_stressed_fields"


def test_unknown_tool_is_rejected_not_raised(
    loaded_db: tuple[Database, list[RunResult]],
) -> None:
    db, _ = loaded_db
    client = ScriptedClient([_tool("drop_table", table="fields"), _text("ok")])
    with db.session() as s:
        answer = run_agent("hi", session=s, llm=client, principal="t", stress_threshold=THRESHOLD)
    assert answer.complete
    assert answer.trace[0].ok is False
    assert "unknown tool" in (answer.trace[0].error or "")
    assert "unknown tool" in str(_last_tool_payload(client)["error"])


def test_invalid_arguments_become_tool_error(
    loaded_db: tuple[Database, list[RunResult]],
) -> None:
    db, _ = loaded_db
    client = ScriptedClient(
        [
            _tool("get_field_trend", field_id="F-103", days=9999, extra="nope"),
            _text("ok"),
        ]
    )
    with db.session() as s:
        answer = run_agent("hi", session=s, llm=client, principal="t", stress_threshold=THRESHOLD)
    assert answer.trace[0].ok is False
    assert "invalid arguments" in (answer.trace[0].error or "")
    assert answer.citations == []


def test_max_steps_is_enforced(loaded_db: tuple[Database, list[RunResult]]) -> None:
    db, _ = loaded_db
    client = ScriptedClient([_tool("list_fields")] * 50)
    with db.session() as s:
        answer = run_agent(
            "loop forever",
            session=s,
            llm=client,
            principal="t",
            stress_threshold=THRESHOLD,
            max_steps=3,
        )
    assert not answer.complete
    assert answer.steps == 3
    assert len(answer.trace) == 3
    assert "stopped after 3" in answer.text


def test_propose_action_requires_matching_evidence(
    loaded_db: tuple[Database, list[RunResult]],
) -> None:
    db, _ = loaded_db
    client = ScriptedClient(
        [
            _tool("get_field_trend", field_id="F-101"),
            # evidence id 9999 does not exist / belong to F-101
            _tool(
                "propose_action",
                field_id="F-101",
                action="scout",
                reason="looks bad to me honestly",
                evidence_observation_ids=[9999],
            ),
            _text("done"),
        ]
    )
    with db.session() as s:
        answer = run_agent("x", session=s, llm=client, principal="t", stress_threshold=THRESHOLD)
        assert s.scalars(select(ActionRequest)).all() == []
    assert answer.pending_actions == []
    assert answer.trace[1].ok is False
    assert "do not belong" in (answer.trace[1].error or "")


def test_pii_is_redacted_before_reaching_model(
    loaded_db: tuple[Database, list[RunResult]],
) -> None:
    db, _ = loaded_db
    client = ScriptedClient([_text("ok")])
    with db.session() as s:
        answer = run_agent(
            "email me at jane@example.com or 831-555-0100",
            session=s,
            llm=client,
            principal="t",
            stress_threshold=THRESHOLD,
        )
    assert answer.redactions == 2
    sent = client.requests[0][0].content
    assert "jane@example.com" not in sent and "831-555-0100" not in sent
    assert "[email]" in sent and "[phone]" in sent


def test_tool_output_is_truncated(loaded_db: tuple[Database, list[RunResult]]) -> None:
    db, _ = loaded_db
    client = ScriptedClient([_tool("list_fields"), _text("ok")])
    with db.session() as s:
        answer = run_agent(
            "x",
            session=s,
            llm=client,
            principal="t",
            stress_threshold=THRESHOLD,
            tool_result_max_chars=600,
        )
    assert answer.trace[0].truncated is True
    tool_msg = next(m for m in client.requests[1] if m.role == "tool")
    assert len(tool_msg.content) <= 600
    assert "[truncated" in tool_msg.content


class _Failing:
    name = "failing"

    def complete(self, **_: object) -> LLMResponse:
        raise LLMProviderError("failing: HTTP 503")


def test_gateway_falls_back_on_provider_error() -> None:
    gw = Gateway([_Failing(), FakeClient()])
    resp = gw.complete(system="s", messages=[Message("user", "hi")], tools=[])
    assert gw.last_used == "fake"
    assert resp.provider == "fake"

    with pytest.raises(LLMProviderError, match="all providers failed"):
        Gateway([_Failing()]).complete(system="s", messages=[], tools=[])
