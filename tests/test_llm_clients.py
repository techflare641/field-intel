"""Wire-format tests for the Anthropic and OpenAI adapters using an httpx MockTransport."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from field_intel.agent.llm import (
    AnthropicClient,
    LLMProviderError,
    Message,
    OpenAIClient,
    ToolCall,
    ToolSpec,
)

SPEC = ToolSpec("rank_stressed_fields", "rank", {"type": "object", "properties": {}})


def _mock(status: int, body: dict[str, Any], seen: list[dict[str, Any]]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append({"path": request.url.path, "json": json.loads(request.content)})
        return httpx.Response(status, json=body)

    return httpx.Client(transport=httpx.MockTransport(handler), base_url="https://x.test")


def test_anthropic_request_and_tool_use_parsing() -> None:
    seen: list[dict[str, Any]] = []
    body = {
        "content": [
            {"type": "text", "text": "Checking."},
            {
                "type": "tool_use",
                "id": "tu_1",
                "name": "rank_stressed_fields",
                "input": {"top_n": 3},
            },
        ]
    }
    client = AnthropicClient("k", "claude-test", http=_mock(200, body, seen))
    history = [
        Message("user", "which fields?"),
        Message("assistant", "", [ToolCall("tu_0", "list_fields", {})]),
        Message("tool", '{"data": []}', tool_call_id="tu_0", tool_name="list_fields"),
    ]
    resp = client.complete(system="sys", messages=history, tools=[SPEC])

    assert resp.text == "Checking."
    assert resp.tool_calls == [ToolCall("tu_1", "rank_stressed_fields", {"top_n": 3})]
    req = seen[0]["json"]
    assert seen[0]["path"] == "/v1/messages"
    assert req["system"] == "sys"
    assert req["tools"][0]["input_schema"] == SPEC.parameters
    # tool result must be sent back as a user turn with a tool_result block
    assert req["messages"][-1]["role"] == "user"
    assert req["messages"][-1]["content"][0]["type"] == "tool_result"
    assert req["messages"][-1]["content"][0]["tool_use_id"] == "tu_0"


def test_openai_request_and_tool_call_parsing() -> None:
    seen: list[dict[str, Any]] = []
    body = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "rank_stressed_fields",
                                "arguments": '{"top_n": 2}',
                            },
                        }
                    ],
                }
            }
        ]
    }
    client = OpenAIClient("k", "gpt-test", http=_mock(200, body, seen))
    history = [
        Message("user", "q"),
        Message("assistant", "", [ToolCall("call_0", "list_fields", {"grower": "x"})]),
        Message("tool", "{}", tool_call_id="call_0", tool_name="list_fields"),
    ]
    resp = client.complete(system="sys", messages=history, tools=[SPEC])

    assert resp.text == ""
    assert resp.tool_calls == [ToolCall("call_1", "rank_stressed_fields", {"top_n": 2})]
    req = seen[0]["json"]
    assert req["messages"][0] == {"role": "system", "content": "sys"}
    assert req["messages"][2]["tool_calls"][0]["function"]["arguments"] == '{"grower": "x"}'
    assert req["messages"][3] == {"role": "tool", "tool_call_id": "call_0", "content": "{}"}
    assert req["tools"][0]["function"]["name"] == "rank_stressed_fields"


@pytest.mark.parametrize("status", [429, 500, 503])
def test_retryable_statuses_raise_provider_error(status: int) -> None:
    client = OpenAIClient("k", "m", http=_mock(status, {}, []))
    with pytest.raises(LLMProviderError):
        client.complete(system="s", messages=[Message("user", "q")], tools=[])


def test_client_errors_are_not_silently_retried() -> None:
    client = AnthropicClient("k", "m", http=_mock(401, {"error": "bad key"}, []))
    with pytest.raises(RuntimeError, match="HTTP 401"):
        client.complete(system="s", messages=[Message("user", "q")], tools=[])


def test_empty_api_key_is_rejected_early() -> None:
    with pytest.raises(ValueError, match="empty"):
        AnthropicClient("", "m")
    with pytest.raises(ValueError, match="empty"):
        OpenAIClient("", "m")
