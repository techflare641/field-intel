"""Provider-neutral LLM client with tool calling.

A tiny gateway instead of a vendor SDK: one ``Message``/``ToolCall`` shape, adapters for
Anthropic Messages and OpenAI Chat Completions over ``httpx``, a deterministic
``FakeClient`` for tests and the offline demo, and a ``Gateway`` that walks an ordered
provider list and falls back on transport/5xx failures.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import httpx

Role = Literal["user", "assistant", "tool"]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class Message:
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)  # assistant only
    tool_call_id: str | None = None  # tool only
    tool_name: str | None = None  # tool only


@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str
    tool_calls: list[ToolCall]
    provider: str
    model: str

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class LLMProviderError(RuntimeError):
    """Transport / server-side failure that is safe to retry on another provider."""


class LLMClient(Protocol):
    name: str

    def complete(
        self, *, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> LLMResponse: ...


# ------------------------------------------------------------------------ Anthropic


class AnthropicClient:
    name = "anthropic"

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str = "https://api.anthropic.com",
        timeout: float = 30.0,
        http: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("Anthropic API key is empty")
        self.model = model
        self._http = http or httpx.Client(base_url=base_url, timeout=timeout)
        self._headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    def complete(
        self, *, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> LLMResponse:
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 1024,
            "system": system,
            "messages": _to_anthropic_messages(messages),
        }
        if tools:
            body["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters}
                for t in tools
            ]
        data = _post(self._http, "/v1/messages", body, self._headers, self.name)

        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in data.get("content", []):
            if block.get("type") == "text":
                text_parts.append(str(block.get("text", "")))
            elif block.get("type") == "tool_use":
                calls.append(
                    ToolCall(
                        id=str(block["id"]),
                        name=str(block["name"]),
                        arguments=dict(block.get("input") or {}),
                    )
                )
        return LLMResponse("\n".join(text_parts).strip(), calls, self.name, self.model)


def _to_anthropic_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "user":
            out.append({"role": "user", "content": m.content})
        elif m.role == "assistant":
            blocks: list[dict[str, Any]] = []
            if m.content:
                blocks.append({"type": "text", "text": m.content})
            for tc in m.tool_calls:
                blocks.append(
                    {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
                )
            out.append({"role": "assistant", "content": blocks})
        else:  # tool result -> user turn with tool_result block; merge consecutive ones
            block = {"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.content}
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
    return out


# --------------------------------------------------------------------------- OpenAI


class OpenAIClient:
    name = "openai"

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 30.0,
        http: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("OpenAI API key is empty")
        self.model = model
        self._http = http or httpx.Client(base_url=base_url, timeout=timeout)
        self._headers = {"authorization": f"Bearer {api_key}", "content-type": "application/json"}

    def complete(
        self, *, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> LLMResponse:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *_to_openai_messages(messages)],
        }
        if tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]
        data = _post(self._http, "/chat/completions", body, self._headers, self.name)
        msg = data["choices"][0]["message"]
        calls = [
            ToolCall(
                id=str(tc["id"]),
                name=str(tc["function"]["name"]),
                arguments=_loads_args(tc["function"].get("arguments")),
            )
            for tc in msg.get("tool_calls") or []
        ]
        return LLMResponse((msg.get("content") or "").strip(), calls, self.name, self.model)


def _to_openai_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "assistant":
            entry: dict[str, Any] = {"role": "assistant", "content": m.content or None}
            if m.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in m.tool_calls
                ]
            out.append(entry)
        elif m.role == "tool":
            out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
        else:
            out.append({"role": "user", "content": m.content})
    return out


def _loads_args(raw: Any) -> dict[str, Any]:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _post(
    http: httpx.Client, path: str, body: dict[str, Any], headers: dict[str, str], provider: str
) -> dict[str, Any]:
    try:
        resp = http.post(path, json=body, headers=headers)
    except httpx.HTTPError as exc:
        raise LLMProviderError(f"{provider}: transport error: {exc}") from exc
    if resp.status_code >= 500 or resp.status_code == 429:
        raise LLMProviderError(f"{provider}: HTTP {resp.status_code}")
    if resp.status_code >= 400:
        # 4xx other than 429 is a caller bug (bad key, bad schema): do not fall back silently.
        raise RuntimeError(f"{provider}: HTTP {resp.status_code}: {resp.text[:300]}")
    payload: Any = resp.json()
    return dict(payload)


# ----------------------------------------------------------------------------- Fake


class FakeClient:
    """Deterministic, network-free policy that exercises the real tool loop.

    Turn 1: rank stressed fields. Turn 2: pull the trend for the worst one. Turn 3: if the
    user asked about scouting/action, propose a scout with the observation ids as
    evidence. Then write a plain-language summary citing the observation ids it saw.
    """

    name = "fake"
    model = "fake-policy-v1"

    def complete(
        self, *, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> LLMResponse:
        del system
        tool_names = {t.name for t in tools}
        question = next((m.content for m in messages if m.role == "user"), "").lower()
        results = [m for m in messages if m.role == "tool"]
        seen = {m.tool_name for m in results}

        if "rank_stressed_fields" in tool_names and "rank_stressed_fields" not in seen:
            return self._call("rank_stressed_fields", {"top_n": 5})

        ranked = _first_result(results, "rank_stressed_fields")
        worst = ranked[0] if ranked else None

        if worst and "get_field_trend" in tool_names and "get_field_trend" not in seen:
            return self._call("get_field_trend", {"field_id": worst["field_id"], "days": 30})

        wants_action = any(w in question for w in ("scout", "should we", "action", "do about"))
        if (
            worst
            and worst.get("stress_flag")
            and wants_action
            and "propose_action" in tool_names
            and "propose_action" not in seen
        ):
            trend = _first_result(results, "get_field_trend")
            evidence = sorted(
                {int(worst["observation_id"])} | {int(t["observation_id"]) for t in trend}
            )
            delta = float(trend[-1].get("ndvi_mean_delta", 0.0)) if len(trend) > 1 else 0.0
            direction = "declining" if delta < -0.05 else "improving" if delta > 0.05 else "flat"
            return self._call(
                "propose_action",
                {
                    "field_id": worst["field_id"],
                    "action": "scout",
                    "reason": (
                        f"Latest mean NDVI {worst['ndvi_mean']:.2f} is below the stress "
                        f"threshold {worst['threshold']:.2f}; 30-day trend is {direction} "
                        f"({delta:+.2f})."
                    ),
                    "evidence_observation_ids": evidence,
                },
            )

        return LLMResponse(self._summarise(ranked, results), [], self.name, self.model)

    def _call(self, name: str, args: dict[str, Any]) -> LLMResponse:
        return LLMResponse(
            "", [ToolCall(f"call_{uuid.uuid4().hex[:8]}", name, args)], self.name, self.model
        )

    @staticmethod
    def _summarise(ranked: list[dict[str, Any]], results: list[Message]) -> str:
        if not ranked:
            return "No field observations are available yet. Run the pipeline first."
        stressed = [r for r in ranked if r.get("stress_flag")]
        lines = []
        if stressed:
            lines.append("Fields currently flagged as stressed (latest scene):")
            for r in stressed:
                lines.append(
                    f"- {r['field_id']} {r['name']} ({r['crop']}): mean NDVI "
                    f"{r['ndvi_mean']:.2f}, p10 {r['ndvi_p10']:.2f} [obs {r['observation_id']}]"
                )
        else:
            lines.append("No fields are below the stress threshold in the latest scene.")
        patchy = [r for r in ranked if not r.get("stress_flag") and r["ndvi_p10"] < 0.3]
        for r in patchy:
            lines.append(
                f"- {r['field_id']} {r['name']} is healthy on average but has a low-NDVI patch "
                f"(p10 {r['ndvi_p10']:.2f}) worth a look [obs {r['observation_id']}]"
            )
        action = _first_result(results, "propose_action")
        if action:
            a = action[0]
            lines.append(
                f"Proposed action #{a['action_request_id']}: {a['action']} on {a['field_id']} "
                f"- pending operator approval."
            )
        return "\n".join(lines)


def _first_result(results: list[Message], tool_name: str) -> list[dict[str, Any]]:
    for m in results:
        if m.tool_name == tool_name:
            try:
                payload: Any = json.loads(m.content)
            except json.JSONDecodeError:
                return []
            data = payload.get("data") if isinstance(payload, dict) else None
            if isinstance(data, list):
                return [dict(x) for x in data]
            if isinstance(data, dict):
                return [dict(data)]
            return []
    return []


# -------------------------------------------------------------------------- Gateway


class Gateway:
    """Ordered provider list with fallback on ``LLMProviderError``."""

    name = "gateway"

    def __init__(self, clients: Sequence[LLMClient]) -> None:
        if not clients:
            raise ValueError("Gateway needs at least one client")
        self.clients = list(clients)
        self.last_used: str | None = None

    def complete(
        self, *, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> LLMResponse:
        errors: list[str] = []
        for client in self.clients:
            try:
                resp = client.complete(system=system, messages=messages, tools=tools)
            except LLMProviderError as exc:
                errors.append(str(exc))
                continue
            self.last_used = client.name
            return resp
        raise LLMProviderError("all providers failed: " + "; ".join(errors))


def build_gateway(
    providers: Sequence[str],
    *,
    anthropic_api_key: str = "",
    anthropic_model: str = "",
    anthropic_base_url: str = "https://api.anthropic.com",
    openai_api_key: str = "",
    openai_model: str = "",
    openai_base_url: str = "https://api.openai.com/v1",
    timeout: float = 30.0,
) -> Gateway:
    clients: list[LLMClient] = []
    for p in providers:
        if p == "fake":
            clients.append(FakeClient())
        elif p == "anthropic":
            clients.append(
                AnthropicClient(
                    anthropic_api_key, anthropic_model, base_url=anthropic_base_url, timeout=timeout
                )
            )
        elif p == "openai":
            clients.append(
                OpenAIClient(
                    openai_api_key, openai_model, base_url=openai_base_url, timeout=timeout
                )
            )
        else:
            raise ValueError(f"unknown provider {p!r}")
    return Gateway(clients)
