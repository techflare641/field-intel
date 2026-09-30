"""Runtime configuration.

Everything is env-driven (prefix ``FIELD_INTEL_``) with defaults that make the
offline demo work with no secrets: SQLite on disk and the deterministic ``fake``
LLM provider.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Role(StrEnum):
    """Principal roles. ``operator`` may approve/reject agent-proposed actions."""

    GROWER = "grower"
    OPERATOR = "operator"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FIELD_INTEL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "sqlite:///./data/local/field_intel.db"
    # POST /pipeline/run may only read files under this directory (path-traversal guard).
    data_root: str = "data"

    # LLM gateway: ordered provider list; first healthy provider wins, others are fallbacks.
    llm_providers: str = "fake"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-20250514"
    anthropic_base_url: str = "https://api.anthropic.com"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"
    llm_timeout_seconds: float = 30.0

    # Agent limits (blast-radius controls)
    agent_max_steps: int = Field(default=6, ge=1, le=20)
    tool_result_max_chars: int = Field(default=6000, ge=500)

    # ``key:role,key:role``
    api_keys: str = "dev-grower-key:grower,dev-operator-key:operator"

    stress_ndvi_threshold: float = Field(default=0.35, ge=-1.0, le=1.0)

    @field_validator("llm_providers")
    @classmethod
    def _validate_providers(cls, value: str) -> str:
        allowed = {"fake", "anthropic", "openai"}
        providers = [p.strip() for p in value.split(",") if p.strip()]
        unknown = set(providers) - allowed
        if unknown:
            raise ValueError(
                f"unknown LLM providers: {sorted(unknown)}; allowed: {sorted(allowed)}"
            )
        if not providers:
            raise ValueError("at least one LLM provider is required")
        return ",".join(providers)

    @property
    def provider_order(self) -> list[str]:
        return [p.strip() for p in self.llm_providers.split(",") if p.strip()]

    @property
    def api_key_roles(self) -> dict[str, Role]:
        mapping: dict[str, Role] = {}
        for pair in self.api_keys.split(","):
            pair = pair.strip()
            if not pair:
                continue
            key, _, role = pair.partition(":")
            mapping[key.strip()] = Role(role.strip() or Role.GROWER)
        return mapping


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
