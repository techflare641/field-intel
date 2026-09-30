"""API-key authentication and role checks.

Deliberately simple: ``X-API-Key`` -> ``Principal``. In production this would be
Supabase/Firebase JWT verification with the same ``Principal`` shape, so route code
does not change. Keys are compared with ``hmac.compare_digest`` to avoid timing leaks.
"""

from __future__ import annotations

import hmac
from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException, Request, status

from field_intel.config import Role, Settings


@dataclass(frozen=True, slots=True)
class Principal:
    key_id: str  # short, non-secret label for logs/audit
    role: Role

    @property
    def audit_name(self) -> str:
        return f"{self.role}:{self.key_id}"


def _settings(request: Request) -> Settings:
    return request.app.state.settings  # type: ignore[no-any-return]


def current_principal(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> Principal:
    if not x_api_key:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing X-API-Key")
    for key, role in _settings(request).api_key_roles.items():
        if hmac.compare_digest(key.encode(), x_api_key.encode()):
            return Principal(key_id=_label(key), role=role)
    raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid API key")


def require_operator(principal: Principal = Depends(current_principal)) -> Principal:
    if principal.role != Role.OPERATOR:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "operator role required")
    return principal


def _label(key: str) -> str:
    """Non-reversible short label for an API key (first 4 chars + length)."""
    return f"{key[:4]}…{len(key)}"
