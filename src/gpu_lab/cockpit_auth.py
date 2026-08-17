"""Small private-cockpit session primitives.

Cockpit authentication is intentionally separate from MCP credentials.  It
never exposes database, browser-profile, or MCP secrets to the browser.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class CockpitSession:
    expires_at: int
    csrf_token: str


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _unb64(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def issue_session(secret: str, *, lifetime_seconds: int = 28_800) -> tuple[str, CockpitSession]:
    """Create a signed, expiring, opaque-enough operator session cookie."""
    session = CockpitSession(int(time.time()) + lifetime_seconds, secrets.token_urlsafe(32))
    payload = _b64(json.dumps({"exp": session.expires_at, "csrf": session.csrf_token}, separators=(",", ":")).encode())
    signature = _b64(hmac.new(secret.encode(), payload.encode(), hashlib.sha256).digest())
    return f"{payload}.{signature}", session


def verify_session(secret: str, value: str | None) -> CockpitSession | None:
    if not value or "." not in value:
        return None
    payload, signature = value.rsplit(".", 1)
    expected = _b64(hmac.new(secret.encode(), payload.encode(), hashlib.sha256).digest())
    if not hmac.compare_digest(signature, expected):
        return None
    try:
        data = json.loads(_unb64(payload))
        expires_at = int(data["exp"])
        csrf = str(data["csrf"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if expires_at <= int(time.time()) or len(csrf) < 24:
        return None
    return CockpitSession(expires_at, csrf)


def password_matches(configured: str | None, submitted: str | None) -> bool:
    return bool(configured and submitted) and hmac.compare_digest(configured, submitted)
