"""Dhan access-token management with automatic refresh.

Dhan access tokens expire roughly 24 hours after generation, so a long-running
collector cannot use a hand-pasted token -- it has to mint a fresh one each
session. This module does that via the TOTP endpoint:

    POST https://auth.dhan.co/app/generateAccessToken
         ?dhanClientId=...&pin=...&totp=...

The TOTP code is derived locally from your authenticator seed, so no browser
step is needed and the collector can run unattended.

-------------------------------------------------------------------------
SECURITY -- read this before configuring.

The three secrets below (client id + PIN + TOTP seed) together grant FULL
account access, including placing orders. They are far more sensitive than a
read-only data key.

  * Never commit them. Never paste them into a chat or an issue.
  * In GCP, put them in Secret Manager and grant the collector's service
    account `secretAccessor` only. Do not bake them into an image or a VM
    startup script.
  * Locally, keep them in .env (already gitignored) and nowhere else.
  * If Dhan offers a data-only API key for your use case, prefer it. This
    collector never needs order placement rights.
  * Rotate immediately if a secret is ever exposed.
-------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

AUTH_URL = "https://auth.dhan.co/app/generateAccessToken"

# Refresh this long before stated expiry, so a token never dies mid-session.
REFRESH_MARGIN = timedelta(hours=2)

# Cache lives outside the repo tree by default; override with DHAN_TOKEN_CACHE.
DEFAULT_CACHE = Path.home() / ".mcube" / "dhan_token.json"


class DhanAuthError(RuntimeError):
    pass


@dataclass
class Token:
    access_token: str
    expiry: datetime
    client_id: str

    @property
    def is_fresh(self) -> bool:
        return datetime.now(timezone.utc) < self.expiry - REFRESH_MARGIN

    def to_dict(self) -> dict:
        return {
            "access_token": self.access_token,
            "expiry": self.expiry.isoformat(),
            "client_id": self.client_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Token":
        return cls(
            access_token=d["access_token"],
            expiry=datetime.fromisoformat(d["expiry"]),
            client_id=d["client_id"],
        )


def _totp_now(secret: str) -> str:
    """Current 6-digit TOTP from the authenticator seed."""
    try:
        import pyotp
    except ImportError as e:
        raise DhanAuthError("pip install pyotp -- needed for automatic token refresh") from e
    return pyotp.TOTP(secret).now()


class TokenManager:
    """Supplies a valid access token, refreshing it when it nears expiry.

    Usage:
        tm = TokenManager.from_env()
        headers = {"access-token": tm.token()}

    Call `token()` before every request rather than caching the string --
    it is cheap when the cached token is still fresh.
    """

    def __init__(
        self,
        *,
        client_id: str,
        pin: str,
        totp_secret: str,
        cache_path: Path | None = None,
    ):
        self.client_id = client_id
        self._pin = pin
        self._totp_secret = totp_secret
        self.cache_path = cache_path or Path(
            os.environ.get("DHAN_TOKEN_CACHE", DEFAULT_CACHE)
        )
        self._token: Token | None = None
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls) -> "TokenManager":
        client_id = os.environ.get("DHAN_CLIENT_ID")
        pin = os.environ.get("DHAN_PIN")
        totp_secret = os.environ.get("DHAN_TOTP_SECRET")

        missing = [
            name
            for name, val in (
                ("DHAN_CLIENT_ID", client_id),
                ("DHAN_PIN", pin),
                ("DHAN_TOTP_SECRET", totp_secret),
            )
            if not val
        ]
        if missing:
            raise DhanAuthError(
                f"Missing environment variables: {', '.join(missing)}. "
                "See .env.example. Never hard-code these."
            )
        return cls(client_id=client_id, pin=pin, totp_secret=totp_secret)

    # ------------------------------------------------------------------

    def token(self) -> str:
        """Return a valid access token, refreshing if needed."""
        with self._lock:
            if self._token is None:
                self._token = self._load_cached()

            if self._token is None or not self._token.is_fresh:
                self._token = self._generate()
                self._save(self._token)

            return self._token.access_token

    def force_refresh(self) -> str:
        """Mint a new token regardless of cache state."""
        with self._lock:
            self._token = self._generate()
            self._save(self._token)
            return self._token.access_token

    # ------------------------------------------------------------------

    def _generate(self) -> Token:
        params = {
            "dhanClientId": self.client_id,
            "pin": self._pin,
            "totp": _totp_now(self._totp_secret),
        }
        resp = requests.post(AUTH_URL, params=params, timeout=30)

        if not resp.ok:
            # Deliberately does not echo params -- they contain the PIN.
            raise DhanAuthError(
                f"Token generation failed: HTTP {resp.status_code} {resp.text[:200]}"
            )

        data = resp.json()
        access_token = data.get("accessToken")
        if not access_token:
            raise DhanAuthError(f"No accessToken in response: {str(data)[:200]}")

        expiry_raw = data.get("expiryTime")
        expiry = self._parse_expiry(expiry_raw)

        return Token(access_token=access_token, expiry=expiry, client_id=self.client_id)

    @staticmethod
    def _parse_expiry(raw) -> datetime:
        """Parse Dhan's expiryTime, falling back to a conservative 20h."""
        fallback = datetime.now(timezone.utc) + timedelta(hours=20)
        if not raw:
            return fallback
        try:
            dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            # Naive timestamps from Dhan are IST.
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone(timedelta(hours=5, minutes=30)))
            return dt.astimezone(timezone.utc)
        except (ValueError, TypeError):
            return fallback

    def _load_cached(self) -> Token | None:
        if not self.cache_path.exists():
            return None
        try:
            tok = Token.from_dict(json.loads(self.cache_path.read_text()))
        except (json.JSONDecodeError, KeyError, ValueError):
            return None
        # A cached token for a different account is not ours to use.
        return tok if tok.client_id == self.client_id else None

    def _save(self, token: Token) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(token.to_dict()))
        try:
            # Owner read/write only -- the cache holds a live credential.
            self.cache_path.chmod(0o600)
        except OSError:
            pass  # best effort; Windows ACLs differ


def token_from_env() -> str:
    """Convenience: a valid token from whichever credentials are configured.

    Prefers a static DHAN_ACCESS_TOKEN when set (useful for quick one-off
    scripts), otherwise mints one via TOTP. A long-running collector should
    always use the TOTP path so it survives the 24-hour expiry.
    """
    static = os.environ.get("DHAN_ACCESS_TOKEN")
    if static:
        return static
    return TokenManager.from_env().token()
