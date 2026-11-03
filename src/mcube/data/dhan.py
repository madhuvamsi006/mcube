"""DhanHQ v2 data client.

Credentials come from the environment, never from code or chat:

    DHAN_CLIENT_ID=100xxxxxxx
    DHAN_ACCESS_TOKEN=eyJ...

Key API facts this client is built around (DhanHQ v2 docs):

  Historical intraday   POST /v2/charts/intraday
                        1/5/15/25/60-min candles, ~5 years back, OI included.
                        Max 90 days per request -> this client chunks for you.

  Rate limits           Data APIs: 5 req/sec, 100,000/day. The client throttles
                        to stay under the per-second cap.

  200-level depth       wss://full-depth-api.dhan.co/twohundreddepth
                        ONE instrument per connection, max 5 sockets total.
                        So deep book is for the future only, not the chain.

  20-level depth        50 instruments per connection -- this is what the
                        option chain must use.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd
import requests

from mcube.data.dhan_auth import TokenManager, token_from_env

BASE_URL = "https://api.dhan.co/v2"
IST = "Asia/Kolkata"

# Data APIs allow 5 req/sec; stay just under it.
MIN_REQUEST_INTERVAL = 0.22
MAX_DAYS_PER_REQUEST = 90


class DhanError(RuntimeError):
    pass


@dataclass
class DhanCredentials:
    client_id: str
    access_token: str

    @classmethod
    def from_env(cls) -> "DhanCredentials":
        """Static token from the environment, for quick one-off scripts.

        Long-running jobs should pass a TokenManager to DhanClient instead --
        Dhan tokens expire in ~24h and a static one will die mid-session.
        """
        client_id = os.environ.get("DHAN_CLIENT_ID")
        if not client_id:
            raise DhanError("Set DHAN_CLIENT_ID (see .env.example).")
        return cls(client_id=client_id, access_token=token_from_env())


class DhanClient:
    """Thin REST client for the Dhan data APIs."""

    def __init__(
        self,
        creds: DhanCredentials | None = None,
        token_manager: TokenManager | None = None,
    ):
        # token_manager wins when supplied: it refreshes before expiry, which
        # is what any process living longer than a day needs.
        self.token_manager = token_manager
        self.creds = creds or DhanCredentials.from_env()
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "access-token": self.creds.access_token,
                "client-id": self.creds.client_id,
            }
        )
        self._last_request = 0.0

    def _auth_header(self) -> None:
        """Refresh the access-token header if a TokenManager is in play."""
        if self.token_manager is not None:
            self._session.headers["access-token"] = self.token_manager.token()

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)
        self._last_request = time.monotonic()

    def _post(self, path: str, payload: dict) -> dict:
        self._auth_header()
        self._throttle()
        resp = self._session.post(f"{BASE_URL}{path}", json=payload, timeout=30)

        if resp.status_code == 429:
            # Rate limited despite throttling -- back off and retry once.
            time.sleep(2.0)
            self._throttle()
            resp = self._session.post(f"{BASE_URL}{path}", json=payload, timeout=30)

        if not resp.ok:
            raise DhanError(f"{path} -> HTTP {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        if isinstance(data, dict) and data.get("errorCode"):
            raise DhanError(f"{path} -> {data.get('errorCode')}: {data.get('errorMessage')}")
        return data

    # ---------------------------------------------------------------- bars

    def intraday(
        self,
        *,
        security_id: str,
        exchange_segment: str,
        instrument: str,
        from_date: str,
        to_date: str,
        interval: int = 1,
        oi: bool = True,
    ) -> pd.DataFrame:
        """Fetch intraday candles, chunking automatically past the 90-day cap.

        Returns a DataFrame indexed by IST timestamp with
        [open, high, low, close, volume, open_interest].
        """
        if interval not in (1, 5, 15, 25, 60):
            raise ValueError(f"interval must be one of 1/5/15/25/60, got {interval}")

        start = datetime.fromisoformat(from_date)
        end = datetime.fromisoformat(to_date)
        frames: list[pd.DataFrame] = []

        while start < end:
            chunk_end = min(start + timedelta(days=MAX_DAYS_PER_REQUEST), end)
            payload = {
                "securityId": str(security_id),
                "exchangeSegment": exchange_segment,
                "instrument": instrument,
                "interval": int(interval),
                "oi": bool(oi),
                "fromDate": start.strftime("%Y-%m-%d %H:%M:%S"),
                "toDate": chunk_end.strftime("%Y-%m-%d %H:%M:%S"),
            }
            data = self._post("/charts/intraday", payload)
            df = self._to_frame(data)
            if not df.empty:
                frames.append(df)
            start = chunk_end

        if not frames:
            return pd.DataFrame()

        out = pd.concat(frames).sort_index()
        return out[~out.index.duplicated(keep="first")]

    def daily(
        self,
        *,
        security_id: str,
        exchange_segment: str,
        instrument: str,
        from_date: str,
        to_date: str,
        oi: bool = True,
        expiry_code: int | None = None,
    ) -> pd.DataFrame:
        payload = {
            "securityId": str(security_id),
            "exchangeSegment": exchange_segment,
            "instrument": instrument,
            "oi": bool(oi),
            "fromDate": from_date,
            "toDate": to_date,
        }
        if expiry_code is not None:
            payload["expiryCode"] = expiry_code
        return self._to_frame(self._post("/charts/historical", payload))

    @staticmethod
    def _to_frame(data: dict) -> pd.DataFrame:
        """Dhan returns parallel arrays, not records."""
        ts = data.get("timestamp") or []
        if not ts:
            return pd.DataFrame()

        df = pd.DataFrame(
            {
                "open": data.get("open", []),
                "high": data.get("high", []),
                "low": data.get("low", []),
                "close": data.get("close", []),
                "volume": data.get("volume", []),
                "open_interest": data.get("open_interest", [None] * len(ts)),
            }
        )
        # Dhan timestamps are epoch seconds.
        df.index = pd.to_datetime(ts, unit="s", utc=True).tz_convert(IST)
        df.index.name = "ts"
        return df

    # ------------------------------------------------------- option chain

    def option_chain(
        self, *, underlying_scrip: int, underlying_seg: str, expiry: str
    ) -> pd.DataFrame:
        """Option chain snapshot, flattened to long format.

        NOTE: this is a SNAPSHOT endpoint -- it returns the chain as of now,
        with no history. Intraday strike-level OI therefore has to be recorded
        live; it cannot be backfilled. That is why the collector matters.
        """
        data = self._post(
            "/optionchain",
            {
                "UnderlyingScrip": int(underlying_scrip),
                "UnderlyingSeg": underlying_seg,
                "Expiry": expiry,
            },
        )
        oc = (data.get("data") or {}).get("oc") or {}

        rows = []
        now = pd.Timestamp.now(tz=IST)
        for strike, legs in oc.items():
            for side, key in (("CE", "ce"), ("PE", "pe")):
                leg = legs.get(key)
                if not leg:
                    continue
                rows.append(
                    {
                        "ts": now,
                        "strike": float(strike),
                        "opt_type": side,
                        "ltp": leg.get("last_price"),
                        "oi": leg.get("oi"),
                        "volume": leg.get("volume"),
                        "iv": leg.get("implied_volatility"),
                        "delta": (leg.get("greeks") or {}).get("delta"),
                    }
                )
        return pd.DataFrame(rows)


def load_scrip_master() -> pd.DataFrame:
    """Download Dhan's instrument master -- maps symbols to securityId.

    Every historical and feed call needs a securityId, so this is the lookup
    table the whole pipeline depends on. Public CSV, no auth needed.
    """
    url = "https://images.dhan.co/api-data/api-scrip-master-detailed.csv"
    return pd.read_csv(url, low_memory=False)


def find_security_id(master: pd.DataFrame, *, contains: str, segment: str | None = None):
    """Locate an instrument in the scrip master.

    Column names in the master have changed between revisions, so this searches
    whatever symbol-like columns exist rather than assuming one.
    """
    cols = [c for c in master.columns if "SYMBOL" in c.upper() or "NAME" in c.upper()]
    mask = pd.Series(False, index=master.index)
    for c in cols:
        mask |= master[c].astype(str).str.upper().str.contains(contains.upper(), na=False)

    hits = master[mask]
    if segment:
        seg_cols = [c for c in master.columns if "SEGMENT" in c.upper() or "EXCH" in c.upper()]
        for c in seg_cols:
            sub = hits[hits[c].astype(str).str.upper().str.contains(segment.upper(), na=False)]
            if not sub.empty:
                hits = sub
                break
    return hits
