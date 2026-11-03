"""DhanHQ live market feed websocket client.

Endpoint:
    wss://api-feed.dhan.co?version=2&token=...&clientId=...&authType=2

Limits (DhanHQ v2):
    5 websocket connections per user
    5,000 instruments per connection
    100 instruments per subscribe message  -> this client chunks automatically

Reconnection matters more than throughput here. A broker feed will drop
several times a session, and a dropped connection that does not come back is
a lost day of data that cannot be re-recorded. So: exponential backoff, full
re-subscribe on every reconnect, and a token refresh if the drop looks like an
auth failure (the token expires every 24h).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field

from mcube.collector.instruments import Instrument
from mcube.collector.packets import parse_stream, subscribe_message

log = logging.getLogger(__name__)

FEED_URL = "wss://api-feed.dhan.co"
MAX_PER_SUBSCRIBE = 100
MAX_PER_CONNECTION = 5000

REQUEST_FULL = 21        # quote + 5-level depth
REQUEST_DISCONNECT = 12


@dataclass
class FeedStats:
    packets: int = 0
    bytes_in: int = 0
    reconnects: int = 0
    last_packet_at: float = 0.0
    connected_since: float = 0.0
    errors: list[str] = field(default_factory=list)


class LiveFeed:
    """One websocket connection carrying many instruments.

    on_ticks(rows) is called with a list of flattened dicts. It must be fast
    and must not block -- append to a buffer and return.
    """

    def __init__(
        self,
        *,
        instruments: list[Instrument],
        token_provider,
        client_id: str,
        on_ticks,
        request_code: int = REQUEST_FULL,
        gap_monitor=None,
    ):
        if len(instruments) > MAX_PER_CONNECTION:
            raise ValueError(
                f"{len(instruments)} instruments exceeds the {MAX_PER_CONNECTION} "
                "per-connection limit; split across connections."
            )
        self.instruments = instruments
        self.token_provider = token_provider
        self.client_id = client_id
        self.on_ticks = on_ticks
        self.request_code = request_code
        self.gap_monitor = gap_monitor
        self.stats = FeedStats()
        self._stop = asyncio.Event()

    def _url(self) -> str:
        token = self.token_provider()
        return (
            f"{FEED_URL}?version=2&token={token}"
            f"&clientId={self.client_id}&authType=2"
        )

    async def _subscribe(self, ws) -> None:
        pairs = [(i.segment, i.security_id) for i in self.instruments]
        for start in range(0, len(pairs), MAX_PER_SUBSCRIBE):
            chunk = pairs[start : start + MAX_PER_SUBSCRIBE]
            await ws.send(json.dumps(subscribe_message(chunk, self.request_code)))
            # Brief pause between chunks; bursting all at once can get the
            # connection dropped.
            await asyncio.sleep(0.15)
        log.info("subscribed %d instruments in %d chunk(s)",
                 len(pairs), (len(pairs) + MAX_PER_SUBSCRIBE - 1) // MAX_PER_SUBSCRIBE)

    async def run(self) -> None:
        """Connect, subscribe, read until stopped. Reconnects on failure."""
        try:
            import websockets
        except ImportError:
            raise RuntimeError("pip install websockets")

        backoff = 1.0
        while not self._stop.is_set():
            try:
                async with websockets.connect(
                    self._url(), ping_interval=20, ping_timeout=20, max_size=None
                ) as ws:
                    self.stats.connected_since = time.time()
                    backoff = 1.0  # reset only after a successful connect
                    log.info("feed connected")
                    await self._subscribe(ws)

                    async for message in ws:
                        if self._stop.is_set():
                            break
                        if isinstance(message, str):
                            log.debug("text frame: %s", message[:200])
                            continue
                        self._handle(message)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.stats.reconnects += 1
                self.stats.errors.append(f"{type(e).__name__}: {e}")
                self.stats.errors = self.stats.errors[-20:]
                log.warning("feed dropped (%s). reconnect in %.0fs", e, backoff)

                # An auth-shaped failure usually means the 24h token expired.
                if any(k in str(e).lower() for k in ("401", "403", "unauthor", "token")):
                    log.info("auth failure -- forcing token refresh")
                    try:
                        self.token_provider(force=True)
                    except TypeError:
                        pass

                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

    def _handle(self, raw: bytes) -> None:
        now = time.time()
        recv_ns = time.time_ns()

        rows = []
        for tick in parse_stream(raw):
            rows.append(tick.to_row(recv_ns))
            if self.gap_monitor is not None:
                self.gap_monitor.mark(tick.security_id, now)

        if rows:
            self.stats.packets += len(rows)
            self.stats.bytes_in += len(raw)
            self.stats.last_packet_at = now
            self.on_ticks(rows)

    def stop(self) -> None:
        self._stop.set()
