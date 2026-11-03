"""Buffered storage for collected ticks.

Design rule: never do network I/O on the receive path. The websocket handler
appends to an in-memory list and returns immediately. A separate task flushes
to local disk, and a slower one uploads to GCS.

If a flush or an upload fails, the collector keeps recording. Losing an upload
is recoverable; losing ticks is not.

Layout:
    data/live/dt=YYYY-MM-DD/ticks_HHMMSS.parquet
    data/live/dt=YYYY-MM-DD/universe.parquet
    data/live/dt=YYYY-MM-DD/chain_HHMMSS.parquet

Hive-style dt= partitioning so BigQuery and pandas can both read it directly.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

log = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")


class TickWriter:
    """Thread-safe buffer that flushes to partitioned parquet."""

    def __init__(self, root: str = "data/live", flush_rows: int = 50_000):
        self.root = Path(root)
        self.flush_rows = flush_rows
        self._buf: list[dict] = []
        self._lock = threading.Lock()
        self.total_rows = 0
        self.files_written = 0

    def append(self, row: dict) -> None:
        with self._lock:
            self._buf.append(row)

    def extend(self, rows: list[dict]) -> None:
        if not rows:
            return
        with self._lock:
            self._buf.extend(rows)

    @property
    def pending(self) -> int:
        with self._lock:
            return len(self._buf)

    def _partition_dir(self, when: datetime) -> Path:
        d = self.root / f"dt={when:%Y-%m-%d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def flush(self, prefix: str = "ticks") -> Path | None:
        """Write the buffer to parquet and clear it. Returns the path written."""
        with self._lock:
            if not self._buf:
                return None
            rows, self._buf = self._buf, []

        now = datetime.now(IST)
        path = self._partition_dir(now) / f"{prefix}_{now:%H%M%S}.parquet"

        try:
            pd.DataFrame(rows).to_parquet(path, compression="snappy", index=False)
        except Exception:
            # Put the rows back rather than dropping them -- a failed write is
            # not a reason to lose data that cannot be re-recorded.
            log.exception("flush failed, returning %d rows to buffer", len(rows))
            with self._lock:
                self._buf = rows + self._buf
            return None

        self.total_rows += len(rows)
        self.files_written += 1
        log.info("flushed %d rows -> %s", len(rows), path.name)
        return path

    def write_frame(self, df: pd.DataFrame, name: str) -> Path | None:
        """Write a one-off frame (universe metadata, an option-chain snapshot)."""
        if df.empty:
            return None
        now = datetime.now(IST)
        path = self._partition_dir(now) / f"{name}.parquet"
        try:
            df.to_parquet(path, compression="snappy", index=False)
            return path
        except Exception:
            log.exception("failed writing %s", name)
            return None


class GCSUploader:
    """Mirrors local parquet to GCS. Optional -- absent config disables it.

    Upload failures are logged and retried on the next pass. They never block
    or stop collection.
    """

    def __init__(self, bucket: str | None, prefix: str = "mcube/live"):
        self.bucket_name = bucket
        self.prefix = prefix
        self._client = None
        self._bucket = None
        self._uploaded: set[str] = set()

        if bucket:
            try:
                from google.cloud import storage

                self._client = storage.Client()
                self._bucket = self._client.bucket(bucket)
                log.info("GCS upload enabled -> gs://%s/%s", bucket, prefix)
            except Exception as e:
                log.warning("GCS disabled (%s). Recording continues locally.", e)
                self._bucket = None
        else:
            log.info("GCS not configured -- local storage only")

    @property
    def enabled(self) -> bool:
        return self._bucket is not None

    def sync(self, root: str = "data/live") -> int:
        """Upload any local parquet not yet sent. Returns count uploaded."""
        if not self.enabled:
            return 0

        n = 0
        for path in Path(root).rglob("*.parquet"):
            key = str(path.relative_to(root)).replace("\\", "/")
            if key in self._uploaded:
                continue
            try:
                blob = self._bucket.blob(f"{self.prefix}/{key}")
                blob.upload_from_filename(str(path))
                self._uploaded.add(key)
                n += 1
            except Exception:
                log.exception("upload failed for %s (will retry)", key)
        if n:
            log.info("uploaded %d file(s) to GCS", n)
        return n


class GapMonitor:
    """Tracks per-instrument silence to detect a stalled feed.

    A websocket that is open but delivering nothing looks healthy to the
    connection layer and is the failure mode most likely to cost a day of data.
    Illiquid far strikes are legitimately quiet, so this only alerts when a
    large share of the universe goes silent at once.
    """

    def __init__(self, stale_seconds: float = 60.0):
        self.stale_seconds = stale_seconds
        self._last_seen: dict[int, float] = {}
        self._lock = threading.Lock()

    def mark(self, security_id: int, when: float) -> None:
        with self._lock:
            self._last_seen[security_id] = when

    def stale(self, now: float) -> list[int]:
        with self._lock:
            return [
                sid for sid, t in self._last_seen.items()
                if now - t > self.stale_seconds
            ]

    def report(self, now: float) -> dict:
        with self._lock:
            total = len(self._last_seen)
            stale = sum(1 for t in self._last_seen.values() if now - t > self.stale_seconds)
        return {
            "tracked": total,
            "stale": stale,
            "stale_pct": round(100 * stale / total, 1) if total else 0.0,
        }
