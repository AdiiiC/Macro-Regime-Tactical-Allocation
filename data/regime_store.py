"""
Persistent regime-history store (SQLite, stdlib only).

Every regime read the API produces is worth keeping: the full historical
monthly series (backfilled once from the fitted detector) plus a running log of
each live refresh's latest read. This lets the frontend show how the regime call
has evolved and when it last flipped, without recomputing from parquet each time.

Design notes
------------
* One table, ``regime_reads``. A read is uniquely identified by
  ``(market, as_of, source)`` so backfills and repeated live logs are
  idempotent (``INSERT OR REPLACE``).
* ``source`` distinguishes provenance: ``backfill`` (historical monthly series),
  ``live`` (logged when the live refresh runs) or ``startup``.
* Pure stdlib ``sqlite3`` — no new dependencies, no ORM.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Iterable, List, Optional

DB_PATH = os.path.join(os.path.dirname(__file__), "regime_history.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS regime_reads (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    market      TEXT    NOT NULL,
    as_of       TEXT    NOT NULL,          -- observation date (YYYY-MM-DD)
    regime      TEXT    NOT NULL,
    confidence  REAL,
    source      TEXT    NOT NULL DEFAULT 'backfill',
    recorded_at TEXT    NOT NULL,          -- when this row was written (ISO)
    UNIQUE (market, as_of, source)
);
CREATE INDEX IF NOT EXISTS idx_regime_reads_market_asof
    ON regime_reads (market, as_of);
"""


@contextmanager
def _connect(db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def backfill_series(
    market: str,
    regimes,  # pd.Series indexed by date, values = regime name
    confidences=None,  # optional pd.Series aligned to regimes
    db_path: str = DB_PATH,
) -> int:
    """
    Persist a full monthly regime series as ``source='backfill'``.

    Idempotent: re-running replaces existing backfill rows for the same
    ``(market, as_of)``. Returns the number of rows written.
    """
    now = datetime.now().isoformat()
    rows: List[tuple] = []
    for ts, regime in regimes.items():
        as_of = str(getattr(ts, "date", lambda: ts)())[:10]
        conf = None
        if confidences is not None and ts in confidences.index:
            try:
                conf = float(confidences.loc[ts])
            except (TypeError, ValueError):
                conf = None
        rows.append((market, as_of, str(regime), conf, "backfill", now))

    if not rows:
        return 0

    with _connect(db_path) as conn:
        conn.executemany(
            """INSERT OR REPLACE INTO regime_reads
               (market, as_of, regime, confidence, source, recorded_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            rows,
        )
    return len(rows)


def log_read(
    market: str,
    as_of: str,
    regime: str,
    confidence: Optional[float],
    source: str = "live",
    db_path: str = DB_PATH,
) -> None:
    """Record a single latest regime read (e.g. after a live refresh)."""
    with _connect(db_path) as conn:
        conn.execute(
            """INSERT OR REPLACE INTO regime_reads
               (market, as_of, regime, confidence, source, recorded_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (market, str(as_of)[:10], str(regime),
             float(confidence) if confidence is not None else None,
             source, datetime.now().isoformat()),
        )


def history(
    market: str,
    limit: int = 120,
    db_path: str = DB_PATH,
) -> List[dict]:
    """
    Return the merged regime timeline for a market, newest first.

    Live/startup reads take precedence over the backfilled monthly point for the
    same ``as_of`` date (dedup keeps the most recently recorded row).
    """
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            SELECT market, as_of, regime, confidence, source, recorded_at
            FROM regime_reads
            WHERE market = ?
            ORDER BY as_of DESC, recorded_at DESC
            """,
            (market,),
        )
        seen: set = set()
        out: List[dict] = []
        for r in cur:
            if r["as_of"] in seen:
                continue
            seen.add(r["as_of"])
            out.append({
                "as_of": r["as_of"],
                "regime": r["regime"],
                "confidence": r["confidence"],
                "source": r["source"],
                "recorded_at": r["recorded_at"],
            })
            if len(out) >= limit:
                break
    return out


def transitions(market: str, db_path: str = DB_PATH) -> List[dict]:
    """Return only the dates where the regime label changed, oldest first."""
    rows = sorted(history(market, limit=10_000, db_path=db_path),
                  key=lambda d: d["as_of"])
    out: List[dict] = []
    prev: Optional[str] = None
    for row in rows:
        if row["regime"] != prev:
            out.append({
                "as_of": row["as_of"],
                "from": prev,
                "to": row["regime"],
                "confidence": row["confidence"],
            })
            prev = row["regime"]
    return out


def stats(db_path: str = DB_PATH) -> dict:
    """Lightweight summary for health/introspection."""
    with _connect(db_path) as conn:
        cur = conn.execute(
            "SELECT market, COUNT(*) AS n, MAX(as_of) AS latest "
            "FROM regime_reads GROUP BY market"
        )
        return {r["market"]: {"rows": r["n"], "latest": r["latest"]} for r in cur}
