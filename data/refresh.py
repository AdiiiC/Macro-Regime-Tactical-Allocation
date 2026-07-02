"""
Live data refresh: pulls the most recent macro (FRED) and market (Yahoo
Finance) data up to *yesterday* and rewrites the parquet caches used by the
API, alongside the existing history.

Design goals
------------
- Real, current data. yfinance is queried with ``end = today`` so the last
  observation is the previous trading day; FRED returns the latest published
  print (macro series lag by weeks).
- Honest monthly series. Daily prices are resampled to month-end, and the
  *incomplete current month* is dropped so we never report a 1-day move as a
  monthly return.
- Safe. Each market is refreshed independently inside a try/except; on any
  failure the previous cache is left untouched.
- Observable. A ``_meta.json`` records, per market, when the refresh ran and
  the last date actually covered by macro features and market returns. The
  API surfaces this so the dashboard can show real data currency.

Run directly (``python -m data.refresh``) or import ``refresh_all`` / call it
from the API's ``POST /model/refresh?live=true``.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from datetime import datetime, timedelta
from typing import Dict, Optional

import pandas as pd

warnings.filterwarnings("ignore")

# Allow running as a script (python data/refresh.py) or module.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:  # dotenv is optional; the API loads it too.
    from dotenv import load_dotenv

    load_dotenv(os.path.join(_ROOT, ".env"))
except Exception:  # pragma: no cover
    pass

from config.settings import BACKTEST_START  # noqa: E402

CACHE_DIR = os.path.join(_ROOT, "data", "cache")
META_PATH = os.path.join(CACHE_DIR, "_meta.json")


# ─── helpers ────────────────────────────────────────────────────────────────
def _yesterday() -> str:
    return (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")


def _end_exclusive() -> str:
    """yfinance treats ``end`` as exclusive, so 'today' yields up to yesterday."""
    return datetime.now().strftime("%Y-%m-%d")


def _monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Month-end returns with the incomplete current month dropped."""
    monthly = prices.resample("ME").last()
    returns = monthly.pct_change().dropna(how="any")
    if len(returns):
        now = pd.Timestamp.now()
        last = returns.index[-1]
        # A month-end label in the current (not-yet-finished) month is partial.
        if last.year == now.year and last.month == now.month and now.day < 28:
            returns = returns.iloc[:-1]
    return returns


def _atomic_write_parquet(df: pd.DataFrame, path: str) -> None:
    tmp = f"{path}.tmp"
    df.to_parquet(tmp)
    os.replace(tmp, path)


def _through(df: Optional[pd.DataFrame]) -> Optional[str]:
    if df is None or len(df) == 0 or df.index.isna().all():
        return None
    return str(df.index.max())[:10]


def _load_meta() -> dict:
    if os.path.exists(META_PATH):
        try:
            with open(META_PATH) as fh:
                return json.load(fh)
        except Exception:
            return {}
    return {}


def _save_meta(meta: dict) -> None:
    tmp = f"{META_PATH}.tmp"
    with open(tmp, "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    os.replace(tmp, META_PATH)


# ─── per-market refresh ─────────────────────────────────────────────────────
def refresh_us(start: str = BACKTEST_START) -> Dict[str, object]:
    from data.fred_pipeline import MacroDataPipeline
    from data.market_data import MarketDataPipeline

    end = _end_exclusive()
    result: Dict[str, object] = {"market": "us", "ok": False}

    macro = MacroDataPipeline()
    macro.fetch_all_indicators(start=start, end=end)
    features = macro.get_model_ready_data()

    mkt = MarketDataPipeline()
    prices = mkt.fetch_prices(start=start, end=end)
    returns = _monthly_returns(prices)

    if features.empty or returns.empty:
        raise ValueError("US refresh produced empty features or returns")

    _atomic_write_parquet(features, os.path.join(CACHE_DIR, "macro_features.parquet"))
    _atomic_write_parquet(returns, os.path.join(CACHE_DIR, "market_returns.parquet"))

    result.update(
        ok=True,
        macro_through=_through(features),
        returns_through=_through(returns),
        prices_through=_through(prices),
        n_macro=len(features),
        n_returns=len(returns),
    )
    return result


def refresh_india(start: str = "2011-01-01") -> Dict[str, object]:
    from data.india_pipeline import IndiaDataPipeline
    from data.india_market import IndiaMarketDataPipeline

    end = _end_exclusive()
    result: Dict[str, object] = {"market": "india", "ok": False}

    macro = IndiaDataPipeline()
    macro.fetch_all_indicators(start=start, end=end)
    features = macro.get_model_ready_data()

    mkt = IndiaMarketDataPipeline()
    prices = mkt.fetch_prices(start=start, end=end)
    returns = _monthly_returns(prices)

    if features.empty:
        raise ValueError("India refresh produced empty macro features")

    _atomic_write_parquet(
        features, os.path.join(CACHE_DIR, "india_macro_features.parquet")
    )
    # Returns may still be short but should no longer be empty with a valid
    # gilt proxy; write whatever we got (the API tolerates missing returns).
    _atomic_write_parquet(
        returns, os.path.join(CACHE_DIR, "india_market_returns.parquet")
    )

    result.update(
        ok=True,
        macro_through=_through(features),
        returns_through=_through(returns),
        prices_through=_through(prices),
        n_macro=len(features),
        n_returns=len(returns),
    )
    return result


# ─── orchestration ──────────────────────────────────────────────────────────
def refresh_all(markets: Optional[list] = None) -> dict:
    """Refresh the given markets (default both). Never raises; per-market
    failures are captured so a partial refresh still updates ``_meta.json``."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    markets = markets or ["us", "india"]
    fns = {"us": refresh_us, "india": refresh_india}

    meta = _load_meta()
    now_iso = datetime.now().isoformat(timespec="seconds")
    summary = {"as_of": now_iso, "yesterday": _yesterday(), "markets": {}}

    for key in markets:
        fn = fns.get(key)
        if fn is None:
            continue
        try:
            res = fn()
            entry = {
                "refreshed_at": now_iso,
                "macro_through": res.get("macro_through"),
                "returns_through": res.get("returns_through"),
                "prices_through": res.get("prices_through"),
                "n_macro": res.get("n_macro"),
                "n_returns": res.get("n_returns"),
                "status": "ok",
            }
        except Exception as exc:  # keep old cache, record the failure
            entry = {
                "refreshed_at": now_iso,
                "status": "error",
                "error": repr(exc)[:300],
            }
        meta[key] = entry
        summary["markets"][key] = entry

    _save_meta(meta)
    return summary


def load_meta() -> dict:
    """Public accessor for the API."""
    return _load_meta()


if __name__ == "__main__":
    sel = sys.argv[1:] or None
    out = refresh_all(sel)
    print(json.dumps(out, indent=2))
