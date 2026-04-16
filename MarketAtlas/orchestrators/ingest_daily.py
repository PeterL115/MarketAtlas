from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, List

import pandas as pd

from ..core.config import AppConfig
from ..core.io import read_parquet, atomic_write_parquet, merge_dedupe_by_ts, last_timestamp
from ..core.validate import validate_bars
from ..core.providers.yfinance_provider import YFinanceProvider, YFPolicy


@dataclass(frozen=True)
class DailyIngestResult:
    ticker: str
    rows_before: int
    rows_fetched: int
    rows_after: int
    last_ts: Optional[str]
    status: str = "ok"
    error: Optional[str] = None


def _provider(cfg: AppConfig) -> YFinanceProvider:
    raw = cfg.raw.get("ingestion", {}).get("yfinance", {}) or {}
    pol = YFPolicy(
        max_attempts=int(raw.get("max_attempts", 6)),
        base_backoff_s=float(raw.get("base_backoff_s", 1.25)),
        jitter_s=float(raw.get("jitter_s", 0.75)),
        max_backoff_s=float(raw.get("max_backoff_s", 20.0)),
        min_interval_s=float(raw.get("min_interval_s", 0.2)),
        post_call_sleep_s=float(raw.get("post_call_sleep_s", 0.0)),
    )
    return YFinanceProvider(market_tz=cfg.market_tz, policy=pol)


def ingest_daily_one(
    cfg: AppConfig,
    ticker: str,
    provider_symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> DailyIngestResult:
    daily_dir = cfg.path("daily_dir")
    out_path = (daily_dir / f"{ticker}.parquet").resolve()

    existing = read_parquet(out_path)
    rows_before = int(len(existing)) if not existing.empty else 0

    eff_start = start
    if (eff_start is None) and (not existing.empty):
        lt = last_timestamp(existing)
        if lt is not None:
            eff_start = (lt - pd.Timedelta(days=7)).date().isoformat()

    if eff_start is None:
        eff_start = str(cfg.raw.get("ingestion", {}).get("daily", {}).get("default_start", "2015-01-01"))

    provider = _provider(cfg)

    try:
        new = provider.fetch_daily(provider_symbol, start=eff_start, end=end)
        if new.empty:
            lt0 = last_timestamp(existing)
            return DailyIngestResult(
                ticker=ticker,
                rows_before=rows_before,
                rows_fetched=0,
                rows_after=rows_before,
                last_ts=lt0.isoformat() if (lt0 is not None and rows_before) else None,
                status="ok",
                error=None,
            )

        validate_bars(new, require_volume=False)

        merged = merge_dedupe_by_ts(existing, new)
        atomic_write_parquet(merged, out_path)

        lt2 = last_timestamp(merged)
        return DailyIngestResult(
            ticker=ticker,
            rows_before=rows_before,
            rows_fetched=int(len(new)),
            rows_after=int(len(merged)),
            last_ts=lt2.isoformat() if lt2 is not None else None,
            status="ok",
            error=None,
        )

    except Exception as e:
        lt0 = last_timestamp(existing)
        return DailyIngestResult(
            ticker=ticker,
            rows_before=rows_before,
            rows_fetched=0,
            rows_after=rows_before,
            last_ts=lt0.isoformat() if lt0 is not None else None,
            status="error",
            error=str(e),
        )


def ingest_daily_all(
    cfg: AppConfig,
    start: Optional[str] = None,
    end: Optional[str] = None,
    tickers: Optional[List[str]] = None,
) -> Dict[str, DailyIngestResult]:
    results: Dict[str, DailyIngestResult] = {}

    items = list(cfg.daily_assets.items())
    if tickers:
        tickerset = set([t.strip().upper() for t in tickers if t and t.strip()])
        items = [(t, s) for (t, s) in items if t.strip().upper() in tickerset]

    for ticker, provider_symbol in items:
        results[ticker] = ingest_daily_one(cfg, ticker, provider_symbol, start=start, end=end)

    return results
