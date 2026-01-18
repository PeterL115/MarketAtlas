from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from ..core.config import AppConfig
from ..core.io import read_parquet, atomic_write_parquet, merge_dedupe_by_ts, last_timestamp, ensure_dir
from ..core.timeutils import add_session_column, trading_day_key
from ..core.validate import validate_bars
from ..core.providers.yfinance_provider import YFinanceProvider


@dataclass(frozen=True)
class IntradayIngestResult:
    ticker: str
    day: str
    rows_before: int
    rows_fetched: int
    rows_after: int
    last_ts: Optional[str]


def _day_path(cfg: AppConfig, ticker: str, day: str) -> Path:
    base = cfg.path("intraday_dir") / ticker
    return (base / f"{day}.parquet").resolve()


def ingest_intraday_one_day(
    cfg: AppConfig,
    ticker: str,
    provider_symbol: str,
    day: str,
    include_extended_hours: bool = True,
) -> IntradayIngestResult:
    out_path = _day_path(cfg, ticker, day)
    ensure_dir(out_path.parent)

    existing = read_parquet(out_path)
    rows_before = int(len(existing)) if not existing.empty else 0

    # Determine fetch window:
    # If existing exists, refetch from last_ts - 60 minutes buffer.
    # If not, fetch the entire day (and allow provider to return only available bars).
    provider = YFinanceProvider(market_tz=cfg.market_tz)

    # Fetch the whole day using DATE strings (yfinance is strict on parsing).
    start = day  # "YYYY-MM-DD"
    end = (pd.Timestamp(day) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


    if not existing.empty:
        lt = last_timestamp(existing)
        if lt is not None:
            start_dt = (lt - pd.Timedelta(minutes=60)).to_pydatetime()
            start = start_dt.isoformat()
    else:
        # Start at 00:00 local for that date; provider will return what's available.
        start = f"{day}T00:00:00"
    # End at next day 00:00 to bracket the date
    end = (pd.Timestamp(day) + pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00:00")

    new = provider.fetch_intraday_5m(
        provider_symbol,
        start=start,
        end=end,
        include_extended_hours=include_extended_hours,
    )

    if new.empty:
        lt0 = last_timestamp(existing)
        return IntradayIngestResult(ticker, day, rows_before, 0, rows_before, lt0.isoformat() if lt0 is not None else None)

    # --- Normalize to strict 5-minute grid (avoid odd stamps like 13:02) ---
    new = new.copy()
    new["ts"] = pd.to_datetime(new["ts"], utc=True).dt.tz_convert(cfg.market_tz)
    new["ts"] = new["ts"].dt.floor("5min")
    new = new.sort_values("ts")
    new = new.drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)

    # Add session column + validate
    new = add_session_column(new, cfg.market_tz)
    validate_bars(new, require_volume=False)

    # Filter strictly to the requested trading-day key (provider can leak adjacent bars)
    new = new[new["ts"].apply(lambda x: trading_day_key(x) == day)].copy()

    merged = merge_dedupe_by_ts(existing, new)
    atomic_write_parquet(merged, out_path)

    lt2 = last_timestamp(merged, market_tz=cfg.market_tz)

    return IntradayIngestResult(
        ticker=ticker,
        day=day,
        rows_before=rows_before,
        rows_fetched=int(len(new)),
        rows_after=int(len(merged)),
        last_ts=lt2.isoformat() if lt2 is not None else None,
    )


def ingest_intraday_for_date(
    cfg: AppConfig,
    day: str,
    lookback_days: int = 0,
    include_extended_hours: Optional[bool] = None,
) -> Dict[str, IntradayIngestResult]:
    if include_extended_hours is None:
        include_extended_hours = bool(cfg.raw.get("ingestion", {}).get("intraday_5m", {}).get("include_extended_hours", True))

    days = [pd.Timestamp(day) - pd.Timedelta(days=i) for i in range(lookback_days + 1)]
    day_keys = [d.strftime("%Y-%m-%d") for d in days]

    results: Dict[str, IntradayIngestResult] = {}
    for ticker, provider_symbol in cfg.intraday_assets.items():
        for dk in reversed(day_keys):
            results[f"{ticker}:{dk}"] = ingest_intraday_one_day(
                cfg, ticker, provider_symbol, dk, include_extended_hours=include_extended_hours
            )
    return results
