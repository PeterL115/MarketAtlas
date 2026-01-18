from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from ..core.config import AppConfig
from ..core.io import read_parquet, atomic_write_parquet, merge_dedupe_by_ts, last_timestamp
from ..core.timeutils import add_session_column
from ..core.validate import validate_bars
from ..core.providers.yfinance_provider import YFinanceProvider


@dataclass(frozen=True)
class DailyIngestResult:
    ticker: str
    rows_before: int
    rows_fetched: int
    rows_after: int
    last_ts: Optional[str]


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

    # Incremental fetch: if existing has data, start from last_ts (minus a small buffer)
    eff_start = start
    if (eff_start is None) and (not existing.empty):
        lt = last_timestamp(existing)
        if lt is not None:
            # Pull a few extra days to be safe around corporate actions/provider fixes
            eff_start = (lt - pd.Timedelta(days=7)).date().isoformat()

    if eff_start is None:
        eff_start = str(cfg.raw.get("ingestion", {}).get("daily", {}).get("default_start", "2015-01-01"))

    provider = YFinanceProvider(market_tz=cfg.market_tz)
    new = provider.fetch_daily(provider_symbol, start=eff_start, end=end)
    if new.empty:
        return DailyIngestResult(ticker, rows_before, 0, rows_before, last_timestamp(existing).isoformat() if rows_before else None)

    # Daily bars: session is not used, but keep schema consistent (CLOSED for daily)
    # We'll omit session assignment for daily unless you want it; keep canonical minimal.
    report = validate_bars(new, require_volume=False)

    merged = merge_dedupe_by_ts(existing, new)
    atomic_write_parquet(merged, out_path)

    lt2 = last_timestamp(merged)
    return DailyIngestResult(
        ticker=ticker,
        rows_before=rows_before,
        rows_fetched=int(len(new)),
        rows_after=int(len(merged)),
        last_ts=lt2.isoformat() if lt2 is not None else None,
    )


def ingest_daily_all(cfg: AppConfig, start: Optional[str] = None, end: Optional[str] = None) -> Dict[str, DailyIngestResult]:
    results: Dict[str, DailyIngestResult] = {}
    for ticker, provider_symbol in cfg.daily_assets.items():
        results[ticker] = ingest_daily_one(cfg, ticker, provider_symbol, start=start, end=end)
    return results
