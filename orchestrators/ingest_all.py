from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

from ..core.config import AppConfig
from .ingest_daily import ingest_daily_all
from .ingest_intraday_5m import ingest_intraday_for_date


@dataclass(frozen=True)
class IngestAllResult:
    daily: Dict[str, Any]
    intraday: Dict[str, Any]


def ingest_all(
    cfg: AppConfig,
    daily_start: Optional[str] = None,
    daily_end: Optional[str] = None,
    intraday_day: Optional[str] = None,
    intraday_lookback_days: Optional[int] = None,
) -> IngestAllResult:
    daily_res = ingest_daily_all(cfg, start=daily_start, end=daily_end)

    if intraday_day is None:
        # default to "today" in market timezone via config; simplest: use local date string
        # (Later we can improve using tz-aware now)
        from datetime import date
        intraday_day = date.today().isoformat()

    if intraday_lookback_days is None:
        intraday_lookback_days = int(cfg.raw.get("ingestion", {}).get("intraday_5m", {}).get("default_lookback_days", 5))

    intraday_res = ingest_intraday_for_date(cfg, day=intraday_day, lookback_days=intraday_lookback_days)

    return IngestAllResult(
        daily={k: vars(v) for k, v in daily_res.items()},
        intraday={k: vars(v) for k, v in intraday_res.items()},
    )
