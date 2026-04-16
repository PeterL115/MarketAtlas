from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Iterable, List, Tuple

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


def _fmt_yf_dt(x: pd.Timestamp) -> str:
    """
    yfinance 对 start/end 字符串很挑剔：
    - 不要 'T'
    - 不要时区偏移 '-05:00'
    用 'YYYY-MM-DD HH:MM:SS'（naive）最稳。
    """
    if x.tzinfo is not None:
        x = x.tz_localize(None)
    return x.strftime("%Y-%m-%d %H:%M:%S")


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

    provider = YFinanceProvider(market_tz=cfg.market_tz)

    # 用 market_tz 的 “当日00:00 ~ 次日00:00” 作为 fetch 框架
    day0 = pd.Timestamp(day).tz_localize(cfg.market_tz)
    day1 = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize(cfg.market_tz)

    # 默认取整天（字符串格式不给 T / 不给 tz offset）
    start_str = _fmt_yf_dt(day0)
    end_str = _fmt_yf_dt(day1)

    # 如果已有数据：从 last_ts - 60min 追加抓（仍然用不带 tz 的字符串）
    if not existing.empty:
        lt = last_timestamp(existing, market_tz=cfg.market_tz)
        if lt is not None:
            # lt 可能已是 tz-aware；统一转为 market_tz，再去 tz 信息
            lt_mkt = pd.Timestamp(lt).tz_convert(cfg.market_tz) if pd.Timestamp(lt).tzinfo else pd.Timestamp(lt).tz_localize(cfg.market_tz)
            start_dt = lt_mkt - pd.Timedelta(minutes=60)
            # 仍然不要小于 day0，避免跨天
            if start_dt < day0:
                start_dt = day0
            start_str = _fmt_yf_dt(start_dt)

    new = provider.fetch_intraday_5m(
        provider_symbol,
        start=start_str,
        end=end_str,
        include_extended_hours=include_extended_hours,
    )

    if new.empty:
        lt0 = last_timestamp(existing, market_tz=cfg.market_tz)
        return IntradayIngestResult(ticker, day, rows_before, 0, rows_before, lt0.isoformat() if lt0 is not None else None)

    # --- Normalize to strict 5-minute grid ---
    new = new.copy()
    new["ts"] = pd.to_datetime(new["ts"], utc=True).dt.tz_convert(cfg.market_tz)
    new["ts"] = new["ts"].dt.floor("5min")
    new = new.sort_values("ts")
    new = new.drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)

    # Add session column + validate
    new = add_session_column(new, cfg.market_tz)
    validate_bars(new, require_volume=False)

    # Filter strictly to requested trading-day key (provider can leak adjacent bars)
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


def _resolve_intraday_universe(
    cfg: AppConfig, tickers: Optional[Iterable[str]] = None
) -> List[Tuple[str, str]]:
    """
    Returns list of (ticker, provider_symbol).
    If tickers provided, filter to those that exist in cfg.intraday_assets.
    """
    assets = list(cfg.intraday_assets.items())

    if tickers is None:
        return assets

    want = [str(t).strip() for t in tickers if str(t).strip()]
    have = dict(assets)
    out: List[Tuple[str, str]] = []
    for t in want:
        if t in have:
            out.append((t, have[t]))
    return out


def ingest_intraday_for_date(
    cfg: AppConfig,
    day: str,
    lookback_days: int = 0,
    include_extended_hours: Optional[bool] = None,
    tickers: Optional[Iterable[str]] = None,
) -> Dict[str, IntradayIngestResult]:
    if include_extended_hours is None:
        include_extended_hours = bool(cfg.raw.get("ingestion", {}).get("intraday_5m", {}).get("include_extended_hours", True))

    days = [pd.Timestamp(day) - pd.Timedelta(days=i) for i in range(lookback_days + 1)]
    day_keys = [d.strftime("%Y-%m-%d") for d in days]

    universe = _resolve_intraday_universe(cfg, tickers=tickers)

    results: Dict[str, IntradayIngestResult] = {}
    for ticker, provider_symbol in universe:
        for dk in reversed(day_keys):
            results[f"{ticker}:{dk}"] = ingest_intraday_one_day(
                cfg,
                ticker=ticker,
                provider_symbol=provider_symbol,
                day=dk,
                include_extended_hours=bool(include_extended_hours),
            )
    return results
