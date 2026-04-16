from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from MarketAtlas.core.config import AppConfig
from MarketAtlas.core.io import read_parquet


def _sf(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        return None if pd.isna(v) else v
    except Exception:
        return None


def compute_pivot_levels(high: float, low: float, close: float) -> Dict[str, Any]:
    """
    Compute CPR (Central Pivot Range) and Classic Daily Pivot Levels from
    prior-day High, Low, Close.

    CPR:
        Pivot = (H + L + C) / 3
        BC    = (H + L) / 2          (Bottom Central)
        TC    = 2 * Pivot - BC       (Top Central)

    Classic (R3/R2/R1/Pivot/S1/S2/S3):
        R1 = 2 * Pivot - L
        R2 = Pivot + (H - L)
        R3 = H + 2 * (Pivot - L)
        S1 = 2 * Pivot - H
        S2 = Pivot - (H - L)
        S3 = L - 2 * (H - Pivot)
    """
    h, l, c = float(high), float(low), float(close)
    pivot = (h + l + c) / 3.0
    bc    = (h + l) / 2.0
    tc    = 2.0 * pivot - bc

    r1 = 2.0 * pivot - l
    r2 = pivot + (h - l)
    r3 = h + 2.0 * (pivot - l)

    s1 = 2.0 * pivot - h
    s2 = pivot - (h - l)
    s3 = l - 2.0 * (h - pivot)

    return {
        "cpr": {
            "pivot": round(pivot, 2),
            "bc":    round(bc, 2),
            "tc":    round(tc, 2),
        },
        "classic": {
            "r3":    round(r3, 2),
            "r2":    round(r2, 2),
            "r1":    round(r1, 2),
            "pivot": round(pivot, 2),
            "s1":    round(s1, 2),
            "s2":    round(s2, 2),
            "s3":    round(s3, 2),
        },
        "inputs": {
            "high":  round(h, 4),
            "low":   round(l, 4),
            "close": round(c, 4),
        },
    }


def _normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a daily OHLC parquet into standard ts/open/high/low/close columns."""
    if df is None or df.empty:
        return pd.DataFrame()

    idx_is_dt = isinstance(df.index, pd.DatetimeIndex) or (
        hasattr(df.index, "dtype") and np.issubdtype(df.index.dtype, np.datetime64)
    )
    if idx_is_dt and "ts" not in df.columns:
        df = df.reset_index().rename(columns={df.index.name or "index": "ts"})

    d = df.copy()
    d.columns = [str(c).strip().lower() for c in d.columns]

    if "ts" not in d.columns:
        for cand in ("date", "datetime", "timestamp", "time"):
            if cand in d.columns:
                d = d.rename(columns={cand: "ts"})
                break

    if "close" not in d.columns:
        for alt in ("adj close", "adj_close", "adjclose"):
            if alt in d.columns:
                d = d.rename(columns={alt: "close"})
                break

    required = {"ts", "high", "low", "close"}
    if not required.issubset(set(d.columns)):
        return pd.DataFrame()

    d["ts"] = pd.to_datetime(d["ts"], errors="coerce")
    d = d.dropna(subset=["ts"])
    d["day"] = d["ts"].dt.date.astype(str)
    return d


def read_pivot_levels_for_day(cfg: AppConfig, ticker: str, asof_day: str) -> Optional[Dict[str, Any]]:
    """
    Read the daily parquet for `ticker`, find the row matching `asof_day`
    (or the last available row on or before that date), then compute and
    return pivot levels.  Returns None if data is unavailable.
    """
    try:
        daily_path = cfg.path("daily_dir") / f"{ticker}.parquet"
        df_raw = read_parquet(daily_path)
        if df_raw is None or df_raw.empty:
            return None

        df = _normalize_ohlc(df_raw)
        if df.empty:
            return None

        df = df[df["day"] <= str(asof_day)].sort_values("day")
        if df.empty:
            return None

        row = df.iloc[-1]
        h = _sf(row.get("high"))
        l = _sf(row.get("low"))
        c = _sf(row.get("close"))
        if h is None or l is None or c is None:
            return None

        levels = compute_pivot_levels(h, l, c)
        levels["date"] = str(row.get("day", asof_day))
        return levels

    except Exception:
        return None
