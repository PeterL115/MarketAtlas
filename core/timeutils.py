from __future__ import annotations

from datetime import datetime, time
from typing import Iterable

import pandas as pd
from dateutil import tz

from .constants import (
    SESSION_AFT,
    SESSION_CLOSED,
    SESSION_PRE,
    SESSION_RTH,
    AFT_END,
    PRE_START,
    RTH_END,
    RTH_START,
)


def ensure_tz(ts: pd.DatetimeIndex | pd.Series, market_tz: str) -> pd.DatetimeIndex:
    """
    Ensure timestamps are timezone-aware and converted to market_tz.
    """
    tzone = tz.gettz(market_tz)
    if tzone is None:
        raise ValueError(f"Invalid timezone: {market_tz}")

    idx = pd.DatetimeIndex(ts)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    return idx.tz_convert(tzone)


def label_session(ts: pd.Timestamp) -> str:
    """
    Label a single timestamp into PRE/RTH/AFT/CLOSED based on ET time of day.
    Assumes ts is already in market timezone.
    """
    tod = ts.timetz()
    # Convert to naive time for comparison
    t_ = time(tod.hour, tod.minute, tod.second)

    pre_start = time(*PRE_START)
    rth_start = time(RTH_START[0], RTH_START[1])
    rth_end = time(*RTH_END)
    aft_end = time(*AFT_END)

    if pre_start <= t_ < rth_start:
        return SESSION_PRE
    if rth_start <= t_ < rth_end:
        return SESSION_RTH
    if rth_end <= t_ < aft_end:
        return SESSION_AFT
    return SESSION_CLOSED


def add_session_column(df: pd.DataFrame, market_tz: str) -> pd.DataFrame:
    if "ts" not in df.columns:
        raise KeyError("Expected column 'ts' in DataFrame")

    ts_idx = ensure_tz(df["ts"], market_tz)
    df = df.copy()
    df["ts"] = ts_idx
    df["session"] = [label_session(t) for t in df["ts"]]
    return df


def trading_day_key(ts: pd.Timestamp) -> str:
    """
    Return YYYY-MM-DD based on the timestamp's local date in market timezone.
    For US equities, PRE/RTH/AFT all map to that same calendar date.
    """
    return ts.date().isoformat()
