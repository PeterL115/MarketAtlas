from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from .config import AppConfig
from .io import read_parquet
from .jsonio import atomic_write_json, read_json
from .timeutils import ensure_tz
from .constants import SESSION_PRE, SESSION_RTH, SESSION_AFT




def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if pd.isna(v):
            return None
        return v
    except Exception:
        return None


def _typical_price(df: pd.DataFrame) -> pd.Series:
    # Typical price is more stable for bar-level VWAP than using "close" only.
    return (df["high"] + df["low"] + df["close"]) / 3.0


def _vwap_update(sum_pv: float, sum_v: float, df: pd.DataFrame) -> Tuple[float, float, Optional[float]]:
    """
    Update running VWAP accumulators using typical price * volume.
    Returns (new_sum_pv, new_sum_v, vwap_value).
    """
    if df.empty:
        vwap = (sum_pv / sum_v) if sum_v > 0 else None
        return sum_pv, sum_v, vwap

    tp = _typical_price(df)
    vol = df["volume"].fillna(0.0)

    add_sum_v = float(vol.sum())
    add_sum_pv = float((tp * vol).sum())

    sum_v2 = sum_v + add_sum_v
    sum_pv2 = sum_pv + add_sum_pv

    vwap = (sum_pv2 / sum_v2) if sum_v2 > 0 else None
    return sum_pv2, sum_v2, vwap


def _opening_range_window(day: str, market_tz: str, minutes: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Opening range window in market time zone: [09:30, 09:30+minutes)
    """
    start = pd.Timestamp(f"{day} 09:30:00").tz_localize(market_tz)
    end = start + pd.Timedelta(minutes=minutes)
    return start, end


def _state_paths(cfg: AppConfig, ticker: str, day: str) -> Tuple[Path, Path]:
    latest = cfg.path("session_state_dir") / f"{ticker}.json"
    snap = cfg.path("session_state_dir") / ticker / f"{day}.json"
    return latest.resolve(), snap.resolve()


def _intraday_day_path(cfg: AppConfig, ticker: str, day: str) -> Path:
    return (cfg.path("intraday_dir") / ticker / f"{day}.parquet").resolve()


def _init_state(ticker: str, day: str, market_tz: str, opening_range_minutes: int) -> Dict[str, Any]:
    return {
        "ticker": ticker,
        "day": day,
        "market_tz": market_tz,
        "last_update_ts": None,  # ISO8601
        "status": "init",
        "opening_range": {
            "minutes": opening_range_minutes,
            "start_ts": None,
            "end_ts": None,
            "high": None,
            "low": None,
            "complete": False,
        },
        "sessions": {
            SESSION_PRE: {
                "high": None, "low": None, "close": None,
                "bar_count": 0,
                "first_ts": None, "last_ts": None,
                "vwap": {"sum_pv": 0.0, "sum_v": 0.0, "value": None},
            },
            SESSION_RTH: {
                "high": None, "low": None, "close": None,
                "bar_count": 0,
                "first_ts": None, "last_ts": None,
                "vwap": {"sum_pv": 0.0, "sum_v": 0.0, "value": None},
            },
            SESSION_AFT: {
                "high": None, "low": None, "close": None,
                "bar_count": 0,
                "first_ts": None, "last_ts": None,
                "vwap": {"sum_pv": 0.0, "sum_v": 0.0, "value": None},
            },
        },
    }


def _update_session_agg(sess: Dict[str, Any], df: pd.DataFrame, compute_vwap: bool) -> None:
    if df.empty:
        return

    # highs/lows
    hi = _safe_float(df["high"].max())
    lo = _safe_float(df["low"].min())
    if hi is not None:
        sess["high"] = hi if sess["high"] is None else max(sess["high"], hi)
    if lo is not None:
        sess["low"] = lo if sess["low"] is None else min(sess["low"], lo)

    # close/ts
    df_sorted = df.sort_values("ts")
    last_row = df_sorted.iloc[-1]
    sess["close"] = _safe_float(last_row["close"])
    sess["bar_count"] = int(sess["bar_count"]) + int(len(df_sorted))

    first_ts = df_sorted.iloc[0]["ts"]
    last_ts = df_sorted.iloc[-1]["ts"]
    if sess["first_ts"] is None:
        sess["first_ts"] = pd.Timestamp(first_ts).isoformat()
    sess["last_ts"] = pd.Timestamp(last_ts).isoformat()

    # VWAP
    if compute_vwap:
        vwap_obj = sess["vwap"]
        sum_pv = float(vwap_obj.get("sum_pv", 0.0))
        sum_v = float(vwap_obj.get("sum_v", 0.0))
        sum_pv2, sum_v2, vwap_val = _vwap_update(sum_pv, sum_v, df_sorted)
        vwap_obj["sum_pv"] = sum_pv2
        vwap_obj["sum_v"] = sum_v2
        vwap_obj["value"] = _safe_float(vwap_val)


def _update_opening_range(state: Dict[str, Any], df_rth: pd.DataFrame, day: str, market_tz: str) -> None:
    """
    Update opening range high/low if not complete.
    Uses RTH bars only.
    """
    or_minutes = int(state["opening_range"]["minutes"])
    or_start, or_end = _opening_range_window(day, market_tz, or_minutes)

    state["opening_range"]["start_ts"] = or_start.isoformat()
    state["opening_range"]["end_ts"] = or_end.isoformat()

    if state["opening_range"]["complete"]:
        return

    if df_rth.empty:
        return

    # ensure tz
    ts = pd.DatetimeIndex(df_rth["ts"])
    # bars within opening range
    win = df_rth[(df_rth["ts"] >= or_start) & (df_rth["ts"] < or_end)].copy()
    if win.empty:
        return

    hi = _safe_float(win["high"].max())
    lo = _safe_float(win["low"].min())
    if hi is not None:
        state["opening_range"]["high"] = hi if state["opening_range"]["high"] is None else max(state["opening_range"]["high"], hi)
    if lo is not None:
        state["opening_range"]["low"] = lo if state["opening_range"]["low"] is None else min(state["opening_range"]["low"], lo)

    # Determine completeness: if we have any bar with ts >= (or_end - bar_interval)
    # With 5-minute bars, the last bar that starts before or_end can be enough.
    # We will mark complete once we have at least one bar at or after (or_end - 5min).
    if (win["ts"].max() >= (or_end - pd.Timedelta(minutes=5))):
        state["opening_range"]["complete"] = True


def update_session_state(cfg: AppConfig, ticker: str, day: str) -> Dict[str, Any]:
    """
    Incremental session state updater:
    - Reads existing latest state (if exists) to get last_update_ts and VWAP accumulators
    - Reads intraday parquet for that day
    - Processes only new bars (ts > last_update_ts)
    - Writes latest + snapshot JSON
    """
    market_tz = cfg.market_tz
    opening_range_minutes = int(cfg.raw.get("session_state", {}).get("opening_range_minutes", 30))
    compute_vwap_sessions = set(cfg.raw.get("session_state", {}).get("compute_vwap_sessions", [SESSION_RTH]))

    latest_path, snap_path = _state_paths(cfg, ticker, day)
    existing = read_json(latest_path)

    if not existing or existing.get("ticker") != ticker or existing.get("day") != day:
        state = _init_state(ticker, day, market_tz, opening_range_minutes)
    else:
        state = existing

    # Load intraday bars for day
    in_path = _intraday_day_path(cfg, ticker, day)
    bars = read_parquet(in_path)
    if bars is None or bars.empty:
        state["status"] = "no_data"
        atomic_write_json(state, latest_path)
        atomic_write_json(state, snap_path)
        return state

    # Ensure timestamp tz in market tz
    bars = bars.copy()
    bars["ts"] = ensure_tz(bars["ts"], market_tz)

    # Filter to only new bars
    last_update_ts = state.get("last_update_ts")
    if last_update_ts:
        cutoff = pd.Timestamp(last_update_ts)
        # cutoff must be tz-aware; if not, localize to market tz
        if cutoff.tz is None:
            cutoff = cutoff.tz_localize(market_tz)
        new_bars = bars[bars["ts"] > cutoff].copy()
    else:
        new_bars = bars.copy()

    if new_bars.empty:
        state["status"] = "up_to_date"
        atomic_write_json(state, latest_path)
        atomic_write_json(state, snap_path)
        return state

    # Split by session
    # Step 1 labels session column; if missing, we can infer later. For now require it.
    if "session" not in new_bars.columns:
        raise ValueError(f"Missing 'session' column in intraday bars for {ticker} {day}. Run Step 1 intraday ingest with session labeling.")

    df_pre = new_bars[new_bars["session"] == SESSION_PRE]
    df_rth = new_bars[new_bars["session"] == SESSION_RTH]
    df_aft = new_bars[new_bars["session"] == SESSION_AFT]

    _update_session_agg(state["sessions"][SESSION_PRE], df_pre, compute_vwap=(SESSION_PRE in compute_vwap_sessions))
    _update_session_agg(state["sessions"][SESSION_RTH], df_rth, compute_vwap=(SESSION_RTH in compute_vwap_sessions))
    _update_session_agg(state["sessions"][SESSION_AFT], df_aft, compute_vwap=(SESSION_AFT in compute_vwap_sessions))

    _update_opening_range(state, bars[bars["session"] == SESSION_RTH], day, market_tz)

    # Update last_update_ts to latest bar timestamp (any session)
    last_ts = pd.DatetimeIndex(new_bars["ts"]).max()
    state["last_update_ts"] = pd.Timestamp(last_ts).isoformat()
    state["status"] = "updated"

    # Write outputs
    atomic_write_json(state, latest_path)
    atomic_write_json(state, snap_path)
    return state
