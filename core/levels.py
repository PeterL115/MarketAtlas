from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .config import AppConfig
from .constants import SESSION_AFT, SESSION_PRE, SESSION_RTH
from .io import read_parquet
from .jsonio import atomic_write_json, read_json
from .timeutils import ensure_tz


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


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    ATR based on True Range. Returns a series aligned to df index.
    Expects columns: high, low, close
    """
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.rolling(period, min_periods=period).mean()


def _zone_half_width(
    price: float,
    atr_val: Optional[float],
    zone_atr_mult: float,
    min_zone_bp: float,
) -> float:
    """
    Half-width of zone. Uses max(ATR*mult, price * bp).
    """
    bp_width = price * (min_zone_bp / 10000.0)
    atr_width = (atr_val * zone_atr_mult) if (atr_val is not None and atr_val > 0) else 0.0
    return float(max(bp_width, atr_width))


def _make_level(
    level_id: str,
    kind: str,
    source: str,
    price: float,
    half_width: float,
    strength: int = 1,
    session: Optional[str] = None,
    notes: Optional[str] = None,
) -> Dict[str, Any]:
    lo = float(price - half_width)
    hi = float(price + half_width)
    return {
        "id": level_id,
        "kind": kind,                  # "support" / "resistance" / "pivot" / "anchor"
        "source": source,              # "prior_day" / "prior_week" / "swing_pivot" / ...
        "session": session,            # PRE/RTH/AFT or None
        "price": float(price),         # zone center
        "zone_low": lo,
        "zone_high": hi,
        "strength": int(strength),
        "notes": notes or "",
    }


def _flatten_daily(df: pd.DataFrame, market_tz: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    out["ts"] = ensure_tz(out["ts"], market_tz)
    out = out.sort_values("ts").reset_index(drop=True)

    for c in ["open", "high", "low", "close", "volume"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=["ts", "open", "high", "low", "close"]).copy()
    return out


def _prior_day_row(daily: pd.DataFrame, day: str) -> Optional[pd.Series]:
    """
    Return the most recent daily bar strictly before `day` (YYYY-MM-DD),
    using market-date day_key.
    """
    if daily.empty or "day_key" not in daily.columns:
        return None

    prior = daily[daily["day_key"] < day]
    if prior.empty:
        return None

    last_day = prior["day_key"].iloc[-1]
    # take the last row of that day (robust to any intraday-like rows mistakenly present)
    return prior[prior["day_key"] == last_day].iloc[-1]


def _prior_week_stats(daily: pd.DataFrame, day: str) -> Optional[Dict[str, float]]:
    """
    Compute prior WEEK H/L/C where week ends on Friday (W-FRI).
    """
    if daily.empty:
        return None

    d = daily.set_index("ts").sort_index()

    weekly = pd.DataFrame({
        "high": d["high"].resample("W-FRI").max(),
        "low": d["low"].resample("W-FRI").min(),
        "close": d["close"].resample("W-FRI").last(),
    }).dropna()

    if weekly.empty:
        return None

    cutoff = pd.Timestamp(day).tz_localize(d.index.tz)
    weekly = weekly[weekly.index < cutoff]
    if weekly.empty:
        return None

    row = weekly.iloc[-1]
    return {"high": float(row["high"]), "low": float(row["low"]), "close": float(row["close"])}


def _swing_pivots(daily: pd.DataFrame, lookback_days: int, left_right: int) -> Tuple[List[float], List[float]]:
    """
    Fractal pivots:
      pivot high if high[i] > highs[i-k..i-1] and > highs[i+1..i+k]
      pivot low similarly.
    Returns (pivot_high_prices, pivot_low_prices)
    """
    if daily.empty:
        return [], []

    d = daily.tail(lookback_days).reset_index(drop=True)
    highs = d["high"].astype(float).values
    lows = d["low"].astype(float).values
    n = len(d)

    ph: List[float] = []
    pl: List[float] = []

    k = int(left_right)
    if n < (2 * k + 1):
        return ph, pl

    for i in range(k, n - k):
        h = highs[i]
        l = lows[i]
        if h > highs[i - k:i].max() and h > highs[i + 1:i + k + 1].max():
            ph.append(float(h))
        if l < lows[i - k:i].min() and l < lows[i + 1:i + k + 1].min():
            pl.append(float(l))

    return ph, pl


def _get_last_price_from_state(state: Dict[str, Any]) -> Optional[float]:
    rth_close = state.get("sessions", {}).get(SESSION_RTH, {}).get("close")
    pre_close = state.get("sessions", {}).get(SESSION_PRE, {}).get("close")
    return _safe_float(rth_close) or _safe_float(pre_close)


def compute_levels(cfg: AppConfig, ticker: str, day: str) -> Dict[str, Any]:
    """
    Compute support/resistance zones for a ticker on a specific day.

    Inputs:
      - daily parquet: data/daily/<ticker>.parquet
      - session state JSON: data/session_state/<ticker>.json

    Output:
      - dict payload suitable for JSON persistence
    """
    market_tz = cfg.market_tz

    # --- Load Step 2 session state ---
    state_path = cfg.path("session_state_dir") / f"{ticker}.json"
    state = read_json(state_path)
    if not state or state.get("ticker") != ticker or state.get("day") != day:
        raise ValueError(
            f"Session state missing or day mismatch for {ticker}. Expected {day}. "
            f"Run Step 2 session-state first."
        )

    last_price = _get_last_price_from_state(state)

    # --- Load daily bars ---
    daily_path = cfg.path("daily_dir") / f"{ticker}.parquet"
    daily_raw = read_parquet(daily_path)
    daily = _flatten_daily(daily_raw, market_tz)
    if daily.empty:
        raise ValueError(f"No daily data found for {ticker} at {daily_path}")

    # IMPORTANT FIX: create a stable market-date key for selecting prior day
    daily = daily.copy()
    daily["day_key"] = daily["ts"].dt.date.astype(str)

    # --- ATR for zone sizing ---
    atr_period = int(cfg.raw.get("levels", {}).get("atr_period", 14))
    atr_series = _atr(daily, period=atr_period)
    atr_val = _safe_float(atr_series.iloc[-1]) if len(atr_series) else None

    zone_atr_mult = float(cfg.raw.get("levels", {}).get("zone_atr_mult", 0.20))
    min_zone_bp = float(cfg.raw.get("levels", {}).get("min_zone_bp", 8))
    max_levels = int(cfg.raw.get("levels", {}).get("max_levels", 40))

    def hw(p: float) -> float:
        return _zone_half_width(p, atr_val, zone_atr_mult, min_zone_bp)

    levels: List[Dict[str, Any]] = []


    # 1) Prior day H/L/C (FIXED selection)
    pd_row = _prior_day_row(daily, day)
    if pd_row is not None:
        pd_high = float(pd_row["high"])
        pd_low = float(pd_row["low"])
        pd_close = float(pd_row["close"])
        levels.append(_make_level("pd_high", "resistance", "prior_day", pd_high, hw(pd_high), strength=4))
        levels.append(_make_level("pd_low", "support", "prior_day", pd_low, hw(pd_low), strength=4))
        levels.append(_make_level("pd_close", "pivot", "prior_day", pd_close, hw(pd_close), strength=3))

    # 2) Prior week H/L/C
    pw = _prior_week_stats(daily, day)
    if pw:
        levels.append(_make_level("pw_high", "resistance", "prior_week", pw["high"], hw(pw["high"]), strength=5))
        levels.append(_make_level("pw_low", "support", "prior_week", pw["low"], hw(pw["low"]), strength=5))
        levels.append(_make_level("pw_close", "pivot", "prior_week", pw["close"], hw(pw["close"]), strength=3))

    # 3) Swing pivots
    pivot_lookback = int(cfg.raw.get("levels", {}).get("pivot_lookback_days", 120))
    left_right = int(cfg.raw.get("levels", {}).get("pivot_left_right", 2))
    ph, pl = _swing_pivots(daily, lookback_days=pivot_lookback, left_right=left_right)

    # Pivot distance filter (keep near-price pivots)
    pivot_max_atr = float(cfg.raw.get("levels", {}).get("pivot_max_atr_distance", 6))
    if last_price is not None and atr_val is not None and atr_val > 0:
        lo_cut = float(last_price - pivot_max_atr * atr_val)
        hi_cut = float(last_price + pivot_max_atr * atr_val)
        ph = [p for p in ph if lo_cut <= p <= hi_cut]
        pl = [p for p in pl if lo_cut <= p <= hi_cut]

    # Append pivots (cap)
    # Resistances: pick higher pivots near price
    for i, p in enumerate(sorted(set(ph))[-12:]):
        levels.append(_make_level(f"sp_h_{i}", "resistance", "swing_pivot", float(p), hw(float(p)), strength=2, notes=f"fractal k={left_right}"))
    # Supports: pick lower pivots near price
    for i, p in enumerate(sorted(set(pl))[:12]):
        levels.append(_make_level(f"sp_l_{i}", "support", "swing_pivot", float(p), hw(float(p)), strength=2, notes=f"fractal k={left_right}"))

    # 4) Intraday anchors from Step 2: VWAP, Opening Range, Session extremes
    vwap_val = _safe_float(state.get("sessions", {}).get(SESSION_RTH, {}).get("vwap", {}).get("value"))
    if vwap_val is not None:
        levels.append(_make_level("rth_vwap", "anchor", "vwap", vwap_val, hw(vwap_val), strength=4, session=SESSION_RTH))

    or_obj = state.get("opening_range", {})
    or_high = _safe_float(or_obj.get("high"))
    or_low = _safe_float(or_obj.get("low"))
    if or_high is not None:
        levels.append(_make_level("or_high", "resistance", "opening_range", or_high, hw(or_high), strength=4, session=SESSION_RTH))
    if or_low is not None:
        levels.append(_make_level("or_low", "support", "opening_range", or_low, hw(or_low), strength=4, session=SESSION_RTH))

    for sess in [SESSION_PRE, SESSION_RTH, SESSION_AFT]:
        s = state.get("sessions", {}).get(sess, {})
        sh = _safe_float(s.get("high"))
        sl = _safe_float(s.get("low"))
        if sh is not None:
            levels.append(_make_level(f"{sess.lower()}_high", "resistance", "session_extreme", sh, hw(sh), strength=3, session=sess))
        if sl is not None:
            levels.append(_make_level(f"{sess.lower()}_low", "support", "session_extreme", sl, hw(sl), strength=3, session=sess))

    include_pm = bool(cfg.raw.get("levels", {}).get("include_contextual_premarket", True))
    if include_pm:
        pre = state.get("sessions", {}).get(SESSION_PRE, {})
        pmh = _safe_float(pre.get("high"))
        pml = _safe_float(pre.get("low"))
        if pmh is not None:
            levels.append(_make_level("pm_high_ctx", "resistance", "premarket_context", pmh, hw(pmh), strength=1, session=SESSION_PRE))
        if pml is not None:
            levels.append(_make_level("pm_low_ctx", "support", "premarket_context", pml, hw(pml), strength=1, session=SESSION_PRE))

    # De-duplicate near-identical levels by (source, rounded price, session)
    dedup: Dict[str, Dict[str, Any]] = {}
    for lv in levels:
        key = f"{lv['source']}:{round(lv['price'], 2)}:{lv.get('session')}"
        if key not in dedup:
            dedup[key] = lv
        else:
            if int(lv["strength"]) > int(dedup[key]["strength"]):
                dedup[key] = lv

    levels = list(dedup.values())

    # Rank by strength desc, then distance to last_price asc
    def rank_key(lv: Dict[str, Any]) -> Tuple:
        strength = int(lv.get("strength", 1))
        if last_price is None:
            return (-strength, 0.0)
        dist = abs(float(lv["price"]) - float(last_price))
        return (-strength, dist)

    levels = sorted(levels, key=rank_key)[:max_levels]

    return {
        "ticker": ticker,
        "day": day,
        "market_tz": market_tz,
        "asof_ts": state.get("last_update_ts"),
        "last_price": last_price,
        "atr": {"period": atr_period, "value": atr_val},
        "params": {
            "pivot_lookback_days": pivot_lookback,
            "pivot_left_right": left_right,
            "zone_atr_mult": zone_atr_mult,
            "min_zone_bp": float(min_zone_bp),
            "max_levels": max_levels,
        },
        "levels": levels,
    }


def write_levels(cfg: AppConfig, ticker: str, day: str, payload: Dict[str, Any]) -> Tuple[str, str]:
    """
    Write latest + snapshot JSON. Returns (latest_path, snapshot_path).
    """
    base = cfg.path("levels_dir")
    latest = (base / f"{ticker}.json").resolve()
    snap = (base / ticker / f"{day}.json").resolve()

    atomic_write_json(payload, latest)
    atomic_write_json(payload, snap)
    return str(latest), str(snap)
