from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from MarketAtlas.core.config import AppConfig
from MarketAtlas.core.jsonio import read_json, atomic_write_json


@dataclass
class TodayRangeParams:
    interval_minutes: int = 5

    # Bands
    z68: float = 1.0
    z95: float = 1.96

    # RTH session definition (ET)
    rth_open_hm: Tuple[int, int] = (9, 30)
    rth_close_hm: Tuple[int, int] = (16, 0)
    pre_open_hm: Tuple[int, int] = (4, 0)
    aft_close_hm: Tuple[int, int] = (20, 0)

    # Vol blending
    # rv is per-5m return std (from Step 4). sigma_intraday_pts = last * rv * sqrt(bars_remaining)
    # sigma_atr_pts = atr * sqrt(remaining_var_frac)
    blend_mode: str = "var"  # "var" or "linear"
    atr_floor_frac: float = 0.10  # prevent sigma from collapsing: sigma >= atr * atr_floor_frac

    # U-shape variance weights within RTH for remaining variance fraction
    u_base: float = 0.35
    u_scale: float = 1.65
    u_power: float = 1.35


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


def _ts(day: str, hm: Tuple[int, int], tz: str) -> pd.Timestamp:
    h, m = hm
    return pd.Timestamp(f"{day} {h:02d}:{m:02d}:00", tz=tz)


def _infer_session(asof: pd.Timestamp, day: str, tz: str, p: TodayRangeParams) -> str:
    pre0 = _ts(day, p.pre_open_hm, tz)
    rth0 = _ts(day, p.rth_open_hm, tz)
    rth1 = _ts(day, p.rth_close_hm, tz)
    aft1 = _ts(day, p.aft_close_hm, tz)

    if pre0 <= asof < rth0:
        return "PRE"
    if rth0 <= asof < rth1:
        return "RTH"
    if rth1 <= asof < aft1:
        return "AFT"
    return "UNK"


def _ceil_div(a: float, b: float) -> int:
    return int(math.ceil(a / b))


def _bars_remaining(asof: pd.Timestamp, end_ts: pd.Timestamp, interval_minutes: int) -> int:
    if end_ts <= asof:
        return 0
    mins = (end_ts - asof).total_seconds() / 60.0
    return max(0, _ceil_div(mins, float(interval_minutes)))


def _u_shape_weights(n: int, base: float, scale: float, power: float) -> np.ndarray:
    """
    Per-bar variance weights across RTH; higher near open/close, lower midday.
    """
    if n <= 1:
        return np.ones((n,), dtype=float)
    x = np.linspace(0.0, 1.0, n, endpoint=False) + (0.5 / n)  # bar centers
    u = (np.abs(x - 0.5) * 2.0) ** power  # 0 at mid, 1 at ends
    w = base + scale * u
    return w.astype(float)


def _remaining_var_frac_rth(bar_index: int, n_bars: int, p: TodayRangeParams) -> float:
    """
    Fraction of total RTH variance expected to remain after bar_index (0-based).
    Uses U-shaped weights.
    """
    w = _u_shape_weights(n_bars, p.u_base, p.u_scale, p.u_power)
    total = float(np.sum(w))
    if total <= 0:
        return 0.0
    # if bar_index points to "current bar", remaining is strictly after it
    rem = float(np.sum(w[(bar_index + 1):])) if bar_index < (n_bars - 1) else 0.0
    return max(0.0, min(1.0, rem / total))


def _band(center: float, sigma_pts: float, z: float) -> Dict[str, float]:
    lo = center - z * sigma_pts
    hi = center + z * sigma_pts
    return {"low": float(lo), "high": float(hi), "width": float(hi - lo)}


def _cond_extremes(
    low_so_far: Optional[float],
    high_so_far: Optional[float],
    center: float,
    sigma_pts: float,
    z: float,
) -> Dict[str, float]:
    lo_cand = center - z * sigma_pts
    hi_cand = center + z * sigma_pts
    lo = float(min(low_so_far, lo_cand)) if low_so_far is not None else float(lo_cand)
    hi = float(max(high_so_far, hi_cand)) if high_so_far is not None else float(hi_cand)
    return {"low": lo, "high": hi, "width": float(hi - lo)}


def _blend_sigma(s1: float, s2: float, w: float, mode: str) -> float:
    """
    Blend intraday (s1) and ATR-based (s2) sigma.
    w is weight on intraday component.
    """
    w = float(max(0.0, min(1.0, w)))
    if mode == "linear":
        return float(w * s1 + (1.0 - w) * s2)
    # default: variance blend
    return float(math.sqrt(w * (s1 ** 2) + (1.0 - w) * (s2 ** 2)))


def nowcast_today_range_one(cfg: AppConfig, ticker: str, day: str, params: Optional[TodayRangeParams] = None) -> Dict[str, Any]:
    p = params or TodayRangeParams()

    feat_path = cfg.path("features_intraday_latest_dir") / f"{ticker}.json"
    state_path = cfg.path("session_state_dir") / f"{ticker}.json"
    lv_path = cfg.path("levels_dir") / f"{ticker}.json"

    feat = read_json(feat_path) or {}
    state = read_json(state_path) or {}
    lv = read_json(lv_path) or {}

    if feat.get("ticker") != ticker:
        raise ValueError(f"Missing/invalid features for {ticker}. Run Step 4.")
    if state.get("ticker") != ticker or state.get("day") != day:
        raise ValueError(f"Session state mismatch for {ticker}. Run Step 2 for {day}.")
    if lv.get("ticker") != ticker:
        raise ValueError(f"Levels missing for {ticker}. Run Step 3.")

    asof = pd.Timestamp(feat["asof_ts"]).tz_convert(cfg.market_tz)
    last_price = float(feat["last_price"])

    rv = _safe_float((feat.get("features") or {}).get("rv"))  # per-5m return std
    atr = _safe_float((lv.get("atr") or {}).get("value"))    # daily ATR in points

    session = feat.get("session") or _infer_session(asof, day, cfg.market_tz, p)
    if session == "UNK":
        session = _infer_session(asof, day, cfg.market_tz, p)

    # define session end for remaining-session band
    if session == "PRE":
        end_ts = _ts(day, p.rth_open_hm, cfg.market_tz)
    elif session == "RTH":
        end_ts = _ts(day, p.rth_close_hm, cfg.market_tz)
    elif session == "AFT":
        end_ts = _ts(day, p.aft_close_hm, cfg.market_tz)
    else:
        # fallback to RTH close
        end_ts = _ts(day, p.rth_close_hm, cfg.market_tz)

    bars_rem = _bars_remaining(asof, end_ts, p.interval_minutes)

    # Remaining variance fraction within RTH (only meaningful during RTH)
    n_rth_bars = int(((p.rth_close_hm[0] * 60 + p.rth_close_hm[1]) - (p.rth_open_hm[0] * 60 + p.rth_open_hm[1])) / p.interval_minutes)
    rth_open = _ts(day, p.rth_open_hm, cfg.market_tz)

    if session == "RTH":
        mins_since_open = (asof - rth_open).total_seconds() / 60.0
        bar_index = max(0, min(n_rth_bars - 1, int(math.floor(mins_since_open / p.interval_minutes))))
        rem_var_frac = _remaining_var_frac_rth(bar_index, n_rth_bars, p)
    else:
        # crude proxy outside RTH: scale by time remaining relative to RTH length
        rem_var_frac = float(max(0.0, min(1.0, bars_rem / max(1, n_rth_bars))))

    # Compute two sigma estimates (points)
    sigma_intraday_pts = None
    if rv is not None and rv > 0 and bars_rem > 0:
        sigma_intraday_pts = float(last_price * rv * math.sqrt(float(bars_rem)))

    sigma_atr_pts = None
    if atr is not None and atr > 0:
        sigma_atr_pts = float(atr * math.sqrt(rem_var_frac))

    # Blend weight: more observed bars => trust intraday more
    rth_bar_count = _safe_float((state.get("sessions", {}).get("RTH", {}) or {}).get("bar_count"))
    if rth_bar_count is None:
        # infer from time since open
        if session == "RTH":
            mins_since_open = (asof - rth_open).total_seconds() / 60.0
            rth_bar_count = float(max(0, min(n_rth_bars, int(math.floor(mins_since_open / p.interval_minutes)) + 1)))
        else:
            rth_bar_count = 0.0

    w = float(max(0.0, min(1.0, (rth_bar_count / max(1.0, float(n_rth_bars))))))

    # Final sigma (points), with floor
    sigma_candidates = []
    if sigma_intraday_pts is not None:
        sigma_candidates.append(sigma_intraday_pts)
    if sigma_atr_pts is not None:
        sigma_candidates.append(sigma_atr_pts)

    if not sigma_candidates:
        raise ValueError(f"Cannot compute sigma for {ticker}: missing rv and atr.")

    if sigma_intraday_pts is None:
        sigma_pts = float(sigma_atr_pts)
        sigma_src = "atr_only"
    elif sigma_atr_pts is None:
        sigma_pts = float(sigma_intraday_pts)
        sigma_src = "intraday_only"
    else:
        sigma_pts = _blend_sigma(sigma_intraday_pts, sigma_atr_pts, w=w, mode=p.blend_mode)
        sigma_src = f"blend_{p.blend_mode}"

    if atr is not None and atr > 0:
        sigma_pts = float(max(sigma_pts, atr * p.atr_floor_frac))

    # Observed extremes so far (prefer current session; fallback to RTH)
    sess = state.get("sessions", {}) or {}
    cur_sess = sess.get(session, {}) if session in sess else {}
    rth_sess = sess.get("RTH", {}) or {}

    low_so_far = _safe_float(cur_sess.get("low")) or _safe_float(rth_sess.get("low"))
    high_so_far = _safe_float(cur_sess.get("high")) or _safe_float(rth_sess.get("high"))

    out: Dict[str, Any] = {
        "ticker": ticker,
        "day": day,
        "market_tz": cfg.market_tz,
        "asof_ts": asof.isoformat(),
        "session": session,
        "last_price": last_price,
        "inputs": {
            "rv_5m": rv,
            "atr_daily": atr,
            "bars_remaining": bars_rem,
            "remaining_var_frac": float(rem_var_frac),
            "blend_weight_intraday": float(w),
            "sigma_source": sigma_src,
            "sigma_intraday_pts": sigma_intraday_pts,
            "sigma_atr_pts": sigma_atr_pts,
        },
        "remaining_session": {
            "end_ts": end_ts.isoformat(),
            "sigma_pts": float(sigma_pts),
            "band_68": _band(last_price, sigma_pts, p.z68),
            "band_95": _band(last_price, sigma_pts, p.z95),
        },
        "conditional_full_day": {
            # Conditional full-day extremes: incorporates highs/lows already printed today
            "low_so_far": low_so_far,
            "high_so_far": high_so_far,
            "band_68": _cond_extremes(low_so_far, high_so_far, last_price, sigma_pts, p.z68),
            "band_95": _cond_extremes(low_so_far, high_so_far, last_price, sigma_pts, p.z95),
        },
    }

    return out


def write_today_range(cfg: AppConfig, payload: Dict[str, Any]) -> Tuple[str, str]:
    import json
    import os
    import tempfile
    from pathlib import Path

    ticker = str(payload["ticker"])
    day = str(payload["day"])

    out_dir = cfg.path("today_range_dir")
    log_dir = cfg.path("today_range_log_dir")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    out_path = (Path(out_dir) / f"{ticker}.json").resolve()
    log_path = (Path(log_dir) / f"{day}.jsonl").resolve()

    # Atomic JSON write
    tmp_fd, tmp_name = tempfile.mkstemp(prefix=f"{ticker}_", suffix=".tmp", dir=str(out_path.parent))
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_name, out_path)
    finally:
        try:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
        except Exception:
            pass

    # Append JSONL log
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False))
        f.write("\n")

    return str(out_path), str(log_path)
