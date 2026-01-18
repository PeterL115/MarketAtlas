from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from MarketAtlas.core.config import AppConfig
from MarketAtlas.core.jsonio import read_json


@dataclass
class TomorrowRangeParams:
    # Base band multipliers in "sigma" units
    # We convert to price points by using ATR and regime multipliers.
    mult_45: float = 0.55
    mult_68: float = 1.00
    mult_95: float = 2.00

    # Regime multipliers applied to ATR-based sigma
    regime_mult = {
        "trend_up": 1.05,
        "trend_down": 1.05,
        "range": 0.90,
        "high_vol": 1.30,
        "transition": 1.15,
    }

    # Guardrails
    min_atr: float = 1e-6


def _sf(x) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if pd.isna(v):
            return None
        return v
    except Exception:
        return None


def _band(center: float, sigma_pts: float, mult: float) -> Dict[str, float]:
    w = sigma_pts * mult
    return {"low": float(center - w), "high": float(center + w)}


def forecast_tomorrow_range_one(cfg: AppConfig, ticker: str, day: str, params: Optional[TomorrowRangeParams] = None) -> Dict[str, Any]:
    """
    Step 12: EOD-only tomorrow range bands (45/68/95).
    Uses daily ATR as baseline sigma proxy, adjusted by latest regime label.
    """
    p = params or TomorrowRangeParams()

    # Step 8 output
    daily_latest = read_json(cfg.path("features_daily_latest_dir") / f"{ticker}.json") or {}
    if daily_latest.get("ticker") != ticker:
        raise ValueError(f"Missing Step 8 daily_latest for {ticker}.")

    # Step 9 output
    reg_latest = read_json(cfg.path("features_regime_labels_latest_dir") / f"{ticker}.json") or {}
    if reg_latest.get("ticker") != ticker:
        raise ValueError(f"Missing Step 9 regime_labels_latest for {ticker}.")

    atr = _sf((daily_latest.get("features") or {}).get("atr"))
    last_close = _sf(daily_latest.get("last_price"))

    if atr is None or atr <= p.min_atr:
        raise ValueError(f"ATR missing/invalid for {ticker}: {atr}")
    if last_close is None:
        raise ValueError(f"Last close missing for {ticker} in daily_latest.")

    regime = str(reg_latest.get("asof_regime") or "transition").lower()
    rmult = float(p.regime_mult.get(regime, p.regime_mult["transition"]))

    # sigma proxy: ATR adjusted
    sigma_pts = atr * rmult

    out = {
        "ticker": ticker,
        "day": day,  # EOD run day
        "market_tz": cfg.market_tz,
        "asof_ts": daily_latest.get("asof_ts"),
        "inputs": {
            "atr": atr,
            "last_close": last_close,
            "regime": regime,
            "regime_multiplier": rmult,
            "sigma_pts": sigma_pts,
        },
        "tomorrow_range": {
            "band_45": _band(last_close, sigma_pts, p.mult_45),
            "band_68": _band(last_close, sigma_pts, p.mult_68),
            "band_95": _band(last_close, sigma_pts, p.mult_95),
        },
    }
    return out
