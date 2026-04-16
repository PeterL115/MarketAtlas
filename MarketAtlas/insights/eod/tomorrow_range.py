from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from MarketAtlas.core.config import AppConfig
from MarketAtlas.core.jsonio import read_json


# ─── Z-scores for proper normal confidence intervals ──────────────────────────
# norm.ppf(0.5 + 0.5 * band_pct)
_Z_FOR_BAND: Dict[int, float] = {
    45: 0.5978,   # norm.ppf(0.725)
    68: 1.0000,   # norm.ppf(0.840)  — exactly 1σ by convention
    95: 1.9600,   # norm.ppf(0.975)
}


@dataclass
class TomorrowRangeParams:
    # Sigma blending weights (must sum to 1.0 within each scenario)
    # IV available + RV available: w_iv + w_rv + w_atr
    w_iv:  float = 0.55
    w_rv:  float = 0.20
    w_atr: float = 0.25

    # IV available, no RV:
    w_iv_atr_only_iv:  float = 0.60
    w_iv_atr_only_atr: float = 0.40

    # No IV, RV available:
    w_rv_atr_only_rv:  float = 0.30
    w_rv_atr_only_atr: float = 0.70

    # ATR → 1-day sigma conversion.
    # For ~normally distributed daily returns: ATR ≈ 1.25 * sigma, so factor ≈ 0.80.
    atr_to_sigma: float = 0.80

    # Regime multipliers applied to the blended sigma.
    # Trending regimes tend to have slightly higher realized vol than range-bound ones.
    regime_mult: Dict[str, float] = None  # set in __post_init__

    min_atr: float = 1e-6

    def __post_init__(self) -> None:
        if self.regime_mult is None:
            self.regime_mult = {
                "trend_up":   1.10,
                "trend_down": 1.10,
                "range":      0.85,
                "high_vol":   1.40,
                "transition": 1.00,
            }


def _sf(x) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        return None if pd.isna(v) else v
    except Exception:
        return None


def _band(center: float, sigma_pts: float, band_pct: int) -> Dict[str, float]:
    z = _Z_FOR_BAND.get(band_pct, 1.0)
    return {"low": float(center - z * sigma_pts), "high": float(center + z * sigma_pts)}


def _blend_sigma(
    sigma_atr: float,
    sigma_rv: Optional[float],
    sigma_iv: Optional[float],
    p: TomorrowRangeParams,
) -> float:
    """
    Blend three sigma estimates in priority order: IV > RV > ATR.
    IV is annualized and already converted to 1-day sigma before this call.
    """
    if sigma_iv is not None and sigma_rv is not None:
        return p.w_iv * sigma_iv + p.w_rv * sigma_rv + p.w_atr * sigma_atr
    if sigma_iv is not None:
        return p.w_iv_atr_only_iv * sigma_iv + p.w_iv_atr_only_atr * sigma_atr
    if sigma_rv is not None:
        return p.w_rv_atr_only_rv * sigma_rv + p.w_rv_atr_only_atr * sigma_atr
    return sigma_atr


def forecast_tomorrow_range_one(
    cfg: AppConfig,
    ticker: str,
    day: str,
    iv: Optional[float] = None,
    params: Optional[TomorrowRangeParams] = None,
) -> Dict[str, Any]:
    """
    Step 12: EOD tomorrow range bands (45 / 68 / 95 percent).

    Sigma estimation (priority order):
      1. IV  — annualized implied vol → 1-day sigma = last_close * iv / sqrt(252)
      2. RV  — 20-day realized log-return std → sigma = last_close * rv
      3. ATR — daily ATR * atr_to_sigma factor

    All three are scaled by a regime-dependent multiplier.
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

    feats = daily_latest.get("features") or {}
    atr        = _sf(feats.get("atr"))
    rv         = _sf(feats.get("rv"))      # 20-day rolling log-return std
    last_close = _sf(daily_latest.get("last_price"))

    if atr is None or atr <= p.min_atr:
        raise ValueError(f"ATR missing/invalid for {ticker}: {atr}")
    if last_close is None:
        raise ValueError(f"Last close missing for {ticker}.")

    # Regime (fixed: was reading wrong key before this patch)
    regime = str(reg_latest.get("asof_regime") or reg_latest.get("regime") or "transition").lower()
    rmult  = float(p.regime_mult.get(regime, p.regime_mult.get("transition", 1.0)))

    # ── Sigma components ──────────────────────────────────────────────────────
    sigma_atr = atr * p.atr_to_sigma * rmult

    sigma_rv: Optional[float] = None
    if rv is not None and rv > 0:
        sigma_rv = last_close * rv * rmult

    sigma_iv: Optional[float] = None
    if iv is not None and iv > 0:
        sigma_iv = last_close * (iv / math.sqrt(252)) * rmult

    sigma_pts = _blend_sigma(sigma_atr, sigma_rv, sigma_iv, p)

    out = {
        "ticker": ticker,
        "day": day,
        "market_tz": cfg.market_tz,
        "asof_ts": daily_latest.get("asof_ts"),
        "inputs": {
            "atr":              atr,
            "rv":               rv,
            "iv":               iv,
            "last_close":       last_close,
            "regime":           regime,
            "regime_multiplier": rmult,
            "sigma_atr_pts":    float(sigma_atr),
            "sigma_rv_pts":     sigma_rv,
            "sigma_iv_pts":     sigma_iv,
            "sigma_pts":        float(sigma_pts),
        },
        "tomorrow_range": {
            "band_45": _band(last_close, sigma_pts, 45),
            "band_68": _band(last_close, sigma_pts, 68),
            "band_95": _band(last_close, sigma_pts, 95),
        },
    }
    return out
