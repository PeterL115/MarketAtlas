from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import re

import numpy as np
import pandas as pd

from MarketAtlas.core.io import read_parquet, ensure_dir, atomic_write_parquet
from MarketAtlas.core.jsonio import atomic_write_json
from MarketAtlas.core.config import AppConfig


# ---------------------------------------------------------------------
# Regime labeling parameters
# ---------------------------------------------------------------------
@dataclass
class RegimeLabelParams:
    # trend / range thresholds
    er_trend_hi: float = 0.55
    er_range_lo: float = 0.40

    # slope normalization threshold (slope_close / close)
    slope_trend_hi: float = 0.00030  # ~0.03% per day in slope units (tunable)

    # volatility regime thresholds using range_atr
    range_atr_highvol: float = 1.35

    # transition rules
    transition_if_er_mid: bool = True
    er_mid_lo: float = 0.40
    er_mid_hi: float = 0.55


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
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


def _label_row(er: Optional[float], slope_n: Optional[float], range_atr: Optional[float], p: RegimeLabelParams) -> str:
    # High-volatility first (independent of direction)
    if range_atr is not None and range_atr >= p.range_atr_highvol:
        return "high_vol"

    # Trend regimes
    if er is not None and slope_n is not None:
        if er >= p.er_trend_hi and slope_n >= p.slope_trend_hi:
            return "trend_up"
        if er >= p.er_trend_hi and slope_n <= -p.slope_trend_hi:
            return "trend_down"

    # Range regime
    if er is not None:
        if er <= p.er_range_lo:
            return "range"

    # Transition (default for ambiguous cases)
    if p.transition_if_er_mid and er is not None and p.er_mid_lo < er < p.er_mid_hi:
        return "transition"

    return "transition"


# Match: YYYY-MM-DD.parquet
_DATE_PARQUET_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\.parquet$")


def _resolve_daily_features_parquet(cfg: AppConfig, ticker: str, day: str) -> Path:
    """
    Resolve Step 8 daily features parquet path.

    Preferred:
      features/daily/<TICKER>/<day>.parquet

    Fallback:
      If the exact date parquet does not exist (common when data provider is delayed
      or date is a weekend/holiday), choose the latest available parquet with
      stem <= day, i.e. the most recent as-of file not after the requested day.

    Returns a Path that may or may not exist (caller decides how to error).
    """
    base_dir = cfg.path("features_daily_dir") / ticker
    exact = base_dir / f"{day}.parquet"
    if exact.exists():
        return exact

    if not base_dir.exists():
        return exact

    candidates: list[tuple[str, Path]] = []
    for p in base_dir.glob("*.parquet"):
        if not _DATE_PARQUET_RE.match(p.name):
            continue
        d = p.stem  # YYYY-MM-DD
        if d <= str(day):
            candidates.append((d, p))

    if not candidates:
        return exact

    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


# ---------------------------------------------------------------------
# Core labeling logic
# ---------------------------------------------------------------------
def label_daily_regimes(df_feat: pd.DataFrame, params: Optional[RegimeLabelParams] = None) -> pd.DataFrame:
    """
    df_feat is the Step 8 daily feature table, with columns at least:
      ts, day, close, er, slope_close, range_atr
    """
    p = params or RegimeLabelParams()

    if df_feat is None or df_feat.empty:
        raise ValueError("Empty daily features table.")

    df = df_feat.copy()
    if "day" not in df.columns:
        raise ValueError("daily features missing 'day' column.")
    if "close" not in df.columns:
        raise ValueError("daily features missing 'close' column.")
    if "er" not in df.columns:
        raise ValueError("daily features missing 'er' column.")
    if "slope_close" not in df.columns:
        raise ValueError("daily features missing 'slope_close' column.")
    if "range_atr" not in df.columns:
        raise ValueError("daily features missing 'range_atr' column.")

    df = df.sort_values("day").reset_index(drop=True)

    close = pd.to_numeric(df["close"], errors="coerce")
    slope = pd.to_numeric(df["slope_close"], errors="coerce")
    er = pd.to_numeric(df["er"], errors="coerce")
    range_atr = pd.to_numeric(df["range_atr"], errors="coerce")

    # Normalize slope by price level to make it comparable across tickers
    slope_n = slope / close.replace(0.0, np.nan)

    labels = []
    for i in range(len(df)):
        lbl = _label_row(
            er=_sf(er.iloc[i]),
            slope_n=_sf(slope_n.iloc[i]),
            range_atr=_sf(range_atr.iloc[i]),
            p=p,
        )
        labels.append(lbl)

    out = pd.DataFrame(
        {
            "day": df["day"].astype(str),
            "close": close,
            "er": er,
            "slope_n": slope_n,
            "range_atr": range_atr,
            "regime": labels,
        }
    )

    # Simple transition flag if label changed vs prior day
    out["regime_prev"] = out["regime"].shift(1)
    out["changed"] = (out["regime"] != out["regime_prev"]).fillna(False)

    return out


def write_regime_outputs(cfg: AppConfig, ticker: str, asof_day: str, df_labels: pd.DataFrame) -> Dict[str, str]:
    """
    Writes:
      - data/regime_labels/<TICKER>/<asof_day>.parquet
      - data/regime_labels_latest/<TICKER>.json
    """
    out_dir = cfg.path("regime_labels_dir") / ticker
    latest_dir = cfg.path("regime_labels_latest_dir")
    ensure_dir(out_dir)
    ensure_dir(latest_dir)

    out_parquet = (out_dir / f"{asof_day}.parquet").resolve()
    latest_json = (latest_dir / f"{ticker}.json").resolve()

    atomic_write_parquet(df_labels, out_parquet)

    row = df_labels[df_labels["day"] == asof_day].tail(1)
    if row.empty:
        row = df_labels.tail(1)
    r = row.iloc[0].to_dict()

    payload: Dict[str, Any] = {
        "ticker": ticker,
        "day": str(r.get("day")),
        "regime": str(r.get("regime")),
        "changed": bool(r.get("changed")),
        "metrics": {
            "er": _sf(r.get("er")),
            "slope_n": _sf(r.get("slope_n")),
            "range_atr": _sf(r.get("range_atr")),
        },
        "paths": {"labels_parquet": str(out_parquet)},
    }

    atomic_write_json(payload, latest_json)

    return {"labels_parquet": str(out_parquet), "latest_json": str(latest_json)}


def generate_regime_labels_one(
    cfg: AppConfig,
    ticker: str,
    day: str,
    params: Optional[RegimeLabelParams] = None
) -> Dict[str, Any]:
    """
    Step 9: Generate regime labels from Step 8 daily features.

    Robust behavior:
      - Prefer exact features file for the requested day
      - If not present, fall back to the latest available features file with date <= requested day
        (common when requested day is non-trading or provider is delayed)

    Output "day" remains the requested day, but "asof_label_day" reflects the actual last
    labeled row available (<= requested day).
    """
    # Step 8 feature table is stored as: features/daily/<TICKER>/<day>.parquet
    feat_path = _resolve_daily_features_parquet(cfg, ticker=ticker, day=str(day))
    df_feat = read_parquet(feat_path)
    if df_feat is None or df_feat.empty:
        raise ValueError(f"Missing Step 8 daily features parquet: {feat_path}")

    df_labels = label_daily_regimes(df_feat, params=params)

    # keep <= requested day only
    df_labels = df_labels[df_labels["day"] <= str(day)].copy()
    if df_labels.empty:
        raise ValueError(f"No label rows <= {day} for {ticker}")

    # We write output parquet using requested day as the filename for determinism,
    # even if the latest available label day is earlier.
    paths = write_regime_outputs(cfg, ticker=ticker, asof_day=str(day), df_labels=df_labels)

    asof_label_day = str(df_labels.tail(1)["day"].iloc[0])
    asof_regime = str(df_labels.tail(1)["regime"].iloc[0])

    return {
        "ticker": ticker,
        "day": str(day),
        "status": "ok",
        "labels_parquet": paths["labels_parquet"],
        "latest_json": paths["latest_json"],
        "rows": int(len(df_labels)),
        "asof_label_day": asof_label_day,
        "asof_regime": asof_regime,
        "features_source": str(feat_path),
    }
