from __future__ import annotations

from typing import Any, Dict

from MarketAtlas.core.config import AppConfig
from MarketAtlas.features.daily.daily_feature_pipeline import generate_daily_features_one


def _get_daily_tickers(cfg: AppConfig) -> list[str]:
    """
    Prefer cfg.daily_assets if AppConfig provides it (most consistent with your other steps).
    Fallback to cfg.raw assets.daily dict/list.
    """
    # Best case: AppConfig parsed assets into dicts
    if hasattr(cfg, "daily_assets") and cfg.daily_assets:
        # cfg.daily_assets is typically a dict mapping {alias: provider_symbol}
        return list(cfg.daily_assets.keys())

    # Fallback: parse raw config
    raw = getattr(cfg, "raw", {}) or {}
    assets = raw.get("assets", {}) or {}
    daily = assets.get("daily", None)

    if isinstance(daily, dict) and daily:
        return list(daily.keys())
    if isinstance(daily, list) and daily:
        return [str(x) for x in daily]

    raise ValueError("No daily assets configured. Add assets.daily in config.")


def daily_features_all(cfg: AppConfig, day: str) -> Dict[str, Any]:
    """
    Step 8 (all tickers used for daily training/EOD):
      - SPX anchor
      - SPY
      - M7

    Input:  data/daily/<TICKER>.parquet
    Output: features/daily/<TICKER>/<day>.parquet + features/daily_latest/<TICKER>.json
    """
    tickers = _get_daily_tickers(cfg)

    out: Dict[str, Any] = {"day": str(day), "results": {}}
    for t in tickers:
        try:
            res = generate_daily_features_one(cfg, ticker=str(t), day=str(day))
            out["results"][str(t)] = res
        except Exception as e:
            out["results"][str(t)] = {
                "ticker": str(t),
                "day": str(day),
                "status": "error",
                "error": str(e),
            }

    return out


def daily_features_one(cfg: AppConfig, ticker: str, day: str) -> Dict[str, Any]:
    return generate_daily_features_one(cfg, ticker=str(ticker), day=str(day))
