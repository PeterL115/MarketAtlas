from __future__ import annotations

from typing import Any, Dict

from MarketAtlas.core.config import AppConfig
from MarketAtlas.regimes.regime_labeler import generate_regime_labels_one


def _get_daily_tickers(cfg: AppConfig) -> list[str]:
    if hasattr(cfg, "daily_assets") and cfg.daily_assets:
        return list(cfg.daily_assets.keys())

    raw = getattr(cfg, "raw", {}) or {}
    assets = raw.get("assets", {}) or {}
    daily = assets.get("daily", None)

    if isinstance(daily, dict) and daily:
        return list(daily.keys())
    if isinstance(daily, list) and daily:
        return [str(x) for x in daily]

    raise ValueError("No daily assets configured. Add assets.daily in config.")


def regime_labels_all(cfg: AppConfig, day: str) -> Dict[str, Any]:
    tickers = _get_daily_tickers(cfg)
    out: Dict[str, Any] = {"day": str(day), "results": {}}

    for t in tickers:
        try:
            res = generate_regime_labels_one(cfg, ticker=str(t), day=str(day))
            out["results"][str(t)] = res
        except Exception as e:
            out["results"][str(t)] = {"ticker": str(t), "day": str(day), "status": "error", "error": str(e)}

    return out


def regime_labels_one(cfg: AppConfig, ticker: str, day: str) -> Dict[str, Any]:
    return generate_regime_labels_one(cfg, ticker=str(ticker), day=str(day))
