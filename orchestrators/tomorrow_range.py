from __future__ import annotations

from typing import Any, Dict, Tuple

from MarketAtlas.core.config import AppConfig
from MarketAtlas.core.jsonio import atomic_write_json
from MarketAtlas.insights.eod.tomorrow_range import forecast_tomorrow_range_one
from MarketAtlas.insights.eod.eod_i18n_views import attach_eod_views


def write_eod_plan(cfg: AppConfig, day: str, ticker: str, payload: Dict[str, Any]) -> str:
    out_dir = cfg.path("outputs_eod_plans_dir") / day
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = (out_dir / f"{ticker}.json").resolve()
    mode = (cfg.raw.get("i18n", {}) or {}).get("mode", "en")
    payload = attach_eod_views(payload, mode=str(mode))
    atomic_write_json(payload, out_path)
    return str(out_path)


def tomorrow_range_all(cfg: AppConfig, day: str) -> Dict[str, Any]:
    tickers = list(cfg.daily_assets.keys())
    out: Dict[str, Any] = {"day": day, "results": {}}

    for t in tickers:
        try:
            payload = forecast_tomorrow_range_one(cfg, ticker=t, day=day)
            out_json = write_eod_plan(cfg, day=day, ticker=t, payload=payload)
            out["results"][t] = {"ticker": t, "day": day, "status": "ok", "out_json": out_json}
        except Exception as e:
            out["results"][t] = {"ticker": t, "day": day, "status": "error", "error": str(e)}

    return out


def tomorrow_range_one(cfg: AppConfig, ticker: str, day: str) -> Dict[str, Any]:
    payload = forecast_tomorrow_range_one(cfg, ticker=ticker, day=day)
    out_json = write_eod_plan(cfg, day=day, ticker=ticker, payload=payload)
    return {"ticker": ticker, "day": day, "status": "ok", "out_json": out_json}
