from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

import json

from MarketAtlas.core.config import AppConfig
from MarketAtlas.core.jsonio import atomic_write_json, read_json
from MarketAtlas.insights.eod.tomorrow_range import forecast_tomorrow_range_one
from MarketAtlas.insights.eod.eod_i18n_views import attach_eod_views

def _compute_predicted_day(day: str) -> str:
    """
    Compute the *predicted* trading day for the 'tomorrow_range' output.

    Priority:
      1) If pandas_market_calendars is available, use NYSE calendar (best).
      2) Otherwise fall back to pandas BusinessDay (Mon–Fri), which ignores market holidays.
    """
    d = pd.Timestamp(day).normalize()

    # Try NYSE trading calendar if installed
    try:
        import pandas_market_calendars as mcal  # type: ignore

        nyse = mcal.get_calendar("NYSE")
        # Search the next ~10 calendar days to find the next valid session
        start = d
        end = d + pd.Timedelta(days=10)
        sched = nyse.schedule(start_date=start, end_date=end)
        sessions = list(sched.index)
        # We want the first session strictly after 'day'
        for s in sessions:
            s_day = pd.Timestamp(s).normalize()
            if s_day > d:
                return s_day.date().isoformat()
    except Exception:
        pass

    # Fallback: next weekday (Mon–Fri)
    return (d + pd.tseries.offsets.BDay(1)).date().isoformat()


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


def _append_jsonl(path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False))
        f.write("\n")



def _extract_key_levels(levels_payload: Dict[str, Any], last: float, topk: int = 6) -> Dict[str, Any]:
    """
    Uses Step 3 levels file (zones) and selects closest supports/resistances around last.
    This is "good enough for UI" and avoids redesigning levels.
    """
    levels = list(levels_payload.get("levels") or [])

    def zone_mid(lv: Dict[str, Any]) -> Optional[float]:
        zlo = _sf(lv.get("zone_low"))
        zhi = _sf(lv.get("zone_high"))
        if zlo is None or zhi is None:
            return None
        return 0.5 * (zlo + zhi)

    sups = []
    ress = []
    for lv in levels:
        mid = zone_mid(lv)
        if mid is None:
            continue
        item = {
            "id": lv.get("id"),
            "kind": lv.get("kind"),
            "zone_low": lv.get("zone_low"),
            "zone_high": lv.get("zone_high"),
            "strength": lv.get("strength"),
            "source": lv.get("source"),
            "dist_pts": float(abs(mid - last)),
        }
        if mid <= last:
            sups.append(item)
        else:
            ress.append(item)

    sups = sorted(sups, key=lambda x: x["dist_pts"])[:topk]
    ress = sorted(ress, key=lambda x: x["dist_pts"])[:topk]

    return {"supports": sups, "resistances": ress}


def _risk_notes(daily_latest: Dict[str, Any], regime_latest: Dict[str, Any]) -> List[str]:
    f = (daily_latest.get("features") or {})
    atr = _sf(f.get("atr"))
    rv = _sf(f.get("rv"))
    gap_atr = _sf(f.get("gap_atr"))
    regime = str(regime_latest.get("asof_regime") or "")

    notes: List[str] = []
    if regime in ("high_vol", "transition"):
        notes.append(f"Regime={regime}: consider wider error bars and defined-risk structures.")
    if gap_atr is not None and abs(gap_atr) >= 0.8:
        notes.append(f"Large gap vs ATR (gap_atr={gap_atr:.2f}): open may be discontinuous.")
    if rv is not None and rv >= 0.02:
        notes.append(f"Elevated realized vol proxy (rv={rv:.3f}).")
    if atr is not None and atr <= 0:
        notes.append("ATR invalid (should not happen).")

    return notes


def eod_run_one(cfg: AppConfig, ticker: str, day: str) -> Dict[str, Any]:
    daily_latest = read_json(cfg.path("features_daily_latest_dir") / f"{ticker}.json") or {}
    if daily_latest.get("ticker") != ticker:
        raise ValueError(f"Missing Step 8 daily_latest for {ticker}.")

    regime_latest = read_json(cfg.path("features_regime_labels_latest_dir") / f"{ticker}.json") or {}
    if regime_latest.get("ticker") != ticker:
        raise ValueError(f"Missing Step 9 regime_labels_latest for {ticker}.")

    # IMPORTANT: use the actual as-of day in the daily data
    asof_day_in_data = str(daily_latest.get("day") or day)

    predicted_day = _compute_predicted_day(asof_day_in_data)

    # Compute tomorrow range based on as-of day in data (not the CLI run day)
    tr = forecast_tomorrow_range_one(cfg, ticker=ticker, day=asof_day_in_data)

    last_close = _sf(daily_latest.get("last_price"))
    if last_close is None:
        raise ValueError(f"daily_latest.last_price missing for {ticker}.")

    lv = read_json(cfg.path("levels_dir") / f"{ticker}.json") or {}
    key_levels = _extract_key_levels(lv, last=last_close, topk=6)

    payload: Dict[str, Any] = {
        "ticker": ticker,

        # Keep run day (what you typed) so logs/paths remain stable
        "day": day,

        # Add explicit as-of day in data
        "asof_day": asof_day_in_data,

        # This is the session the forecast is meant for
        "predicted_day": predicted_day,

        "market_tz": cfg.market_tz,
        "asof_ts": daily_latest.get("asof_ts"),
        "today_close": last_close,
        "regime_context": {
            "asof_label_day": regime_latest.get("asof_label_day"),
            "asof_regime": regime_latest.get("asof_regime"),
        },
        "tomorrow_range": tr.get("tomorrow_range"),
        "tomorrow_key_levels": key_levels,
        "risk_notes": _risk_notes(daily_latest, regime_latest),
        "sources": {
            "daily_latest": str((cfg.path("features_daily_latest_dir") / f"{ticker}.json").resolve()),
            "regime_labels_latest": str((cfg.path("features_regime_labels_latest_dir") / f"{ticker}.json").resolve()),
            "levels_latest": str((cfg.path("levels_dir") / f"{ticker}.json").resolve()),
        },
    }
    return payload



def _write_eod_plan(cfg: AppConfig, day: str, ticker: str, payload: Dict[str, Any]) -> str:
    out_dir = cfg.path("outputs_eod_plans_dir") / day
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = (out_dir / f"{ticker}.json").resolve()
    mode = (cfg.raw.get("i18n", {}) or {}).get("mode", "en")
    payload = attach_eod_views(payload, mode=str(mode))
    atomic_write_json(payload, out_path)
    return str(out_path)


def eod_run_all(cfg: AppConfig, day: str, ticker: Optional[str] = None) -> Dict[str, Any]:
    tickers = [ticker] if ticker else list(cfg.daily_assets.keys())
    out: Dict[str, Any] = {"day": day, "results": {}}

    log_path = None
    if "outputs_eod_log_dir" in (cfg.raw or {}):
        log_path = (cfg.path("outputs_eod_log_dir") / f"{day}.jsonl").resolve()

    for t in tickers:
        try:
            payload = eod_run_one(cfg, ticker=str(t), day=day)
            out_json = _write_eod_plan(cfg, day=day, ticker=str(t), payload=payload)

            if log_path is not None:
                _append_jsonl(log_path, payload)

            out["results"][str(t)] = {"ticker": str(t), "day": day, "status": "ok", "out_json": out_json}
        except Exception as e:
            out["results"][str(t)] = {"ticker": str(t), "day": day, "status": "error", "error": str(e)}

    return out
