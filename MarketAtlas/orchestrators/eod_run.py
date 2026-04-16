from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional

import pandas as pd

from MarketAtlas.core.config import AppConfig
from MarketAtlas.core.jsonio import atomic_write_json, read_json
from MarketAtlas.core.pivot_levels import read_pivot_levels_for_day
from MarketAtlas.insights.eod.tomorrow_range import forecast_tomorrow_range_one
from MarketAtlas.insights.eod.eod_i18n_views import attach_eod_views


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _compute_predicted_day(day: str) -> str:
    """Next NYSE trading day after `day`. Falls back to next weekday."""
    d = pd.Timestamp(day).normalize()
    try:
        import pandas_market_calendars as mcal  # type: ignore
        nyse = mcal.get_calendar("NYSE")
        sched = nyse.schedule(start_date=d, end_date=d + pd.Timedelta(days=10))
        for s in sched.index:
            s_day = pd.Timestamp(s).normalize()
            if s_day > d:
                return s_day.date().isoformat()
    except Exception:
        pass
    return (d + pd.tseries.offsets.BDay(1)).date().isoformat()


def _sf(x) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        return None if pd.isna(v) else v
    except Exception:
        return None


def _append_jsonl(path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False))
        f.write("\n")


def _extract_key_levels(levels_payload: Dict[str, Any], last: float, topk: int = 6) -> Dict[str, Any]:
    """Select the closest supports/resistances above and below `last`."""
    levels = list(levels_payload.get("levels") or [])

    def zone_mid(lv: Dict[str, Any]) -> Optional[float]:
        zlo = _sf(lv.get("zone_low"))
        zhi = _sf(lv.get("zone_high"))
        return None if zlo is None or zhi is None else 0.5 * (zlo + zhi)

    sups, ress = [], []
    for lv in levels:
        mid = zone_mid(lv)
        if mid is None:
            continue
        item = {
            "id":        lv.get("id"),
            "kind":      lv.get("kind"),
            "zone_low":  lv.get("zone_low"),
            "zone_high": lv.get("zone_high"),
            "strength":  lv.get("strength"),
            "source":    lv.get("source"),
            "dist_pts":  float(abs(mid - last)),
        }
        (sups if mid <= last else ress).append(item)

    sups = sorted(sups, key=lambda x: x["dist_pts"])[:topk]
    ress = sorted(ress, key=lambda x: x["dist_pts"])[:topk]
    return {"supports": sups, "resistances": ress}


def _read_iv_from_intraday(cfg: AppConfig, ticker: str) -> Optional[float]:
    """
    Best-effort: pull IV from today's intraday insights output.
    This is the most up-to-date forward-looking vol estimate available.
    """
    try:
        p = cfg.path("outputs_intraday_latest_dir") / f"{ticker}.json"
        intraday = read_json(p) or {}
        iv = _sf((intraday.get("diagnostics") or {}).get("iv"))
        if iv is None:
            iv = _sf((intraday.get("features") or {}).get("iv"))
        return iv
    except Exception:
        return None


def _risk_notes(daily_latest: Dict[str, Any], regime: str) -> List[str]:
    f       = daily_latest.get("features") or {}
    atr     = _sf(f.get("atr"))
    rv      = _sf(f.get("rv"))
    gap_atr = _sf(f.get("gap_atr"))
    er      = _sf(f.get("er"))
    vol_z   = _sf(f.get("vol_z"))

    notes: List[str] = []
    if regime in ("high_vol", "transition"):
        notes.append(f"Regime={regime}: consider wider stops and defined-risk structures.")
    if gap_atr is not None and abs(gap_atr) >= 0.8:
        notes.append(f"Large overnight gap (gap_atr={gap_atr:.2f}): open may be discontinuous.")
    if rv is not None and rv >= 0.02:
        notes.append(f"Elevated 20-day realized vol (rv={rv:.3f}).")
    if vol_z is not None and abs(vol_z) >= 1.5:
        direction = "above" if vol_z > 0 else "below"
        notes.append(f"Volume {direction} average (vol_z={vol_z:.1f}).")
    if er is not None and er >= 0.60:
        notes.append(f"Strong trend efficiency (ER={er:.2f}): breakout structures may be preferred.")
    return notes


def _generate_guidance(
    regime: str,
    daily_latest: Dict[str, Any],
    iv: Optional[float] = None,
    tomorrow_range: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Derive options trade guidance from regime, daily diagnostics, IV, and
    tomorrow's expected range bands.
    """
    f         = daily_latest.get("features") or {}
    er        = _sf(f.get("er"))
    range_atr = _sf(f.get("range_atr"))
    rv        = _sf(f.get("rv"))

    # ── Trade bias ────────────────────────────────────────────────────────────
    bias_map = {
        "trend_up":   "bullish",
        "trend_down": "bearish",
        "range":      "neutral",
        "high_vol":   "neutral",
        "transition": "neutral",
    }
    trade_bias = bias_map.get(regime, "neutral")

    # Avoid trading: high-vol spike with extreme range expansion
    avoid = bool(regime == "high_vol" and range_atr is not None and range_atr > 1.6)

    # Prefer spreads: elevated vol or ambiguous regime
    prefer_spreads = bool(
        regime in ("high_vol", "transition")
        or (rv is not None and rv >= 0.018)
        or (range_atr is not None and range_atr > 1.2)
    )

    # Mean reversion: low ER (choppy) or explicit range regime
    mean_reversion_preferred = bool(regime == "range" or (er is not None and er < 0.35))

    # ── IV vs RV premium environment ─────────────────────────────────────────
    # rv is daily log-return std; annualise × sqrt(252) to compare with IV
    iv_rv_ratio: Optional[float] = None
    premium_env = "neutral"
    if iv is not None and rv is not None and rv > 0:
        rv_ann = rv * math.sqrt(252)
        iv_rv_ratio = round(iv / rv_ann, 3)
        if iv_rv_ratio > 1.15:
            premium_env = "rich"    # IV > realised: premium sellers favoured
        elif iv_rv_ratio < 0.85:
            premium_env = "cheap"   # IV < realised: premium buyers favoured

    # ── Recommended option structure ─────────────────────────────────────────
    if avoid:
        structure = "avoid"
        structure_reason = (
            "High-vol spike + extreme range expansion — skip new positions today."
        )
    elif mean_reversion_preferred and premium_env in ("rich", "neutral"):
        structure = "iron_condor"
        structure_reason = (
            f"Range/choppy regime (ER={er:.2f}) + IV {'elevated' if premium_env == 'rich' else 'fair'} "
            f"→ sell iron condor centred on today's close."
        )
    elif mean_reversion_preferred:
        structure = "iron_condor"
        structure_reason = (
            "Range regime — iron condor or credit spread inside tomorrow's expected range."
        )
    elif trade_bias == "bullish" and premium_env == "rich":
        structure = "bull_put_spread"
        structure_reason = (
            f"Bullish trend (regime={regime}) + elevated IV (IV/RV={iv_rv_ratio:.2f}) "
            "→ sell bull put spread below nearest support."
        )
    elif trade_bias == "bearish" and premium_env == "rich":
        structure = "bear_call_spread"
        structure_reason = (
            f"Bearish trend + elevated IV (IV/RV={iv_rv_ratio:.2f}) "
            "→ sell bear call spread above nearest resistance."
        )
    elif trade_bias == "bullish":
        structure = "long_call_spread"
        structure_reason = (
            "Bullish trending regime; premium not elevated → debit call spread for defined-risk directional play."
        )
    elif trade_bias == "bearish":
        structure = "long_put_spread"
        structure_reason = (
            "Bearish trending regime → debit put spread for defined-risk directional play."
        )
    elif prefer_spreads and premium_env == "rich":
        structure = "short_strangle"
        structure_reason = (
            "Elevated IV, no clear direction → short strangle / wide iron condor at ±1σ edges."
        )
    else:
        structure = "wait"
        structure_reason = (
            "Transition regime, no strong edge — wait for a cleaner setup next session."
        )

    # ── Strike placement context from tomorrow's range ────────────────────────
    strike_notes: List[str] = []
    if tomorrow_range:
        b45 = tomorrow_range.get("band_45") or {}
        b68 = tomorrow_range.get("band_68") or {}
        b95 = tomorrow_range.get("band_95") or {}
        lo45, hi45 = _sf(b45.get("low")), _sf(b45.get("high"))
        lo68, hi68 = _sf(b68.get("low")), _sf(b68.get("high"))
        lo95, hi95 = _sf(b95.get("low")), _sf(b95.get("high"))
        if lo45 is not None and hi45 is not None:
            strike_notes.append(
                f"Short strikes (IC body): inside {lo45:.0f}–{hi45:.0f} (45% expected range)"
            )
        if lo68 is not None and hi68 is not None:
            strike_notes.append(
                f"Long wing protection: beyond {lo68:.0f} / {hi68:.0f} (68% / 1σ)"
            )
        if lo95 is not None and hi95 is not None:
            strike_notes.append(
                f"Stress boundary: {lo95:.0f}–{hi95:.0f} (95% / 2σ) — size so max loss is tolerable here."
            )

    # ── Risk rationale ────────────────────────────────────────────────────────
    rationale = _risk_notes(daily_latest, regime)
    if not rationale:
        if trade_bias in ("bullish", "bearish"):
            rationale.append(f"Trending regime ({regime}): directional bias is {trade_bias}.")
        else:
            rationale.append("No strong directional signal; range/neutral bias.")

    return {
        "trade_bias":               trade_bias,
        "avoid_trading":            avoid,
        "prefer_spreads":           prefer_spreads,
        "mean_reversion_preferred": mean_reversion_preferred,
        "structure":                structure,
        "structure_reason":         structure_reason,
        "premium_environment":      premium_env,
        "iv_rv_ratio":              iv_rv_ratio,
        "strike_notes":             strike_notes,
        "rationale":                rationale,
    }


# ─── Main entry points ────────────────────────────────────────────────────────

def eod_run_one(cfg: AppConfig, ticker: str, day: str) -> Dict[str, Any]:
    daily_latest = read_json(cfg.path("features_daily_latest_dir") / f"{ticker}.json") or {}
    if daily_latest.get("ticker") != ticker:
        raise ValueError(f"Missing Step 8 daily_latest for {ticker}.")

    regime_latest = read_json(cfg.path("features_regime_labels_latest_dir") / f"{ticker}.json") or {}
    if regime_latest.get("ticker") != ticker:
        raise ValueError(f"Missing Step 9 regime_labels_latest for {ticker}.")

    # Use the as-of day in the data (avoids off-by-one on weekends/holidays)
    asof_day = str(daily_latest.get("day") or day)
    predicted_day = _compute_predicted_day(asof_day)

    # Best-effort IV from today's intraday insights (most accurate forward vol)
    iv = _read_iv_from_intraday(cfg, ticker)

    # Compute tomorrow range using all available vol signals
    tr = forecast_tomorrow_range_one(cfg, ticker=ticker, day=asof_day, iv=iv)

    last_close = _sf(daily_latest.get("last_price"))
    if last_close is None:
        raise ValueError(f"daily_latest.last_price missing for {ticker}.")

    # Key levels for reference
    lv = read_json(cfg.path("levels_dir") / f"{ticker}.json") or {}
    key_levels = _extract_key_levels(lv, last=last_close, topk=6)

    # Regime (reads correct keys after regime_labeler fix)
    regime = str(
        regime_latest.get("asof_regime")
        or regime_latest.get("regime")
        or "transition"
    ).lower()

    guidance = _generate_guidance(regime, daily_latest, iv=iv, tomorrow_range=tr.get("tomorrow_range"))

    # Pivot levels computed from prior-day (asof_day) OHLC
    pivot_levels = read_pivot_levels_for_day(cfg, ticker=ticker, asof_day=asof_day)

    payload: Dict[str, Any] = {
        "ticker":        ticker,
        "day":           day,           # CLI run day — used for output path
        "asof_day":      asof_day,      # actual data as-of day
        "predicted_day": predicted_day, # session this forecast targets
        "market_tz":     cfg.market_tz,
        "asof_ts":       daily_latest.get("asof_ts"),
        "today_close":   last_close,
        "iv":            iv,

        "regime_context": {
            "asof_label_day": regime_latest.get("asof_label_day") or regime_latest.get("day"),
            "asof_regime":    regime,
        },

        # inputs block mirrors what forecast_tomorrow_range_one uses (for transparency)
        "inputs": tr.get("inputs") or {},

        "tomorrow_range":     tr.get("tomorrow_range"),
        "tomorrow_key_levels": key_levels,
        "pivot_levels":       pivot_levels,
        "guidance":           guidance,

        "sources": {
            "daily_latest":         str((cfg.path("features_daily_latest_dir") / f"{ticker}.json").resolve()),
            "regime_labels_latest": str((cfg.path("features_regime_labels_latest_dir") / f"{ticker}.json").resolve()),
            "levels_latest":        str((cfg.path("levels_dir") / f"{ticker}.json").resolve()),
        },
    }
    return payload


def _write_eod_plan(cfg: AppConfig, day: str, ticker: str, payload: Dict[str, Any]) -> str:
    out_dir = cfg.path("outputs_eod_plans_dir") / day
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = (out_dir / f"{ticker}.json").resolve()
    mode = str((cfg.raw.get("i18n") or {}).get("mode", "en"))
    payload = attach_eod_views(payload, mode=mode)
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
            payload  = eod_run_one(cfg, ticker=str(t), day=day)
            out_json = _write_eod_plan(cfg, day=day, ticker=str(t), payload=payload)
            if log_path is not None:
                _append_jsonl(log_path, payload)
            out["results"][str(t)] = {"ticker": str(t), "day": day, "status": "ok", "out_json": out_json}
        except Exception as e:
            out["results"][str(t)] = {"ticker": str(t), "day": day, "status": "error", "error": str(e)}

    return out
