from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import traceback

from MarketAtlas.core.config import AppConfig
from MarketAtlas.orchestrators.ingest_intraday_5m import ingest_intraday_for_date
from MarketAtlas.orchestrators.update_session_state import update_session_state_all
from MarketAtlas.orchestrators.compute_levels import compute_levels_all
from MarketAtlas.orchestrators.intraday_features import (
    build_intraday_features_all,
    build_intraday_features_one,
)
from MarketAtlas.orchestrators.today_range import (
    nowcast_today_range_all,
    nowcast_today_range_one_ticker,
)
from MarketAtlas.orchestrators.intraday_insights import (
    intraday_insights_all,
    intraday_insights_one,
)


@dataclass
class IntradayRunResult:
    day: str
    status: str  # "ok" | "error"
    stages: Dict[str, Any]
    error: Optional[str] = None


def _stage_set(stages: Dict[str, Any], name: str, status: str, **payload: Any) -> None:
    """
    Helper to record stage status + payload consistently.
    """
    d = stages.get(name, {})
    d["status"] = status
    if payload:
        d.update(payload)
    stages[name] = d


def intraday_run(
    cfg: AppConfig,
    day: str,
    lookback_days: Optional[int] = None,
    include_extended_hours: bool = True,
    ticker: Optional[str] = None,
) -> IntradayRunResult:
    """
    Step 7: Intraday Orchestrator.

    Runs:
      1) ingest-5m
      2) session-state
      3) levels
      4) intraday-features
      5) today-range
      6) intraday-insights (final UI output)

    Strictly does NOT compute any EOD/tomorrow plan.
    """
    stages: Dict[str, Any] = {}

    try:
        # -------------------------
        # Stage 1: Ingest intraday 5m
        # -------------------------
        lb = lookback_days
        if lb is None:
            lb = int(cfg.raw.get("ingestion", {}).get("intraday_5m", {}).get("default_lookback_days", 5))

        _stage_set(stages, "ingest_5m", "running", lookback_days=lb, include_extended_hours=include_extended_hours)
        ingest_res = ingest_intraday_for_date(
            cfg,
            day=day,
            lookback_days=lb,
            include_extended_hours=include_extended_hours,
        )

        # ingest_res is dict[ticker] -> dataclass result (or plain dict)
        if isinstance(ingest_res, dict):
            ingest_payload = {}
            for k, v in ingest_res.items():
                if hasattr(v, "__dict__"):
                    ingest_payload[k] = vars(v)
                else:
                    ingest_payload[k] = v
        else:
            ingest_payload = ingest_res

        _stage_set(stages, "ingest_5m", "ok", result=ingest_payload)

        # -------------------------
        # Stage 2: Session state
        # -------------------------
        _stage_set(stages, "session_state", "running")
        ss_res = update_session_state_all(cfg, day=day)
        _stage_set(stages, "session_state", "ok", result=ss_res)

        # -------------------------
        # Stage 3: Levels
        # -------------------------
        _stage_set(stages, "levels", "running")
        lv_res = compute_levels_all(cfg, day=day)
        _stage_set(stages, "levels", "ok", result=lv_res)

        # -------------------------
        # Stage 4: Intraday features
        # -------------------------
        _stage_set(stages, "intraday_features", "running", ticker=ticker)
        if ticker:
            feat_res = build_intraday_features_one(cfg, ticker=ticker, day=day)
        else:
            feat_res = build_intraday_features_all(cfg, day=day)
        _stage_set(stages, "intraday_features", "ok", result=feat_res)

        # -------------------------
        # Stage 5: Today range
        # -------------------------
        _stage_set(stages, "today_range", "running", ticker=ticker)
        if ticker:
            tr_res = nowcast_today_range_one_ticker(cfg, ticker=ticker, day=day)
        else:
            tr_res = nowcast_today_range_all(cfg, day=day)
        _stage_set(stages, "today_range", "ok", result=tr_res)

        # -------------------------
        # Stage 6: Intraday insights (final outputs)
        # -------------------------
        _stage_set(stages, "intraday_insights", "running", ticker=ticker)
        if ticker:
            ins_res = intraday_insights_one(cfg, ticker=ticker, day=day)
        else:
            ins_res = intraday_insights_all(cfg, day=day)
        _stage_set(stages, "intraday_insights", "ok", result=ins_res)

        return IntradayRunResult(day=day, status="ok", stages=stages)

    except Exception:
        # Mark the orchestrator itself as failed, and include full stack trace
        _stage_set(stages, "intraday_run", "error")
        return IntradayRunResult(
            day=day,
            status="error",
            stages=stages,
            error=traceback.format_exc(),
        )
