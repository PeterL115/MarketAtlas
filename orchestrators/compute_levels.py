from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from ..core.config import AppConfig
from ..core.levels import compute_levels, write_levels


@dataclass(frozen=True)
class LevelsRunResult:
    ticker: str
    day: str
    status: str
    latest_path: str | None
    snapshot_path: str | None
    level_count: int


def compute_levels_all(cfg: AppConfig, day: str) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for ticker in cfg.intraday_assets.keys():
        try:
            payload = compute_levels(cfg, ticker=ticker, day=day)
            latest, snap = write_levels(cfg, ticker=ticker, day=day, payload=payload)
            results[ticker] = LevelsRunResult(
                ticker=ticker,
                day=day,
                status="ok",
                latest_path=latest,
                snapshot_path=snap,
                level_count=int(len(payload.get("levels", []))),
            )
        except Exception as e:
            results[ticker] = LevelsRunResult(
                ticker=ticker,
                day=day,
                status=f"error: {e}",
                latest_path=None,
                snapshot_path=None,
                level_count=0,
            )
    return {k: vars(v) for k, v in results.items()}
