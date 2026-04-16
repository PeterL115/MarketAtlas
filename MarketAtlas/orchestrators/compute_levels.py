from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

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


def _resolve_tickers(cfg: AppConfig, tickers: Optional[Iterable[str]] = None) -> list[str]:
    if tickers is None:
        return list(cfg.intraday_assets.keys())
    want = [str(t).strip() for t in tickers if str(t).strip()]
    have = set(cfg.intraday_assets.keys())
    return [t for t in want if t in have]


def compute_levels_all(cfg: AppConfig, day: str, tickers: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """
    Compute support/resistance zones from daily bars + session state.

    - tickers=None  => all configured intraday assets
    - tickers=[...] => only those tickers (must exist in cfg.intraday_assets)
    """
    results: Dict[str, Any] = {}
    universe = _resolve_tickers(cfg, tickers=tickers)

    for ticker in universe:
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
