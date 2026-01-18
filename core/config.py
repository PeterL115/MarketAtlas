from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass(frozen=True)
class AppConfig:
    raw: Dict[str, Any]
    root_dir: Path

    @property
    def market_tz(self) -> str:
        return str(self.raw.get("timezone_market", "America/New_York"))

    @property
    def provider_name(self) -> str:
        return str(self.raw.get("provider", {}).get("name", "yfinance"))

    @property
    def daily_assets(self) -> Dict[str, str]:
        return dict(self.raw.get("assets", {}).get("daily", {}))

    @property
    def intraday_assets(self) -> Dict[str, str]:
        return dict(self.raw.get("assets", {}).get("intraday_5m", {}))

    def path(self, key: str) -> Path:
        """
        Resolve a path from config relative to project root.
        """
        val = self.raw.get(key)
        if not val:
            raise KeyError(f"Missing config key: {key}")
        return (self.root_dir / str(val)).resolve()


def load_config(project_root: str | Path, rel_path: str = "config/app.yaml") -> AppConfig:
    root = Path(project_root).resolve()
    cfg_path = (root / rel_path).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return AppConfig(raw=raw, root_dir=root)
