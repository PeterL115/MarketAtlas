from __future__ import annotations

from typing import Any, Dict, Optional

from MarketAtlas.core.config import AppConfig
from MarketAtlas.training.spx_regime.train_spx_regime import train_spx_regime, TrainParams


def train_spx_regime_model(cfg: AppConfig, day: str, version: str = "v1") -> Dict[str, Any]:
    # Keep params minimal for now; tune later in Step 15.
    params = TrainParams()
    return train_spx_regime(cfg, day=day, version=version, params=params)
