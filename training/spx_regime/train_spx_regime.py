from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix

from MarketAtlas.core.config import AppConfig
from MarketAtlas.core.io import read_parquet, ensure_dir
from MarketAtlas.core.jsonio import atomic_write_json, read_json


@dataclass
class TrainParams:
    min_rows: int = 300
    val_frac: float = 0.20
    random_state: int = 42

    # Model hyperparams
    max_iter: int = 800
    c: float = 1.0


def _safe_int(x: Any, default: int) -> int:
    try:
        v = int(x)
        return v
    except Exception:
        return default


def _safe_float(x: Any, default: float) -> float:
    try:
        v = float(x)
        return v
    except Exception:
        return default


def _select_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Select numeric feature columns from Step 8 daily features.
    Exclude identity columns and raw OHLC columns to reduce leakage/scale issues.
    Keep engineered columns like atr/rv/er/range_atr/gap_atr/slope_close/pos_hhll/vol_z etc.
    """
    exclude = {
        "ts", "day",
        "open", "high", "low", "close", "volume",
    }
    cols = []
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in exclude:
            continue
        # keep numeric only
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    if not cols:
        raise ValueError("No numeric feature columns found after filtering.")
    return cols


def _load_step8_step9(cfg: AppConfig, ticker: str, day: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    feat_path = cfg.path("features_daily_dir") / ticker / f"{day}.parquet"
    lab_path = cfg.path("regime_labels_dir") / ticker / f"{day}.parquet"

    df_feat = read_parquet(feat_path)
    if df_feat is None or df_feat.empty:
        raise ValueError(f"Missing Step 8 daily features: {feat_path}")

    df_lab = read_parquet(lab_path)
    if df_lab is None or df_lab.empty:
        raise ValueError(f"Missing Step 9 labels: {lab_path}")

    if "day" not in df_feat.columns:
        raise ValueError("Step 8 daily features missing 'day'.")
    if "day" not in df_lab.columns or "regime" not in df_lab.columns:
        raise ValueError("Step 9 labels missing 'day' or 'regime'.")

    df_feat["day"] = df_feat["day"].astype(str)
    df_lab["day"] = df_lab["day"].astype(str)

    return df_feat, df_lab


def _time_holdout_split(df: pd.DataFrame, val_frac: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("day").reset_index(drop=True)
    n = len(df)
    n_val = max(1, int(round(n * val_frac)))
    n_train = max(1, n - n_val)
    return df.iloc[:n_train].copy(), df.iloc[n_train:].copy()


def _ensure_json(path: Path, obj: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    atomic_write_json(obj, path)


def _update_registry(registry_path: Path, record: Dict[str, Any]) -> None:
    reg = read_json(registry_path) or {}
    reg.setdefault("models", {})
    reg["models"]["spx_regime"] = record
    _ensure_json(registry_path, reg)


def train_spx_regime(cfg: AppConfig, day: str, version: str = "v1", params: Optional[TrainParams] = None) -> Dict[str, Any]:
    p = params or TrainParams()

    ticker = "SPX"
    df_feat, df_lab = _load_step8_step9(cfg, ticker=ticker, day=day)

    # Join on day
    df = df_feat.merge(df_lab[["day", "regime"]], on="day", how="inner")
    df = df.sort_values("day").reset_index(drop=True)

    if len(df) < p.min_rows:
        raise ValueError(f"Not enough labeled rows to train (rows={len(df)} < {p.min_rows}).")

    feature_cols = _select_feature_columns(df)

    X = df[feature_cols].copy()
    y = df["regime"].astype(str).copy()

    # Drop rows where y is missing
    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]

    # Holdout split by time
    df_xy = pd.concat([df[["day"]], X, y.rename("regime")], axis=1).dropna(subset=["regime"])
    train_df, val_df = _time_holdout_split(df_xy, val_frac=p.val_frac)

    X_train = train_df[feature_cols]
    y_train = train_df["regime"]
    X_val = val_df[feature_cols]
    y_val = val_df["regime"]

    # Pipeline: impute -> scale -> multinomial logistic regression
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "clf",
                LogisticRegression(
                    multi_class="multinomial",
                    solver="lbfgs",
                    max_iter=p.max_iter,
                    C=p.c,
                    class_weight="balanced",
                    random_state=p.random_state,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    yhat = model.predict(X_val)
    acc = float(accuracy_score(y_val, yhat))
    bacc = float(balanced_accuracy_score(y_val, yhat))
    labels = sorted(list(pd.unique(y)))

    cm = confusion_matrix(y_val, yhat, labels=labels).tolist()
    report = classification_report(y_val, yhat, labels=labels, output_dict=True, zero_division=0)

    # Paths
    models_dir = cfg.path("models_dir")
    out_dir = (models_dir / "spx_regime" / version).resolve()
    ensure_dir(out_dir)

    model_path = out_dir / "model.joblib"
    schema_path = out_dir / "feature_schema.json"
    metrics_path = out_dir / "metrics.json"
    registry_path = (models_dir / "registry.json").resolve()

    dump(model, model_path)

    schema = {
        "ticker": ticker,
        "version": version,
        "feature_cols": feature_cols,
        "label_space": labels,
        "train_range": {"start_day": str(train_df["day"].iloc[0]), "end_day": str(train_df["day"].iloc[-1])},
        "val_range": {"start_day": str(val_df["day"].iloc[0]), "end_day": str(val_df["day"].iloc[-1])},
    }
    _ensure_json(schema_path, schema)

    metrics = {
        "ticker": ticker,
        "version": version,
        "rows_total": int(len(df_xy)),
        "rows_train": int(len(train_df)),
        "rows_val": int(len(val_df)),
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "confusion_matrix": {"labels": labels, "matrix": cm},
        "classification_report": report,
        "asof_day_requested": str(day),
        "asof_day_in_data": str(df_xy["day"].iloc[-1]),
    }
    _ensure_json(metrics_path, metrics)

    registry_record = {
        "latest": version,
        "model_path": str(model_path),
        "schema_path": str(schema_path),
        "metrics_path": str(metrics_path),
        "trained_on": {"ticker": ticker, "asof_day_in_data": str(df_xy["day"].iloc[-1])},
    }
    _update_registry(registry_path, registry_record)

    return {
        "status": "ok",
        "ticker": ticker,
        "version": version,
        "model_path": str(model_path),
        "schema_path": str(schema_path),
        "metrics_path": str(metrics_path),
        "registry_path": str(registry_path),
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "asof_day_in_data": str(df_xy["day"].iloc[-1]),
    }
