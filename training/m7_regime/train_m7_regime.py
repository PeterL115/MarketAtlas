from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


from MarketAtlas.core.config import AppConfig
from MarketAtlas.core.io import ensure_dir, read_parquet
from MarketAtlas.core.jsonio import atomic_write_json, read_json


# -----------------------------
# Parameters
# -----------------------------
@dataclass
class TrainParams:
    min_rows: int = 200
    test_ratio: float = 0.20          # time-based split: last 20% as test
    max_iter: int = 2000
    C: float = 1.0
    random_state: int = 42


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if pd.isna(v):
            return None
        return v
    except Exception:
        return None


def _pick_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Use all numeric columns except obvious non-features.
    This makes Step 11 robust even if you add features later.
    """
    exclude = {
        "ts", "day", "open", "high", "low", "close", "volume",
        "adj close", "adj_close", "adjclose",
        "label", "regime", "regime_label",
    }
    cols = []
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(str(c))
    return cols


def _balanced_accuracy_no_warning(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> float:
    """
    Balanced accuracy without sklearn's 'y_pred contains classes not in y_true' warning.
    Computes mean recall over classes that appear in y_true (row_sum > 0).
    """
    lab_to_i = {lab: i for i, lab in enumerate(labels)}
    k = len(labels)
    cm = np.zeros((k, k), dtype=float)

    for yt, yp in zip(y_true, y_pred):
        if yt in lab_to_i and yp in lab_to_i:
            cm[lab_to_i[yt], lab_to_i[yp]] += 1.0

    recalls = []
    for i in range(k):
        row_sum = cm[i, :].sum()
        if row_sum > 0:
            recalls.append(cm[i, i] / row_sum)
    if not recalls:
        return float("nan")
    return float(np.mean(recalls))


def _load_daily_features(cfg: AppConfig, ticker: str, day: str) -> pd.DataFrame:
    p = cfg.path("features_daily_dir") / ticker / f"{day}.parquet"
    df = read_parquet(p)
    if df is None or df.empty:
        raise ValueError(f"Missing daily features parquet for {ticker} at {p}")
    if "day" not in df.columns:
        raise ValueError(f"Daily features missing 'day' column for {ticker}: {p}")
    return df.copy()


def _load_regime_labels(cfg: AppConfig, ticker: str, day: str) -> pd.DataFrame:
    p = cfg.path("features_regime_labels_dir") / ticker / f"{day}.parquet"
    df = read_parquet(p)
    if df is None or df.empty:
        raise ValueError(f"Missing regime labels parquet for {ticker} at {p}")
    # expected: columns include 'day' + 'regime' (or 'label')
    cols = {c.lower(): c for c in df.columns}
    if "day" not in cols:
        raise ValueError(f"Regime labels missing 'day' column for {ticker}: {p}")
    if "regime" not in cols and "label" not in cols and "regime_label" not in cols:
        raise ValueError(f"Regime labels missing regime column for {ticker}: cols={list(df.columns)}")
    return df.copy()


def _get_regime_col(df: pd.DataFrame) -> str:
    lc = {c.lower(): c for c in df.columns}
    for k in ("regime", "regime_label", "label"):
        if k in lc:
            return lc[k]
    raise ValueError("No regime column found.")


def _update_registry(cfg: AppConfig, entry: Dict[str, Any]) -> str:
    registry_path = (cfg.path("models_dir") / "registry.json").resolve()
    reg = read_json(registry_path) or {}

    key = str(entry["model_name"])
    reg[key] = entry

    atomic_write_json(reg, registry_path)
    return str(registry_path)


def train_one_ticker_regime(
    cfg: AppConfig,
    ticker: str,
    day: str,
    version: str,
    params: Optional[TrainParams] = None,
) -> Dict[str, Any]:
    p = params or TrainParams()

    df_feat = _load_daily_features(cfg, ticker=ticker, day=day)
    df_lab = _load_regime_labels(cfg, ticker=ticker, day=day)

    regime_col = _get_regime_col(df_lab)

    # Join on day
    df = df_feat.merge(df_lab[["day", regime_col]].rename(columns={regime_col: "regime"}), on="day", how="inner")
    df = df.dropna(subset=["regime"]).copy()

    # Ensure we only train up to requested day
    df = df[df["day"] <= str(day)].copy()
    if len(df) < p.min_rows:
        raise ValueError(f"Not enough rows to train {ticker} regime model (rows={len(df)} < {p.min_rows}).")

    feat_cols = _pick_feature_columns(df)
    if not feat_cols:
        raise ValueError(f"No numeric feature columns found for {ticker}. cols={list(df.columns)}")

    # Replace inf -> NaN, then drop columns that are entirely NaN
    df_featmat = df[feat_cols].replace([np.inf, -np.inf], np.nan).copy()
    feat_cols = [c for c in feat_cols if not df_featmat[c].isna().all()]
    if not feat_cols:
        raise ValueError(f"All candidate feature columns are NaN for {ticker} after cleaning.")

    X = df_featmat[feat_cols].astype(float).values
    y = df["regime"].astype(str).values


    # Time-based split (avoid leakage)
    n = len(df)
    n_test = max(1, int(round(n * p.test_ratio)))
    n_train = n - n_test
    if n_train < 50:
        raise ValueError(f"Not enough training rows after split for {ticker} (train_rows={n_train}).")

    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                multi_class="multinomial",
                max_iter=p.max_iter,
                C=p.C,
                n_jobs=None,
                random_state=p.random_state,
            )),
        ]
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Metrics (no sklearn warning)
    labels_all = sorted(list(set(y_train.tolist()) | set(y_test.tolist()) | set(y_pred.tolist())))
    acc = float((y_pred == y_test).mean())
    bacc = _balanced_accuracy_no_warning(y_test, y_pred, labels=labels_all)

    # Save
    model_name = f"{ticker}_regime"
    out_dir = (cfg.path("models_dir") / model_name / version).resolve()
    ensure_dir(out_dir)

    model_path = out_dir / "model.joblib"
    schema_path = out_dir / "feature_schema.json"
    metrics_path = out_dir / "metrics.json"

    dump(model, model_path)

    atomic_write_json(
        {
            "ticker": ticker,
            "model_name": model_name,
            "version": version,
            "feature_columns": feat_cols,
        },
        schema_path,
    )

    atomic_write_json(
        {
            "ticker": ticker,
            "model_name": model_name,
            "version": version,
            "rows_total": int(n),
            "rows_train": int(n_train),
            "rows_test": int(n_test),
            "accuracy": acc,
            "balanced_accuracy": bacc,
            "classes": labels_all,
            "asof_day_in_data": str(df["day"].iloc[-1]),
        },
        metrics_path,
    )

    registry_path = _update_registry(
        cfg,
        entry={
            "ticker": ticker,
            "model_name": model_name,
            "version": version,
            "model_path": str(model_path),
            "schema_path": str(schema_path),
            "metrics_path": str(metrics_path),
            "trained_asof_day": str(df["day"].iloc[-1]),
        },
    )

    return {
        "status": "ok",
        "ticker": ticker,
        "version": version,
        "model_path": str(model_path),
        "schema_path": str(schema_path),
        "metrics_path": str(metrics_path),
        "registry_path": registry_path,
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "asof_day_in_data": str(df["day"].iloc[-1]),
    }
