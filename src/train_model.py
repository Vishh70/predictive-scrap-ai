from __future__ import annotations

import gc
import json
import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import psutil
except Exception:
    psutil = None


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import src.config as cfg
    from src.data_loading import fetch_balanced_training_sample
except ImportError as exc:
    raise ImportError("Could not import training dependencies from src package.") from exc


logger = cfg.get_logger(
    "TE_Trainer_DuckDB",
    log_file=cfg.LOGS_DIR / "training_trace.log",
    level=logging.INFO,
)


def _ram_mb() -> float:
    if psutil is None:
        return 0.0
    try:
        return float(psutil.Process().memory_info().rss / (1024 * 1024))
    except Exception:
        return 0.0


def _artifact_dirs() -> Tuple[Path, Path, Path]:
    model_dir_primary = PROJECT_ROOT / "models"
    model_dir_mirror = cfg.MODELS_DIR
    reports_dir = cfg.DATA_DIR / "reports"

    model_dir_primary.mkdir(parents=True, exist_ok=True)
    model_dir_mirror.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    return model_dir_primary, model_dir_mirror, reports_dir


def _save_to_primary_and_mirror(
    name: str, obj: Any, primary_dir: Path, mirror_dir: Path
) -> None:
    primary_path = primary_dir / name
    mirror_path = mirror_dir / name
    joblib.dump(obj, primary_path)
    if primary_path != mirror_path:
        shutil.copy2(primary_path, mirror_path)


def _inject_synthetic_scrap(
    y: pd.Series, random_state: int
) -> Tuple[pd.Series, bool, int]:
    total_scrap = int(y.sum())
    if total_scrap >= 10:
        return y, False, 0

    zero_idx = y[y == 0].index.to_numpy()
    if len(zero_idx) == 0:
        return y, False, 0

    n_synthetic = min(max(1, int(len(y) * 0.01)), len(zero_idx))
    rng = np.random.default_rng(seed=random_state)
    synthetic_idx = rng.choice(zero_idx, size=n_synthetic, replace=False)
    y.loc[synthetic_idx] = 1
    return y, True, int(n_synthetic)


def _select_training_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    lower_name_map = {c: c.lower() for c in df.columns}
    explicit_drop = {
        "timestamp",
        "actual_scrap_qty",
        "is_scrap",
        "machine_id",
        "plant_shift_date",
        "machine_status_name",
        "segment_abbr_name",
        "manufacturing_plant_name",
    }
    drop_cols = [c for c, cl in lower_name_map.items() if cl in explicit_drop]
    X = df.drop(columns=drop_cols, errors="ignore").copy()

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.dropna(axis=1, how="all")
    feature_names = X.columns.tolist()
    if not feature_names:
        raise ValueError("No numeric training features after preprocessing.")
    return X, feature_names


def train_scrap_model(
    random_state: int = 42,
    test_size: float = 0.2,
    target_total_rows: int = 500_000,
) -> Dict[str, Any]:
    """
    Big-data training pipeline:
    - data pulled from parquet via DuckDB
    - balanced sample selected in storage layer (50/50 where possible)
    - memory-safe model training with single-process RF
    """
    total_start = time.perf_counter()
    start_ram = _ram_mb()

    logger.info("Starting DuckDB + Parquet training pipeline.")
    df, sample_meta = fetch_balanced_training_sample(
        target_total_rows=target_total_rows,
        random_state=random_state,
        batch_rows=100_000,
        bucket_minutes=1,
    )
    if df.empty:
        raise ValueError("DuckDB returned empty training dataset.")

    if "is_scrap" not in df.columns:
        df["is_scrap"] = 0
    y = pd.to_numeric(df["is_scrap"], errors="coerce").fillna(0).astype(int)
    y = (y > 0).astype(int)

    y, injected, injected_count = _inject_synthetic_scrap(y, random_state=random_state)
    df["is_scrap"] = y

    X, feature_names = _select_training_features(df)
    del df
    gc.collect()

    stratify_target = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        class_weight="balanced_subsample",
        n_jobs=1,
        random_state=random_state,
    )
    rf.fit(X_train_scaled, y_train)

    y_pred = rf.predict(X_test_scaled)
    y_prob = rf.predict_proba(X_test_scaled)[:, 1]

    precision_1, recall_1, f1_1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=[1], zero_division=0
    )
    auc_roc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else float("nan")
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    feature_importance_map = dict(
        sorted(
            zip(feature_names, rf.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        )
    )
    rf.training_features_ = feature_names
    rf.feature_importances_map_ = feature_importance_map

    model_dir_primary, model_dir_mirror, reports_dir = _artifact_dirs()
    _save_to_primary_and_mirror("scrap_model.joblib", rf, model_dir_primary, model_dir_mirror)
    _save_to_primary_and_mirror("model_features.joblib", feature_names, model_dir_primary, model_dir_mirror)
    _save_to_primary_and_mirror("imputer.joblib", imputer, model_dir_primary, model_dir_mirror)
    _save_to_primary_and_mirror("scaler.joblib", scaler, model_dir_primary, model_dir_mirror)
    _save_to_primary_and_mirror("threshold.joblib", 0.5, model_dir_primary, model_dir_mirror)

    feature_importance_df = pd.DataFrame(
        {
            "feature": list(feature_importance_map.keys()),
            "importance": list(feature_importance_map.values()),
        }
    )
    feature_importance_df.to_csv(reports_dir / "feature_importance.csv", index=False)

    train_elapsed = time.perf_counter() - total_start
    peak_ram_mb = max(start_ram, _ram_mb())
    logger.info("⏱️ DuckDB Query Time: %.2fs", float(sample_meta.get("query_time_s", 0.0)))
    logger.info("📉 Peak RAM Usage: %.1f MB", peak_ram_mb)
    logger.info("Training complete in %.2fs", train_elapsed)

    metadata = {
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_source": "DuckDB + Parquet",
        "query_stats": sample_meta,
        "final_training_row_count": int(len(X_train)),
        "holdout_row_count": int(len(X_test)),
        "synthetic_injection_applied": bool(injected),
        "synthetic_injected_count": int(injected_count),
        "holdout_metrics_class_1": {
            "precision": float(precision_1[0]),
            "recall": float(recall_1[0]),
            "f1": float(f1_1[0]),
            "roc_auc": float(auc_roc),
            "tp": int(tp),
            "fn": int(fn),
            "fp": int(fp),
            "tn": int(tn),
        },
        "model_params": rf.get_params(),
        "training_time_s": float(round(train_elapsed, 3)),
        "peak_ram_mb": float(round(peak_ram_mb, 2)),
    }
    rf.training_metadata_ = metadata

    with open(reports_dir / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, default=str)

    logger.info("Saved model to %s", model_dir_primary / "scrap_model.joblib")
    logger.info("Mirrored model to %s", model_dir_mirror / "scrap_model.joblib")
    return metadata


class ScrapPredictionTrainer:
    """
    Compatibility wrapper used by main_engine.py.
    """

    def __init__(
        self,
        random_state: int = 42,
        test_size: float = 0.2,
        target_total_rows: int = 500_000,
    ) -> None:
        self.random_state = random_state
        self.test_size = test_size
        self.target_total_rows = target_total_rows

    def train_pipeline(self) -> Dict[str, Any]:
        return train_scrap_model(
            random_state=self.random_state,
            test_size=self.test_size,
            target_total_rows=self.target_total_rows,
        )


if __name__ == "__main__":
    train_scrap_model()
