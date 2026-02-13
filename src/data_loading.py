from __future__ import annotations

import gc
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import psutil
except Exception:
    psutil = None


try:
    import src.config as cfg
except ImportError:
    class cfg:  # type: ignore
        BASE_DIR = Path.cwd()
        DATA_DIR = BASE_DIR / "data"
        MODELS_DIR = DATA_DIR / "models"
        REPORTS_DIR = BASE_DIR / "reports"
        LOGS_DIR = BASE_DIR / "logs"
        REQUIRED_PARAM_COLS = [
            "injection_pressure",
            "cycle_time",
            "cushion",
            "injection_time",
            "plasticizing_time",
            "cyl_tmp_z1",
            "cyl_tmp_z2",
            "cyl_tmp_z3",
            "cyl_tmp_z4",
            "melt_temp",
        ]

        @staticmethod
        def get_logger(
            name: str,
            log_file: Optional[Path] = None,
            level: int = logging.INFO,
        ) -> logging.Logger:
            logger = logging.getLogger(name)
            logger.setLevel(level)
            logger.propagate = False
            if not logger.handlers:
                fmt = logging.Formatter(
                    "%(asctime)s - [%(name)s] - %(levelname)s - %(message)s"
                )
                sh = logging.StreamHandler(sys.stdout)
                sh.setFormatter(fmt)
                logger.addHandler(sh)
                if log_file is not None:
                    log_file.parent.mkdir(parents=True, exist_ok=True)
                    fh = logging.FileHandler(log_file, encoding="utf-8")
                    fh.setFormatter(fmt)
                    logger.addHandler(fh)
            return logger


logger = cfg.get_logger("TE_DataLoader_DuckDB")

DEFAULT_PARQUET_PRIORITY = [
    "demo_ready_data.parquet",
    "optimized_scrap_data.parquet",
    "processed_full_dataset.parquet",
]


def _require_duckdb() -> Any:
    try:
        import duckdb  # type: ignore

        return duckdb
    except Exception as exc:
        raise ImportError(
            "DuckDB is required for the parquet-native data layer. "
            "Install it with: pip install duckdb"
        ) from exc


def _ram_mb() -> float:
    if psutil is None:
        return 0.0
    try:
        return float(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024))
    except Exception:
        return 0.0


def _log_perf(start_ts: float, start_ram_mb: float, event: str) -> None:
    elapsed = time.perf_counter() - start_ts
    peak_ram_mb = max(start_ram_mb, _ram_mb())
    logger.info("⏱️ DuckDB Query Time: %.2fs (%s)", elapsed, event)
    logger.info("📉 Peak RAM Usage: %.1f MB (%s)", peak_ram_mb, event)


def _duckdb_connect(memory_limit: str = "512MB", threads: int = 4) -> Any:
    duckdb = _require_duckdb()
    con = duckdb.connect(database=":memory:")
    con.execute(f"PRAGMA memory_limit='{memory_limit}'")
    con.execute(f"PRAGMA threads={max(1, int(threads))}")
    return con


def _sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    return "'" + str(value).replace("'", "''") + "'"


def _quote_ident(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _normalize_machine_id(series: pd.Series) -> pd.Series:
    ids = series.astype(str).str.extract(r"(\d+)")[0]
    ids = ids.fillna(series.astype(str))
    ids = ids.str.lstrip("0")
    ids = ids.where(ids != "", "0")
    return ids


def _resolve_parquet_source(parquet_source: Optional[str] = None) -> str:
    if parquet_source:
        path = Path(parquet_source)
        if path.exists():
            return path.as_posix()
        raise FileNotFoundError(f"Parquet source not found: {parquet_source}")

    for name in DEFAULT_PARQUET_PRIORITY:
        candidate = cfg.DATA_DIR / name
        if candidate.exists():
            return candidate.as_posix()

    wildcard = cfg.DATA_DIR / "*.parquet"
    matches = list(cfg.DATA_DIR.glob("*.parquet"))
    if matches:
        return wildcard.as_posix()

    raise FileNotFoundError(
        f"No parquet files found in {cfg.DATA_DIR}. "
        "Run optimizer first to create optimized parquet data."
    )


def _describe_parquet_columns(con: Any, parquet_source: str) -> List[str]:
    sql = (
        "DESCRIBE SELECT * FROM "
        f"read_parquet({_sql_literal(parquet_source)}, union_by_name=true)"
    )
    rows = con.execute(sql).fetchall()
    return [str(r[0]) for r in rows]


def _pick_column(schema_map: Dict[str, str], candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in schema_map:
            return schema_map[c]
    return None


def _build_where_clause(
    machine_id: Optional[str],
    start_time: Optional[str],
    end_time: Optional[str],
) -> str:
    clauses: List[str] = []
    if machine_id:
        clauses.append(f"CAST(machine_id AS VARCHAR) = {_sql_literal(machine_id)}")
    if start_time:
        start_ts = pd.to_datetime(start_time, errors="coerce")
        if pd.notna(start_ts):
            clauses.append(f"ts >= {_sql_literal(start_ts.isoformat())}")
    if end_time:
        end_ts = pd.to_datetime(end_time, errors="coerce")
        if pd.notna(end_ts):
            clauses.append(f"ts <= {_sql_literal(end_ts.isoformat())}")

    if not clauses:
        return ""
    return "WHERE " + " AND ".join(clauses)


def _build_wide_downsample_sql(
    parquet_source: str,
    schema_cols: List[str],
    machine_id: Optional[str],
    start_time: Optional[str],
    end_time: Optional[str],
    bucket_minutes: int,
    max_rows: Optional[int],
) -> str:
    schema_map = {c.lower(): c for c in schema_cols}

    machine_col = _pick_column(schema_map, ["machine_id", "machine", "machine_nr"])
    ts_col = _pick_column(schema_map, ["timestamp", "time", "event_time", "datetime"])

    if not machine_col or not ts_col:
        raise ValueError(
            "Parquet schema must contain machine_id and timestamp columns for DuckDB querying."
        )

    is_scrap_col = _pick_column(schema_map, ["is_scrap"])
    scrap_qty_col = _pick_column(schema_map, ["actual_scrap_qty", "scrap_quantity"])

    all_feature_candidates = [
        str(c).strip().lower() for c in getattr(cfg, "REQUIRED_PARAM_COLS", [])
    ]
    feature_cols = [schema_map[c] for c in all_feature_candidates if c in schema_map]
    if not feature_cols:
        fallback_drop = {machine_col, ts_col}
        if is_scrap_col:
            fallback_drop.add(is_scrap_col)
        if scrap_qty_col:
            fallback_drop.add(scrap_qty_col)
        feature_cols = [c for c in schema_cols if c not in fallback_drop][:25]

    if not feature_cols:
        raise ValueError("No feature columns available for wide parquet downsampling.")

    feature_cast_expr = ",\n            ".join(
        [f"TRY_CAST({_quote_ident(c)} AS DOUBLE) AS {_quote_ident(c)}" for c in feature_cols]
    )
    feature_avg_expr = ",\n            ".join(
        [f"AVG({_quote_ident(c)}) AS {_quote_ident(c)}" for c in feature_cols]
    )

    is_scrap_expr = (
        f"TRY_CAST({_quote_ident(is_scrap_col)} AS INTEGER)"
        if is_scrap_col
        else "NULL"
    )
    scrap_qty_expr = (
        f"TRY_CAST({_quote_ident(scrap_qty_col)} AS DOUBLE)"
        if scrap_qty_col
        else "NULL"
    )
    where_clause = _build_where_clause(machine_id, start_time, end_time)
    limit_clause = f"\nLIMIT {int(max_rows)}" if max_rows is not None else ""

    return f"""
WITH base AS (
    SELECT
        CAST({_quote_ident(machine_col)} AS VARCHAR) AS machine_id,
        TRY_CAST({_quote_ident(ts_col)} AS TIMESTAMP) AS ts,
        {is_scrap_expr} AS is_scrap,
        {scrap_qty_expr} AS actual_scrap_qty,
        {feature_cast_expr}
    FROM read_parquet({_sql_literal(parquet_source)}, union_by_name=true)
),
filtered AS (
    SELECT *
    FROM base
    WHERE ts IS NOT NULL
),
bucketed AS (
    SELECT
        machine_id,
        time_bucket(INTERVAL '{int(bucket_minutes)} minute', ts) AS timestamp,
        MAX(
            COALESCE(
                is_scrap,
                CASE WHEN COALESCE(actual_scrap_qty, 0) > 0 THEN 1 ELSE 0 END
            )
        ) AS is_scrap,
        AVG(actual_scrap_qty) AS actual_scrap_qty,
        {feature_avg_expr}
    FROM filtered
    {where_clause}
    GROUP BY 1, 2
)
SELECT *
FROM bucketed
ORDER BY timestamp DESC
{limit_clause}
"""


def _build_long_downsample_sql(
    parquet_source: str,
    schema_cols: List[str],
    machine_id: Optional[str],
    start_time: Optional[str],
    end_time: Optional[str],
    bucket_minutes: int,
    max_rows: Optional[int],
) -> str:
    schema_map = {c.lower(): c for c in schema_cols}

    machine_col = _pick_column(schema_map, ["machine_id", "machine", "machine_nr"])
    ts_col = _pick_column(schema_map, ["timestamp", "time", "event_time", "datetime"])
    variable_col = _pick_column(schema_map, ["variable_name", "sensor", "parameter", "param"])
    value_col = _pick_column(schema_map, ["value", "sensor_value", "reading"])

    if not machine_col or not ts_col or not variable_col or not value_col:
        raise ValueError(
            "Long parquet schema must contain machine_id, timestamp, variable_name, and value."
        )

    is_scrap_col = _pick_column(schema_map, ["is_scrap"])
    scrap_qty_col = _pick_column(schema_map, ["actual_scrap_qty", "scrap_quantity"])

    sensor_cols = [str(c).strip().lower() for c in getattr(cfg, "REQUIRED_PARAM_COLS", [])]
    if not sensor_cols:
        sensor_cols = [
            "injection_pressure",
            "cycle_time",
            "cushion",
            "injection_time",
            "plasticizing_time",
            "cyl_tmp_z1",
            "melt_temp",
        ]

    sensor_filter = ", ".join(_sql_literal(c) for c in sensor_cols)
    pivot_expr = ",\n        ".join(
        [
            f"AVG(value_avg) FILTER (WHERE variable_name = {_sql_literal(c)}) AS {_quote_ident(c)}"
            for c in sensor_cols
        ]
    )

    is_scrap_expr = (
        f"TRY_CAST({_quote_ident(is_scrap_col)} AS INTEGER)"
        if is_scrap_col
        else "NULL"
    )
    scrap_qty_expr = (
        f"TRY_CAST({_quote_ident(scrap_qty_col)} AS DOUBLE)"
        if scrap_qty_col
        else "NULL"
    )
    where_clause = _build_where_clause(machine_id, start_time, end_time)
    limit_clause = f"\nLIMIT {int(max_rows)}" if max_rows is not None else ""

    return f"""
WITH base AS (
    SELECT
        CAST({_quote_ident(machine_col)} AS VARCHAR) AS machine_id,
        TRY_CAST({_quote_ident(ts_col)} AS TIMESTAMP) AS ts,
        LOWER(TRIM(CAST({_quote_ident(variable_col)} AS VARCHAR))) AS variable_name,
        TRY_CAST({_quote_ident(value_col)} AS DOUBLE) AS value,
        {is_scrap_expr} AS is_scrap,
        {scrap_qty_expr} AS actual_scrap_qty
    FROM read_parquet({_sql_literal(parquet_source)}, union_by_name=true)
),
filtered AS (
    SELECT *
    FROM base
    WHERE ts IS NOT NULL
      AND variable_name IN ({sensor_filter})
      AND value IS NOT NULL
),
bucketed AS (
    SELECT
        machine_id,
        time_bucket(INTERVAL '{int(bucket_minutes)} minute', ts) AS timestamp,
        variable_name,
        AVG(value) AS value_avg,
        MAX(
            COALESCE(
                is_scrap,
                CASE WHEN COALESCE(actual_scrap_qty, 0) > 0 THEN 1 ELSE 0 END
            )
        ) AS is_scrap,
        AVG(actual_scrap_qty) AS actual_scrap_qty
    FROM filtered
    {where_clause}
    GROUP BY 1, 2, 3
),
wide AS (
    SELECT
        machine_id,
        timestamp,
        MAX(is_scrap) AS is_scrap,
        AVG(actual_scrap_qty) AS actual_scrap_qty,
        {pivot_expr}
    FROM bucketed
    GROUP BY 1, 2
)
SELECT *
FROM wide
ORDER BY timestamp DESC
{limit_clause}
"""


def _build_downsampled_query(
    parquet_source: str,
    schema_cols: List[str],
    machine_id: Optional[str],
    start_time: Optional[str],
    end_time: Optional[str],
    bucket_minutes: int,
    max_rows: Optional[int],
) -> str:
    schema_map = {c.lower(): c for c in schema_cols}
    has_long = (
        _pick_column(schema_map, ["variable_name", "sensor", "parameter", "param"]) is not None
        and _pick_column(schema_map, ["value", "sensor_value", "reading"]) is not None
    )
    if has_long:
        return _build_long_downsample_sql(
            parquet_source=parquet_source,
            schema_cols=schema_cols,
            machine_id=machine_id,
            start_time=start_time,
            end_time=end_time,
            bucket_minutes=bucket_minutes,
            max_rows=max_rows,
        )
    return _build_wide_downsample_sql(
        parquet_source=parquet_source,
        schema_cols=schema_cols,
        machine_id=machine_id,
        start_time=start_time,
        end_time=end_time,
        bucket_minutes=bucket_minutes,
        max_rows=max_rows,
    )


def _postprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if "machine_id" in df.columns:
        df["machine_id"] = _normalize_machine_id(df["machine_id"]).astype(str)
    if "is_scrap" in df.columns:
        df["is_scrap"] = pd.to_numeric(df["is_scrap"], errors="coerce").fillna(0).astype(int)
    else:
        df["is_scrap"] = 0

    float_cols = df.select_dtypes(include=["float64"]).columns
    if len(float_cols) > 0:
        df[float_cols] = df[float_cols].astype(np.float32)

    return df


def query_sensor_data(
    machine_id: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    *,
    bucket_minutes: int = 1,
    max_rows: Optional[int] = None,
    parquet_source: Optional[str] = None,
) -> pd.DataFrame:
    """
    Query parquet data through DuckDB and return only filtered data to pandas.
    This keeps RAM bounded by executing filtering, downsampling, and pivot in SQL.
    """
    query_start = time.perf_counter()
    start_ram = _ram_mb()
    parquet = _resolve_parquet_source(parquet_source)

    con = _duckdb_connect(memory_limit="512MB", threads=4)
    try:
        schema_cols = _describe_parquet_columns(con, parquet)
        sql = _build_downsampled_query(
            parquet_source=parquet,
            schema_cols=schema_cols,
            machine_id=machine_id,
            start_time=start_time,
            end_time=end_time,
            bucket_minutes=bucket_minutes,
            max_rows=max_rows,
        )
        df = con.execute(sql).fetch_df()
        df = _postprocess_df(df)
        _log_perf(query_start, start_ram, "query_sensor_data")
        return df
    finally:
        con.close()
        gc.collect()


def fetch_balanced_training_sample(
    *,
    target_total_rows: int = 500_000,
    random_state: int = 42,
    batch_rows: int = 100_000,
    bucket_minutes: int = 1,
    parquet_source: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Pull a balanced 50/50 sample directly from parquet via DuckDB.
    Uses record-batch streaming to avoid loading all rows at once.
    """
    query_start = time.perf_counter()
    start_ram = _ram_mb()
    parquet = _resolve_parquet_source(parquet_source)

    con = _duckdb_connect(memory_limit="1GB", threads=4)
    try:
        schema_cols = _describe_parquet_columns(con, parquet)
        base_sql = _build_downsampled_query(
            parquet_source=parquet,
            schema_cols=schema_cols,
            machine_id=None,
            start_time=None,
            end_time=None,
            bucket_minutes=bucket_minutes,
            max_rows=None,
        )

        count_sql = f"""
WITH base AS ({base_sql})
SELECT
    COUNT(*) AS total_rows,
    SUM(CASE WHEN COALESCE(is_scrap, 0) = 1 THEN 1 ELSE 0 END) AS pos_rows,
    SUM(CASE WHEN COALESCE(is_scrap, 0) = 0 THEN 1 ELSE 0 END) AS neg_rows
FROM base
"""
        total_rows, pos_rows, neg_rows = con.execute(count_sql).fetchone()
        total_rows = int(total_rows or 0)
        pos_rows = int(pos_rows or 0)
        neg_rows = int(neg_rows or 0)

        if total_rows == 0:
            _log_perf(query_start, start_ram, "fetch_balanced_training_sample")
            return pd.DataFrame(), {
                "total_rows": 0,
                "positive_rows": 0,
                "negative_rows": 0,
                "query_time_s": round(time.perf_counter() - query_start, 3),
                "peak_ram_mb": round(max(start_ram, _ram_mb()), 2),
            }

        per_class = min(int(target_total_rows // 2), pos_rows, neg_rows)
        if per_class > 0:
            sample_sql = f"""
WITH base AS ({base_sql}),
pos AS (
    SELECT *
    FROM base
    WHERE COALESCE(is_scrap, 0) = 1
    USING SAMPLE reservoir({per_class}) REPEATABLE ({int(random_state)})
),
neg AS (
    SELECT *
    FROM base
    WHERE COALESCE(is_scrap, 0) = 0
    USING SAMPLE reservoir({per_class}) REPEATABLE ({int(random_state) + 1})
)
SELECT * FROM pos
UNION ALL
SELECT * FROM neg
"""
        else:
            fallback_rows = min(int(target_total_rows), total_rows)
            sample_sql = f"""
WITH base AS ({base_sql})
SELECT *
FROM base
USING SAMPLE reservoir({fallback_rows}) REPEATABLE ({int(random_state)})
"""

        chunks: List[pd.DataFrame] = []
        try:
            batch_reader = con.execute(sample_sql).fetch_record_batch(rows_per_batch=int(batch_rows))
            for batch in batch_reader:
                chunks.append(batch.to_pandas())
        except Exception:
            chunks.append(con.execute(sample_sql).fetch_df())

        sample_df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        sample_df = _postprocess_df(sample_df)

        pos_after = int((sample_df["is_scrap"] == 1).sum()) if "is_scrap" in sample_df.columns else 0
        neg_after = int((sample_df["is_scrap"] == 0).sum()) if "is_scrap" in sample_df.columns else 0

        _log_perf(query_start, start_ram, "fetch_balanced_training_sample")
        return sample_df, {
            "total_rows": total_rows,
            "positive_rows": pos_rows,
            "negative_rows": neg_rows,
            "sample_rows": int(len(sample_df)),
            "sample_positive_rows": pos_after,
            "sample_negative_rows": neg_after,
            "query_time_s": round(time.perf_counter() - query_start, 3),
            "peak_ram_mb": round(max(start_ram, _ram_mb()), 2),
        }
    finally:
        con.close()
        gc.collect()


def load_and_merge_data(use_cache: bool = True) -> pd.DataFrame:
    """
    Backward-compatible entrypoint for app/reporting:
    - uses DuckDB parquet scan
    - performs SQL downsampling + pivot before pandas
    - caches a compact pandas frame for dashboard startup
    """
    cache_path = cfg.DATA_DIR / "processed_full_dataset.pkl"
    query_start = time.perf_counter()
    start_ram = _ram_mb()

    if use_cache and cache_path.exists():
        try:
            file_age = time.time() - cache_path.stat().st_mtime
            if file_age < 12 * 3600:
                logger.info("⚡ Loading Data from Cache (DuckDB layer)...")
                df_cache = pd.read_pickle(cache_path)
                _log_perf(query_start, start_ram, "load_and_merge_data(cache)")
                return df_cache
        except Exception as exc:
            logger.warning("Cache read failed (%s). Re-querying parquet.", exc)

    df = query_sensor_data(
        machine_id=None,
        start_time=None,
        end_time=None,
        bucket_minutes=1,
        max_rows=1_000_000,
    )

    if df.empty:
        _log_perf(query_start, start_ram, "load_and_merge_data(empty)")
        return df

    if "actual_scrap_qty" in df.columns:
        scrap_qty = pd.to_numeric(df["actual_scrap_qty"], errors="coerce").fillna(0)
        df["scrap_rate"] = np.where(scrap_qty > 0, 1.0, 0.0).astype(np.float32)
    else:
        df["scrap_rate"] = pd.to_numeric(df["is_scrap"], errors="coerce").fillna(0).astype(np.float32)

    if "plant_shift_date" not in df.columns:
        df["plant_shift_date"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.date
    if "machine_status_name" not in df.columns:
        df["machine_status_name"] = "Unknown"
    if "segment_abbr_name" not in df.columns:
        df["segment_abbr_name"] = "Unknown"
    if "manufacturing_plant_name" not in df.columns:
        df["manufacturing_plant_name"] = "TE"

    if use_cache:
        try:
            df.to_pickle(cache_path)
        except Exception as exc:
            logger.warning("Could not save cache (%s).", exc)

    _log_perf(query_start, start_ram, "load_and_merge_data(query)")
    return df


def load_param_data() -> pd.DataFrame:
    """Compatibility wrapper for legacy callers."""
    return query_sensor_data(max_rows=1_000_000)


def load_hydra_data() -> pd.DataFrame:
    """
    Hydra stream is now represented in parquet-native output.
    Keep a minimal compatibility frame for older call sites.
    """
    df = load_and_merge_data(use_cache=True)
    cols = [
        c
        for c in [
            "machine_id",
            "timestamp",
            "plant_shift_date",
            "machine_status_name",
            "segment_abbr_name",
            "manufacturing_plant_name",
        ]
        if c in df.columns
    ]
    return df[cols].copy() if cols else pd.DataFrame()


if __name__ == "__main__":
    preview_df = query_sensor_data(max_rows=20_000)
    print(preview_df.head())
    print(preview_df.shape)
