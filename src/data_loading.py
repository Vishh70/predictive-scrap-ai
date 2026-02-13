import pandas as pd
import numpy as np
import os
import re
import time
import logging
import warnings
import gc
from pathlib import Path
from datetime import datetime
import csv
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# =========================================================
# CONFIGURATION & SETUP
# =========================================================

# Internal Config Import (Robust Fallback)
try:
    import src.config as cfg
except ImportError:
    # Fallback config if running standalone or in testing
    class cfg:
        BASE_DIR = Path.cwd()
        DATA_DIR = Path.cwd() / 'data'
        LOGS_DIR = BASE_DIR / 'logs'
        # UPDATED: Common columns from AI_cup_parameter_info
        REQUIRED_PARAM_COLS = [
        # --- Critical Quality Sensors (canonical simplified names) ---
        'injection_pressure',
        'cycle_time',
        'cushion',
        'injection_time',
        'plasticizing_time',
        'max_injection_speed',
        'transfer_pressure',
        # --- Temperature Zones ---
        'cyl_tmp_z1',
        'cyl_tmp_z2',
        'cyl_tmp_z3',
        'cyl_tmp_z4',
        'flange_temperature',
        'nozzle_temperature',

        # --- Mold Safety ---
        'mold_protection_force_peak',
        'clamping_force_peak'
    ]

        # Keep your file patterns
        HYDRA_PATTERN = "*Hydra*.csv"
        PARAM_PATTERN = "*.csv"  # Scan all CSVs; parameter validity is detected from file content
        PHYSICS_RANGES = {
            'injection_pressure': (0, 2000),
            'melt_temp': (100, 400),
            'cycle_time': (0, 120),
        }
        MIN_VALID_CYCLE_TIME_SEC = 1.0
        SLOPE_LAG_WINDOW = 5
        PARAM_SATURATION_RATIO = 0.995
        
        @staticmethod
        def get_logger(name: str, log_file: Optional[Union[str, Path]] = None, level: int = logging.INFO) -> logging.Logger:
            logger = logging.getLogger(name)
            logger.setLevel(level)
            logger.propagate = False
            formatter = logging.Formatter("%(asctime)s - [%(name)s] - %(levelname)s - %(message)s")
            if not logger.handlers:
                stream_handler = logging.StreamHandler()
                stream_handler.setFormatter(formatter)
                logger.addHandler(stream_handler)
                if log_file is not None:
                    file_path = Path(log_file)
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_handler = logging.FileHandler(file_path, encoding="utf-8")
                    file_handler.setFormatter(formatter)
                    logger.addHandler(file_handler)
            return logger

# --- LOGGING SETUP ---
# Centralized logger format from config utility.
logger = cfg.get_logger("TE_DataLoader")
warnings.filterwarnings("ignore")

# =========================================================
# 1. HELPER FUNCTIONS
# =========================================================

def clean_col_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes column names to snake_case.
    Removes units like '(bar)' or '(s)' and replaces spaces with underscores.
    Example: "Injection Pressure (bar)" -> "injection_pressure_bar"
    """
    df.columns = [
        str(c).strip().lower()
        .replace(' ', '_')
        .replace('-', '_')
        .replace('(', '')
        .replace(')', '')
        .replace('.', '')
        .replace('/', '_per_')
        for c in df.columns
    ]
    return df

def get_machine_id(filename: str) -> str:
    """
    Extracts Machine ID from filename using Regex.
    Handles 'M-471', 'Machine_471', or just '471'.
    """
    match = re.search(r'M-?(\d+)', str(filename), re.IGNORECASE)
    return match.group(1) if match else "Unknown"


def _can_use_pyarrow() -> bool:
    """Checks whether pyarrow is installed for fast CSV parsing."""
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        return False


PYARROW_AVAILABLE = _can_use_pyarrow()


def _read_csv_optimized(
    file_path: Union[str, Path],
    sep: str = ",",
    chunked: bool = False,
    **kwargs: Any
) -> Any:
    """
    Reads CSV with fast-engine preference:
    1) pyarrow (if installed), 2) c, 3) python fallback.
    This preserves chunking while only falling back to python on parse failure.
    """
    if chunked:
        # pandas+pyarrow does not support `chunksize`; skip it for iterative loading.
        engine_order: List[str] = ["c", "python"]
    else:
        engine_order = ["c", "python"]
        if PYARROW_AVAILABLE:
            engine_order.insert(0, "pyarrow")

    last_error: Optional[Exception] = None
    for engine_name in engine_order:
        try:
            return pd.read_csv(file_path, sep=sep, engine=engine_name, **kwargs)
        except Exception as read_err:
            last_error = read_err
            if engine_name != "python":
                logger.info(
                    "   ℹ️ %s engine failed for %s (%s). Trying fallback engine.",
                    engine_name,
                    Path(file_path).name,
                    read_err
                )

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Failed to read CSV file: {file_path}")


SENSOR_ALIAS_GROUPS = {
    'cyl_tmp_z1': ['zone_1_temp', 'zone_1_temperature', 'act_zone_1', 'act_zone_1_temp', 'act_zone_1_temperature'],
    'cyl_tmp_z2': ['zone_2_temp', 'zone_2_temperature', 'act_zone_2', 'act_zone_2_temp', 'act_zone_2_temperature'],
    'cyl_tmp_z3': ['zone_3_temp', 'zone_3_temperature', 'act_zone_3', 'act_zone_3_temp', 'act_zone_3_temperature'],
    'cyl_tmp_z4': ['zone_4_temp', 'zone_4_temperature', 'act_zone_4', 'act_zone_4_temp', 'act_zone_4_temperature'],
    'plasticizing_time': ['act_charge_time', 'charge_time', 'plasticizing_tm'],
    'injection_time': ['act_inj_time', 'inj_time'],
    'injection_pressure': ['injection_peak_pressure', 'max_injection_pressure', 'peak_injection_pressure', 'max_inj_pres'],
    'cushion': ['cushion_position', 'act_cushion'],
}


def normalize_machine_id(series: pd.Series) -> pd.Series:
    """Normalize machine IDs to cleaned integer-like strings (e.g., 'M-231' -> '231')."""
    ids = series.astype(str).str.extract(r'(\d+)')[0]
    ids = ids.str.lstrip('0')
    # Keep single 0 if all digits were zeros.
    ids = ids.where(ids != "", "0")
    return ids


def normalize_timestamp_to_utc_naive(series: pd.Series) -> pd.Series:
    """Parse mixed timestamp inputs and force UTC-naive for consistent matching."""
    ts = pd.to_datetime(series, errors='coerce', utc=True)
    return ts.dt.tz_convert(None)


def downcast_float64_to_float32(df: pd.DataFrame, label: str = "dataframe") -> pd.DataFrame:
    """Downcast float64 columns to float32 to reduce RAM footprint."""
    if df.empty:
        return df

    float_cols = df.select_dtypes(include=['float64']).columns.tolist()
    if not float_cols:
        return df

    df[float_cols] = df[float_cols].astype(np.float32)
    logger.info(
        f"   🧠 Downcasted {len(float_cols)} float64 columns to float32 in {label}"
    )
    return df


def get_sensor_alias_map() -> Dict[str, str]:
    """Build alias->canonical sensor mapping using hardcoded aliases + config mapping."""
    alias_map = {}
    for canonical, aliases in SENSOR_ALIAS_GROUPS.items():
        alias_map[clean_string(canonical)] = clean_string(canonical)
        for alias in aliases:
            alias_map[clean_string(alias)] = clean_string(canonical)

    # Include config-driven aliases (raw header -> canonical)
    for raw_name, canonical_name in getattr(cfg, 'COLUMN_MAPPING', {}).items():
        alias_map[clean_string(raw_name)] = clean_string(canonical_name)

    return alias_map


def recover_sensor_schema(param_wide: pd.DataFrame, sensor_global_means: Dict[str, float]) -> pd.DataFrame:
    """
    Recover missing canonical sensors from alias columns and safe-fill missing values.
    Safe fill uses global sensor mean first, then 0.0 as fallback.
    """
    if param_wide.empty:
        return param_wide

    # Ensure column names are clean for alias matching.
    param_wide = clean_col_names(param_wide)

    # 1) Alias-based recovery at wide-table level.
    for canonical, aliases in SENSOR_ALIAS_GROUPS.items():
        canonical_clean = clean_string(canonical)
        alias_cols = [clean_string(a) for a in aliases if clean_string(a) in param_wide.columns]
        if canonical_clean not in param_wide.columns and alias_cols:
            param_wide[canonical_clean] = param_wide[alias_cols].mean(axis=1, skipna=True)
            logger.warning(
                f"   ⚠️ Recovered missing sensor '{canonical_clean}' from aliases: {alias_cols}"
            )

    # 2) Safe-fill required sensors.
    for required_col in getattr(cfg, 'REQUIRED_PARAM_COLS', []):
        required_clean = clean_string(required_col)
        fallback_mean = sensor_global_means.get(required_clean, np.nan)
        if pd.isna(fallback_mean):
            fallback_mean = 0.0

        if required_clean not in param_wide.columns:
            logger.warning(
                f"   ⚠️ Missing sensor '{required_clean}' after pivot. "
                f"Safe-fill with global mean={fallback_mean:.5f}"
            )
            param_wide[required_clean] = fallback_mean
        else:
            param_wide[required_clean] = param_wide[required_clean].fillna(fallback_mean)

    return param_wide


def enforce_physics_limits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clips key sensor values to physically valid ranges from config.
    Keeps ETL stable against impossible readings without dropping rows.
    """
    if df.empty:
        return df

    clipped_df = df.copy()
    physics_ranges = getattr(cfg, 'PHYSICS_RANGES', {})

    for param_name, bounds in physics_ranges.items():
        try:
            if param_name not in clipped_df.columns:
                continue
            lower, upper = bounds
            numeric_series = pd.to_numeric(clipped_df[param_name], errors='coerce')
            out_of_range = ((numeric_series < lower) | (numeric_series > upper)).sum()
            clipped_df[param_name] = numeric_series.clip(lower=lower, upper=upper)
            if int(out_of_range) > 0:
                logger.info(
                    "   🛡️ Physics clipping applied on %s: %s rows adjusted to [%s, %s]",
                    param_name,
                    int(out_of_range),
                    lower,
                    upper
                )
        except Exception as limit_err:
            logger.warning(
                "⚠️ Failed enforcing physics limits for %s: %s",
                param_name,
                limit_err
            )

    return clipped_df


def summarize_zero_merge_failure(
    hydra_df: pd.DataFrame,
    hydra_ts_col: str,
    param_df: pd.DataFrame,
    param_ts_col: str
) -> str:
    """Generate a detailed message for zero-row merge failures."""
    hydra_ids = set(hydra_df['machine_id'].dropna().astype(str).unique())
    param_ids = set(param_df['machine_id'].dropna().astype(str).unique())

    only_hydra = sorted(hydra_ids - param_ids)
    only_param = sorted(param_ids - hydra_ids)
    common_ids = sorted(hydra_ids & param_ids)

    # Check time overlap on shared IDs.
    no_time_overlap = []
    if common_ids:
        h_ranges = hydra_df.groupby('machine_id')[hydra_ts_col].agg(['min', 'max'])
        p_ranges = param_df.groupby('machine_id')[param_ts_col].agg(['min', 'max'])
        for mid in common_ids:
            if mid not in h_ranges.index or mid not in p_ranges.index:
                continue
            h_min, h_max = h_ranges.loc[mid, 'min'], h_ranges.loc[mid, 'max']
            p_min, p_max = p_ranges.loc[mid, 'min'], p_ranges.loc[mid, 'max']
            if pd.isna(h_min) or pd.isna(h_max) or pd.isna(p_min) or pd.isna(p_max):
                continue
            if h_max < p_min or p_max < h_min:
                no_time_overlap.append(mid)

    msg = (
        "Zero-row merge detected after tolerance fallback (15 min -> 60 min). "
        f"Hydra IDs={len(hydra_ids)}, Param IDs={len(param_ids)}, Common IDs={len(common_ids)}. "
        f"Hydra-only IDs sample={only_hydra[:15]}, Param-only IDs sample={only_param[:15]}, "
        f"No-time-overlap IDs sample={no_time_overlap[:15]}."
    )
    return msg

# =========================================================
# 2. FEATURE ENGINEERING ENGINE
# =========================================================

def add_advanced_features(param_df):
    """
    Generates Physics-based and Statistical features.
    OPTIMIZED: Forces float32 on creation to prevent MemoryErrors.
    """
    if param_df.empty:
        return param_df

    logger.info("✨ Generating Advanced Engineering Features (Memory Safe Mode)...")

    # Sort for time-series calculations
    param_df = param_df.sort_values(['machine_id', 'timestamp'])

    # Identify Sensor Columns
    available_cols = [c for c in param_df.columns if c in cfg.REQUIRED_PARAM_COLS]

    for col in available_cols:
        # Group by Machine
        grouped = param_df.groupby('machine_id')[col]

        # 1. History (Lag) - Force float32
        param_df[f'{col}_lag_1'] = grouped.shift(1).astype(np.float32)

        # 2. Rate of Change (Delta)
        # Note: Operations between float32 columns generally stay float32, but we cast to be safe
        param_df[f'{col}_delta'] = (param_df[col] - param_df[f'{col}_lag_1']).astype(np.float32)

        # 3. Short-Term Trend (Rolling Average)
        param_df[f'{col}_roll_avg_3'] = grouped.transform(
            lambda x: x.rolling(3, min_periods=1).mean()
        ).astype(np.float32)

        # 4. Stability (Rolling Std Dev)
        param_df[f'{col}_roll_std_3'] = grouped.transform(
            lambda x: x.rolling(3, min_periods=1).std()
        ).astype(np.float32)

        # 5. Long-Term Trend (EMA)
        param_df[f'{col}_ema_5'] = grouped.transform(
            lambda x: x.ewm(span=5, adjust=False).mean()
        ).astype(np.float32)

    # 6. Interaction Features (Physics)
    if 'injection_pressure' in param_df.columns and 'cyl_tmp_z1' in param_df.columns:
        param_df['pressure_x_temp'] = (param_df['injection_pressure'] * param_df['cyl_tmp_z1']).astype(np.float32)

    if 'cycle_time' in param_df.columns and 'cushion' in param_df.columns:
        param_df['fill_efficiency'] = (param_df['cushion'] / (param_df['cycle_time'] + 1e-5)).astype(np.float32)

    # MEMORY FIX: In-place fill to avoid copying the dataframe
    param_df.fillna(0, inplace=True)

    # Cleanup intermediate grouping objects
    import gc
    gc.collect()

    logger.info(f"   -> Added {len(param_df.columns) - len(available_cols)} new derived features.")
    return param_df

# =========================================================
# 3. DATA LOADING & EXTRACTION
# =========================================================

def load_hydra_data() -> pd.DataFrame:
    """Loads and standardizes Order/Production data from Hydra and MES files (CSV/XLSX)."""
    configured_pattern = getattr(cfg, 'HYDRA_PATTERN', '*Hydra*.csv')
    search_patterns = {
        configured_pattern,
        "*Hydra*.csv",
        "*Hydra*.xlsx",
        "MES*.csv",
        "MES*.xlsx",
    }
    if configured_pattern.lower().endswith(".csv"):
        search_patterns.add(configured_pattern[:-4] + ".xlsx")

    discovered = []
    for pattern in search_patterns:
        discovered.extend(cfg.DATA_DIR.glob(pattern))

    # De-duplicate paths while preserving a stable order
    unique_files = {}
    for f in discovered:
        unique_files[str(f.resolve())] = f
    files = sorted(unique_files.values(), key=lambda p: p.name.lower())

    if not files:
        logger.error("❌ No Hydra/MES files found!")
        return pd.DataFrame()

    all_hydra = []
    logger.info(f"📄 Found {len(files)} Hydra/MES files.")

    for f in files:
        try:
            suffix = f.suffix.lower()
            if suffix == '.csv':
                df = _read_csv_optimized(
                    f,
                    low_memory=False,
                    on_bad_lines='skip'
                )
            elif suffix in ['.xlsx', '.xls']:
                df = pd.read_excel(f)
            else:
                logger.warning(f"   ⚠️ Skipping unsupported file type: {f.name}")
                continue

            df = clean_col_names(df)

            # Normalize machine ID from known columns; fallback to file name.
            machine_col = next(
                (c for c in ['machine_nr', 'machine_id', 'machine_no', 'machine'] if c in df.columns),
                None
            )
            if machine_col:
                extracted = df[machine_col].astype(str).str.extract(r'(?i)m[-_ ]?(\d+)|(\d+)')
                df['machine_id'] = extracted.bfill(axis=1).iloc[:, 0].fillna(df[machine_col].astype(str))
            else:
                df['machine_id'] = get_machine_id(f.name)

            # CRITICAL: Fuzzy ID normalization for merge reliability.
            df['machine_id'] = normalize_machine_id(df['machine_id'])
            df = df.dropna(subset=['machine_id'])

            all_hydra.append(df)
        except Exception as e:
            logger.warning(f"   ⚠️ Failed to load {f.name}: {e}")

    if not all_hydra:
        return pd.DataFrame()

    full_hydra = pd.concat(all_hydra, ignore_index=True)
    del all_hydra
    gc.collect()
    full_hydra = downcast_float64_to_float32(full_hydra, label="hydra")
    logger.info(f"   ✅ Hydra/MES Data Loaded: {full_hydra.shape[0]} rows")
    return full_hydra

def clean_string(s: Any) -> str:
    """Helper to normalize a single string for comparison."""
    return str(s).strip().lower().replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '').replace('.', '').replace('/', '_per_')


def _load_valid_sensors_from_manual(manual_path: Path) -> Set[str]:
    """Reads the parameter manual and returns a cleaned set of valid sensor names."""
    if not manual_path.exists():
        logger.warning(f"⚠️ Sensor manual not found: {manual_path.name}. Continuing without sensor filter.")
        return set()

    try:
        manual_df = pd.read_excel(manual_path)
        if manual_df.empty:
            logger.warning("⚠️ Sensor manual is empty. Continuing without sensor filter.")
            return set()

        # Prefer a column explicitly describing variable/sensor names.
        candidate_cols = [
            c for c in manual_df.columns
            if any(k in str(c).lower() for k in ['variable', 'sensor', 'parameter', 'name'])
        ]
        target_col = candidate_cols[0] if candidate_cols else manual_df.columns[0]

        sensors = {
            clean_string(v)
            for v in manual_df[target_col].dropna().astype(str)
            if str(v).strip()
        }
        logger.info(f"   ✅ Loaded {len(sensors)} valid sensors from {manual_path.name}.")
        return sensors
    except Exception as e:
        logger.warning(f"⚠️ Could not read sensor manual ({manual_path.name}): {e}. Continuing without sensor filter.")
        return set()


def _detect_delimiter(file_path: Path) -> str:
    """Detects a CSV delimiter using a small raw-text sample."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore', newline='') as fh:
            sample = fh.read(4096)
        if not sample.strip():
            return ','
        dialect = csv.Sniffer().sniff(sample, delimiters=[',', ';', '\t', '|'])
        return dialect.delimiter
    except Exception:
        return ','


def _detect_param_columns(sample_df: pd.DataFrame, valid_sensors_clean: Set[str]) -> Dict[str, int]:
    """
    Detects indices of machine_id, timestamp, variable_name, and value
    from first 5 rows of a headerless parameter file.
    """
    if sample_df.empty:
        return {'machine_id': -1, 'timestamp': -1, 'variable_name': -1, 'value': -1}

    scores = {}
    for col in sample_df.columns:
        ser = sample_df[col].astype(str).str.strip()
        ser = ser[ser != ""]
        if ser.empty:
            continue

        clean_ser = ser.map(clean_string)
        ts_ratio = pd.to_datetime(ser, errors='coerce').notna().mean()
        machine_ratio = ser.str.contains(r'(?i)\bM[-_ ]?\d+\b', regex=True, na=False).mean()
        value_ratio = pd.to_numeric(ser.str.replace(',', '.', regex=False), errors='coerce').notna().mean()

        if valid_sensors_clean:
            var_ratio = clean_ser.isin(valid_sensors_clean).mean()
        else:
            var_ratio = 0.0

        # Keyword boost helps when manual contains aliases and exact match is unavailable.
        keyword_ratio = clean_ser.str.contains(
            r'(temp|pressure|time|speed|position|force|cushion|zone|inject|charge|mold|clamp)',
            regex=True,
            na=False
        ).mean()
        var_score = max(var_ratio, keyword_ratio * 0.6)

        scores[col] = {
            'machine_id': machine_ratio,
            'timestamp': ts_ratio,
            'variable_name': var_score,
            'value': value_ratio
        }

    selected = {'machine_id': -1, 'timestamp': -1, 'variable_name': -1, 'value': -1}
    used_cols = set()
    thresholds = {'variable_name': 0.2, 'timestamp': 0.4, 'machine_id': 0.2, 'value': 0.4}

    for role in ['variable_name', 'timestamp', 'machine_id', 'value']:
        best_col, best_score = None, -1.0
        for col, role_scores in scores.items():
            if col in used_cols:
                continue
            if role_scores[role] > best_score:
                best_col, best_score = col, role_scores[role]
        if best_col is not None and best_score >= thresholds[role]:
            selected[role] = best_col
            used_cols.add(best_col)

    return selected

def detect_csv_layout(file_path: Path, valid_sensors_clean: Set[str]) -> Tuple[int, str, List[str]]:
    """
    Intelligently scans the first 50 lines of a CSV to find:
    1. The Header Row (skips metadata)
    2. The Delimiter (comma, semicolon, tab)
    Returns: (header_row_index, delimiter, found_columns)
    """
    best_score = 0
    best_config = (0, ',', []) # Default
    
    delimiters = [',', ';', '\t', '|']
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [f.readline() for _ in range(50)] # Read first 50 lines
            
        for i, line in enumerate(lines):
            if not line.strip(): continue # Skip empty lines
            
            for sep in delimiters:
                cols = [c.strip().replace('"', '') for c in line.split(sep)]
                
                # Normalize columns to match manual
                # We check how many "Manual Sensors" appear in this line
                matches = 0
                clean_cols = []
                for c in cols:
                    c_clean = str(c).strip().lower().replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
                    clean_cols.append(c_clean)
                    if c_clean in valid_sensors_clean:
                        matches += 1
                
                # If this line has more matches than previous best, it's the header!
                if matches > best_score:
                    best_score = matches
                    best_config = (i, sep, cols)
                    
        return best_config
        
    except Exception as e:
        logger.warning(f"   ⚠️ Layout Detection Failed: {e}")
        return (0, ',', [])

def load_param_data() -> pd.DataFrame:
    """
    Content-based loader for parameter CSVs.
    - Scans all CSV files under data directory.
    - Reads first 5 rows and detects schema using _detect_param_columns.
    - Loads a file only if timestamp and value columns are detected.
    - Filters rows to valid/required sensors after loading.
    """
    # 1) Read valid sensors from manual
    manual_path_xlsx = cfg.DATA_DIR / "AI_cup_parameter_info.xlsx"
    valid_sensors = _load_valid_sensors_from_manual(manual_path_xlsx)

    # 2) Build canonical target names and alias map.
    required_params = {clean_string(c) for c in getattr(cfg, 'REQUIRED_PARAM_COLS', [])}
    alias_map = get_sensor_alias_map()

    # 3) Discover parameter files (all CSVs by configuration + hard fallback).
    param_patterns = {
        getattr(cfg, 'PARAM_PATTERN', '*.csv'),
        "*.csv",
    }
    discovered = []
    for pattern in param_patterns:
        discovered.extend(cfg.DATA_DIR.glob(pattern))

    unique_files = {}
    for f in discovered:
        unique_files[str(f.resolve())] = f
    files = sorted(unique_files.values(), key=lambda p: p.name.lower())

    if not files:
        logger.error("❌ No parameter files found!")
        return pd.DataFrame()

    all_params = []
    logger.info(
        f"📄 Found {len(files)} CSV files. Auto-detecting parameter schema from first 5 rows..."
    )

    for f in files:
        try:
            delimiter = _detect_delimiter(f)
            sample = _read_csv_optimized(
                f,
                header=None,
                nrows=5,
                sep=delimiter,
                on_bad_lines='skip',
                dtype=str
            )

            detected = _detect_param_columns(sample, valid_sensors)
            idx_mach = detected['machine_id']
            idx_time = detected['timestamp']
            idx_var = detected['variable_name']
            idx_val = detected['value']

            # Content gate: only files with detectable timestamp + value columns are attempted.
            if idx_time == -1 or idx_val == -1:
                logger.warning(
                    f"   ⚠️ Skipping {f.name}: timestamp/value columns not detected. "
                    f"Detected={detected}. Skipping this file."
                )
                continue

            use_cols = sorted(set(i for i in [idx_mach, idx_time, idx_var, idx_val] if i != -1))
            rename_map = {}
            if idx_mach != -1:
                rename_map[idx_mach] = 'machine_id'
            rename_map[idx_time] = 'timestamp'
            if idx_var != -1:
                rename_map[idx_var] = 'variable_name'
            rename_map[idx_val] = 'value'

            logger.info(
                f"   ⏳ Loading {f.name} in chunks (100000 rows) "
                f"(Detected columns: {detected}, sep='{delimiter}')"
            )

            file_chunks = []
            raw_rows = 0
            kept_rows = 0
            for chunk in _read_csv_optimized(
                f,
                header=None,
                usecols=use_cols,
                sep=delimiter,
                on_bad_lines='skip',
                dtype=str,
                chunksize=100000,
                chunked=True
            ):
                chunk = chunk.rename(columns=rename_map)
                raw_rows += len(chunk)

                # Fill machine_id from filename if it was not detected in content.
                if 'machine_id' not in chunk.columns:
                    chunk['machine_id'] = get_machine_id(f.name)
                else:
                    chunk['machine_id'] = chunk['machine_id'].fillna("").astype(str).str.strip()
                    missing_machine = chunk['machine_id'].isin(["", "nan", "None"])
                    if missing_machine.any():
                        chunk.loc[missing_machine, 'machine_id'] = get_machine_id(f.name)
                chunk['machine_id'] = normalize_machine_id(chunk['machine_id'])
                chunk = chunk.dropna(subset=['machine_id'])
                if chunk.empty:
                    continue

                # If variable name was not detected, downstream filtering will safely drop rows.
                if 'variable_name' not in chunk.columns:
                    chunk['variable_name'] = ""

                # Normalize sensor names, then filter early for low memory use.
                chunk['variable_name'] = chunk['variable_name'].astype(str).map(clean_string)
                if valid_sensors:
                    allowed_sensors = set(valid_sensors) | set(alias_map.keys()) | set(required_params)
                    chunk = chunk[chunk['variable_name'].isin(allowed_sensors)]
                if chunk.empty:
                    continue

                chunk['variable_name'] = chunk['variable_name'].replace(alias_map)
                if required_params:
                    chunk = chunk[chunk['variable_name'].isin(required_params)]
                if chunk.empty:
                    continue

                # Parse value + timestamp safely.
                chunk['value'] = pd.to_numeric(
                    chunk['value'].astype(str).str.replace(',', '.', regex=False),
                    errors='coerce'
                )
                chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], errors='coerce')

                # Keep only rows that can be used downstream.
                chunk = chunk.dropna(subset=['timestamp', 'variable_name'])
                if chunk.empty:
                    continue

                kept_rows += len(chunk)
                file_chunks.append(chunk[['machine_id', 'timestamp', 'variable_name', 'value']].copy())

            if not file_chunks:
                logger.warning(f"   ⚠️ No valid rows after parsing/filtering in {f.name}")
                continue

            logger.info(f"      -> Kept {kept_rows}/{raw_rows} rows after in-chunk filtering")
            file_df = pd.concat(file_chunks, ignore_index=True)
            del file_chunks
            gc.collect()
            all_params.append(file_df)

        except Exception as e:
            logger.warning(f"   ⚠️ Failed to load {f.name}: {e}")

    if not all_params:
        return pd.DataFrame()

    full_params = pd.concat(all_params, ignore_index=True)
    del all_params
    gc.collect()
    full_params['machine_id'] = normalize_machine_id(full_params['machine_id'])
    full_params = full_params.dropna(subset=['machine_id'])
    full_params = downcast_float64_to_float32(full_params, label="param")
    logger.info(f"   ✅ Smart Parameter Data Loaded: {full_params.shape[0]} rows")
    return full_params

# =========================================================
# 4. MAIN MERGE LOGIC (THE BRAIN)
# =========================================================

def load_and_merge_data(use_cache: bool = True) -> pd.DataFrame:
    """
    Main pipeline function.
    1. Checks cache for speed.
    2. Loads Raw Data.
    3. Pivots & Engineers Features.
    4. Merges with 2-minute tolerance.
    5. Saves cache for next time.
    """
    start_time = time.time()
    cache_path = cfg.DATA_DIR / "processed_full_dataset.pkl"

    # --- Step 0: Cache Check ---
    if use_cache and cache_path.exists():
        # Check if cache is older than 24 hours
        file_age = time.time() - cache_path.stat().st_mtime
        source_mtime = Path(__file__).stat().st_mtime if '__file__' in globals() else 0
        cache_is_newer_than_code = cache_path.stat().st_mtime >= source_mtime
        if file_age < 86400 and cache_is_newer_than_code: # 24 hours
            logger.info("⚡ Loading Data from Cache (Fast Mode)...")
            return pd.read_pickle(cache_path)
        else:
            logger.info("⏳ Cache expired/stale vs code. Reloading fresh data...")

    # --- Step 1: Load Raw ---
    logger.info("🚀 Starting Full ETL Pipeline...")
    hydra_df = load_hydra_data()
    param = load_param_data()

    # Standardize Hydra columns before any merge logic.
    if not hydra_df.empty:
        hydra_df.columns = [str(c).lower().strip() for c in hydra_df.columns]
        hydra_df = clean_col_names(hydra_df)
        rename_map = {
            'plant_shift_calendar_date': 'plant_shift_date',
            'date': 'plant_shift_date',
            'status_name': 'machine_status_name',
            'machine_status': 'machine_status_name',
            'segment_name': 'segment_abbr_name',
            'business_segment': 'segment_abbr_name',
            'segment': 'segment_abbr_name',
            'machine_event_create_time': 'plant_shift_date'
        }
        hydra_df = hydra_df.rename(columns=rename_map)
        hydra_df = hydra_df.loc[:, ~hydra_df.columns.duplicated(keep='first')]

    hydra = downcast_float64_to_float32(hydra_df, label="hydra_raw")
    param = downcast_float64_to_float32(param, label="param_raw")

    if param.empty:
        logger.error("❌ Critical Error: Parameter stream is empty.")
        return pd.DataFrame()
    hydra_available = not hydra.empty
    if not hydra_available:
        logger.warning(
            "⚠️ Hydra stream is empty. Proceeding in sensor-only monitor mode."
        )

    # --- Step 2: Timestamp + ID Normalization ---
    logger.info("⏰ Standardizing IDs and Timestamps...")

    # Identify date column in Hydra (if available)
    h_date_col = next((c for c in hydra.columns if 'time' in c or 'date' in c), None) if hydra_available else None
    p_date_col = 'timestamp'

    if hydra_available and not h_date_col:
        logger.error("❌ Could not find timestamp/date column in Hydra data.")
        return pd.DataFrame()

    # CRITICAL: Normalize IDs in both streams to cleaned numeric string.
    param['machine_id'] = normalize_machine_id(param['machine_id'])
    param = param.dropna(subset=['machine_id'])
    if hydra_available:
        hydra['machine_id'] = normalize_machine_id(hydra['machine_id'])
        hydra = hydra.dropna(subset=['machine_id'])

    # Force UTC-naive timestamps to handle mixed timezone sources.
    param[p_date_col] = normalize_timestamp_to_utc_naive(param[p_date_col])
    param = param.dropna(subset=[p_date_col])
    if hydra_available:
        hydra[h_date_col] = normalize_timestamp_to_utc_naive(hydra[h_date_col])
        hydra = hydra.dropna(subset=[h_date_col])

    # Time pruning intentionally disabled to preserve full monitor timeline.
    logger.info(
        f"🛰️ Time-pruning disabled. Keeping all sensor rows in memory: {len(param):,}"
    )

    # --- Step 3: Pivoting, Schema Recovery & Feature Engineering ---
    logger.info("🔄 Pivoting Sensor Data (Long -> Wide) with schema recovery...")

    alias_map = get_sensor_alias_map()
    relevant_params = {clean_string(p) for p in cfg.REQUIRED_PARAM_COLS}

    # Normalize + map aliases before filtering.
    param['variable_name'] = param['variable_name'].astype(str).map(clean_string).replace(alias_map)
    param = param[param['variable_name'].isin(relevant_params)]
    param['value'] = pd.to_numeric(param['value'], errors='coerce')

    sensor_global_means = (
        param.dropna(subset=['value'])
        .groupby('variable_name')['value']
        .mean()
        .to_dict()
    )

    param_wide = param.pivot_table(
        index=[p_date_col, 'machine_id'],
        columns='variable_name',
        values='value',
        aggfunc='mean'
    ).reset_index()
    param_wide = clean_col_names(param_wide)
    param_wide = recover_sensor_schema(param_wide, sensor_global_means)
    param_wide = enforce_physics_limits(param_wide)
    param_wide = downcast_float64_to_float32(param_wide, label="param_wide_pre_features")

    # Preserve existing advanced feature engineering.
    param_wide = add_advanced_features(param_wide)
    param_wide = downcast_float64_to_float32(param_wide, label="param_wide_post_features")

    # Release long-format sensor table before merge.
    del param
    gc.collect()

    # --- Step 4: Monitor-First Merge (ASOF) ---
    logger.info("🔗 Linking Sensor and Hydra streams (sensor-left asof merge)...")

    # Keep explicit right timestamp column for diagnostics.
    param_wide.rename(columns={p_date_col: 'param_timestamp'}, inplace=True)
    p_merge_col = 'param_timestamp'

    # Normalize IDs/timestamps again after transforms.
    param_wide['machine_id'] = normalize_machine_id(param_wide['machine_id'])
    param_wide[p_merge_col] = normalize_timestamp_to_utc_naive(param_wide[p_merge_col])

    param_wide = param_wide.dropna(subset=['machine_id', p_merge_col]).copy()
    param_wide['machine_id'] = param_wide['machine_id'].astype(str)
    sensor_rows_before_merge = len(param_wide)

    logger.info(
        f"   🕒 Param timestamp range: {param_wide[p_merge_col].min()} -> {param_wide[p_merge_col].max()}"
    )

    if hydra_available:
        hydra['machine_id'] = normalize_machine_id(hydra['machine_id'])
        hydra[h_date_col] = normalize_timestamp_to_utc_naive(hydra[h_date_col])
        hydra = hydra.dropna(subset=['machine_id', h_date_col]).copy()
        hydra['machine_id'] = hydra['machine_id'].astype(str)

        # Timestamp overlap diagnostics.
        logger.info(
            f"   🕒 Hydra timestamp range: {hydra[h_date_col].min()} -> {hydra[h_date_col].max()}"
        )

        # Pre-merge diagnostics (requested): show machine_id + timestamp heads.
        hydra_preview = hydra[['machine_id', h_date_col]].rename(columns={h_date_col: 'timestamp'}).head(5)
        param_preview = param_wide[['machine_id', p_merge_col]].rename(columns={p_merge_col: 'timestamp'}).head(5)
        logger.info("   🔍 Hydra preview before merge:\n%s", hydra_preview.to_string(index=False))
        logger.info("   🔍 Param preview before merge:\n%s", param_preview.to_string(index=False))

        # Sort by by-key + time-key for merge_asof stability.
        hydra = hydra.sort_values([h_date_col, 'machine_id'])
        param_wide = param_wide.sort_values([p_merge_col, 'machine_id'])
        gc.collect()

        def _merge_with_tolerance(tol_minutes: int) -> Tuple[pd.DataFrame, int]:
            # Monitor-first merge: left side is sensor timeline, so all sensor rows are retained.
            merged_local = pd.merge_asof(
                param_wide,
                hydra,
                left_on=p_merge_col,
                right_on=h_date_col,
                by='machine_id',
                direction='nearest',
                tolerance=pd.Timedelta(minutes=tol_minutes)
            )
            matched_local = merged_local[h_date_col].notna().sum()
            logger.info(
                f"   📌 merge_asof tolerance={tol_minutes} min -> hydra-matched rows: {matched_local}/{len(merged_local)}"
            )
            return merged_local, matched_local

        merged, matched = _merge_with_tolerance(15)
        if matched == 0:
            logger.warning("⚠️ No matches at 15 min tolerance. Retrying with 60 min.")
            merged, matched = _merge_with_tolerance(60)

        if matched == 0:
            logger.warning(
                "⚠️ No Hydra rows matched even at 60 min tolerance. Keeping sensor timeline with empty Hydra fields."
            )
    else:
        merged = param_wide.copy()
        matched = 0
        logger.warning(
            "⚠️ Hydra unavailable. Skipping merge and keeping full sensor timeline."
        )

    # Ensure monitor timeline timestamp is preserved downstream.
    merged['timestamp'] = merged[p_merge_col]
    if len(merged) != sensor_rows_before_merge:
        logger.warning(
            f"⚠️ Sensor-row count changed during merge: before={sensor_rows_before_merge:,}, after={len(merged):,}"
        )

    # Filter physically impossible cycle speeds after merge alignment.
    try:
        min_cycle_time = float(getattr(cfg, 'MIN_VALID_CYCLE_TIME_SEC', 1.0))
        if 'cycle_time' in merged.columns:
            cycle_series = pd.to_numeric(merged['cycle_time'], errors='coerce')
            invalid_cycle_mask = cycle_series.notna() & (cycle_series < min_cycle_time)
            dropped_cycles = int(invalid_cycle_mask.sum())
            if dropped_cycles > 0:
                merged = merged.loc[~invalid_cycle_mask].copy()
            logger.info(
                "🧪 Cycle-time sanity filter applied (< %.2fs): dropped %s rows",
                min_cycle_time,
                dropped_cycles
            )
        else:
            logger.warning("⚠️ cycle_time column missing. Skipping half-baked cycle filter.")
    except Exception as cycle_filter_err:
        logger.warning(f"⚠️ Failed cycle-time sanity filtering: {cycle_filter_err}")

    # --- Step 5: Scrap Labeling (Business Logic) ---
    logger.info("🏷️  Applying Scrap Labeling...")
    if 'actual_scrap_qty' not in merged.columns:
        logger.warning(
            "⚠️ 'actual_scrap_qty' not found after merge. Creating zero-filled fallback labels."
        )
        merged['actual_scrap_qty'] = 0

    if 'actual_qty' not in merged.columns:
        logger.warning("⚠️ 'actual_qty' missing. Creating zero-filled column for scrap_rate.")
        merged['actual_qty'] = 0

    merged['actual_qty'] = pd.to_numeric(merged['actual_qty'], errors='coerce').fillna(0)
    merged['actual_scrap_qty'] = pd.to_numeric(merged['actual_scrap_qty'], errors='coerce').fillna(0)
    merged['scrap_rate'] = np.where(
        merged['actual_qty'] > 0,
        merged['actual_scrap_qty'] / merged['actual_qty'],
        0.0
    )
    # Label is based directly on actual_scrap_qty as requested.
    merged['is_scrap'] = (merged['actual_scrap_qty'] > 0).astype(int)
    # Preserve monitor rows without orders as "Good/Unknown" for dashboard safety.
    merged['is_scrap'] = merged['is_scrap'].fillna(0).astype(int)

    # =========================================================
    # 🛡️ SAFETY FIX FOR MISSING COLUMNS (SAFE MEAN FILL)
    # =========================================================
    logger.info("🛡️ Performing Schema Validation & Safe Fill...")

    for required_col_raw in cfg.REQUIRED_PARAM_COLS:
        required_col = clean_string(required_col_raw)
        fallback_mean = sensor_global_means.get(required_col, np.nan)
        if pd.isna(fallback_mean):
            fallback_mean = 0.0

        if required_col not in merged.columns:
            logger.warning(
                f"   ⚠️ Missing Config Column: '{required_col}'. "
                f"Safe-fill with global mean={fallback_mean:.5f}"
            )
            merged[required_col] = fallback_mean
        else:
            merged[required_col] = pd.to_numeric(merged[required_col], errors='coerce').fillna(fallback_mean)

    # =========================================================
    # --- Step 6: Iterative Chunked Resampling (Memory-Safe) ---
    # =========================================================
    logger.info(
        f"⏱️ Iterative Chunking: Resampling {len(merged):,} rows to 1-minute bins by machine_id."
    )

    merged['timestamp'] = pd.to_datetime(merged['timestamp'], errors='coerce')
    merged = merged.dropna(subset=['machine_id', 'timestamp']).copy()

    chunks = []
    unique_machines = merged['machine_id'].dropna().astype(str).unique()
    context_cols_to_keep = [
        'plant_shift_date',
        'machine_status_name',
        'segment_abbr_name',
        'manufacturing_plant_name'
    ]

    for mid in unique_machines:
        # Process one machine at a time to keep peak memory usage bounded.
        df_machine = merged[merged['machine_id'] == mid].copy()
        if df_machine.empty:
            del df_machine
            gc.collect()
            continue

        try:
            df_machine = df_machine.sort_values('timestamp')
            resampled = (
                df_machine
                .set_index('timestamp')
                .resample('1min')
                .mean(numeric_only=True)
                .reset_index()
            )
            resampled['machine_id'] = mid

            context_cols = [c for c in context_cols_to_keep if c in df_machine.columns]
            if context_cols:
                context_resampled = (
                    df_machine
                    .set_index('timestamp')[context_cols]
                    .resample('1min')
                    .last()
                    .ffill()
                    .bfill()
                    .reset_index(drop=True)
                )
                for col in context_cols:
                    if col in context_resampled.columns:
                        resampled[col] = context_resampled[col].values

            if 'is_scrap' in df_machine.columns:
                scrap_1min = (
                    df_machine
                    .set_index('timestamp')['is_scrap']
                    .resample('1min')
                    .max()
                )
                resampled['is_scrap'] = scrap_1min.values

            chunks.append(resampled)
        except Exception as e:
            logger.warning(f"⚠️ Skipping machine {mid} during chunked resample: {e}")

        del df_machine
        gc.collect()

    if chunks:
        merged = pd.concat(chunks, ignore_index=True)
    else:
        merged = merged.iloc[0:0].copy()
        logger.warning("⚠️ Chunked resampling produced no rows.")

    del chunks
    gc.collect()

    # ---------------------------------------------------------
    # STEP 9: FINAL CLEANUP & SAVE
    # ---------------------------------------------------------
    # --- Step 7: Data Quality Report ---
    report_quality(merged)

    # --- Step 8: Caching ---
    logger.info("💾 Caching Processed Dataset...")
    # --- Step 8a: Final Memory Optimization (CRITICAL) ---
    # Pandas resample() often reverts to float64. We must force float32 before saving.
    float64_cols = merged.select_dtypes(include=['float64']).columns
    if len(float64_cols) > 0:
        logger.info(
            f"   📉 Final Optimization: Converting {len(float64_cols)} float64 columns to float32 to save RAM."
        )
        merged[float64_cols] = merged[float64_cols].astype(np.float32)
    merged.to_pickle(cache_path)
    
    elapsed = time.time() - start_time
    logger.info(f"✅ Data Pipeline Complete in {elapsed:.2f}s. Shape: {merged.shape}")
    logger.info(f"Final columns available: {list(merged.columns)}")
    
    return merged

def report_quality(df: pd.DataFrame) -> None:
    """Prints a mini-report on data health."""
    logger.info("-" * 40)
    logger.info("📊 DATA QUALITY REPORT")
    logger.info("-" * 40)
    logger.info(f"Total Samples: {len(df)}")
    logger.info(f"Scrap Events:  {df['is_scrap'].sum()} ({(df['is_scrap'].mean()*100):.2f}%)")
    logger.info(f"Features:      {len(df.columns)}")
    
    # Check for NaNs in critical columns (after we fixed them above)
    # This block should now be safe from KeyError
    try:
        nans = df[cfg.REQUIRED_PARAM_COLS].isna().sum().sum()
        if nans > 0:
            logger.warning(f"⚠️  Found {nans} missing sensor values (Imputed with 0/Forward Fill)")
        else:
            logger.info("✅ No missing sensor data.")
    except KeyError as e:
        logger.error(f"❌ Safety check failed: {e}. But pipeline will continue.")
        
    logger.info("-" * 40)

if __name__ == "__main__":
    # Test Run
    df = load_and_merge_data(use_cache=False)
    print(df.head())
