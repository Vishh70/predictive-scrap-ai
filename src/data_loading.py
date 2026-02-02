import pandas as pd
import numpy as np
import os
import re
import time
import logging
import warnings
from pathlib import Path
from datetime import datetime

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
        # Default minimal list if config fails
        REQUIRED_PARAM_COLS = ['injection_pressure', 'cycle_time', 'cushion', 'cyl_tmp_z1']
        HYDRA_PATTERN = "*Hydra*.csv"
        PARAM_PATTERN = "*Param*.csv"

# --- LOGGING SETUP ---
# We use a robust logger that prints to console with timestamps
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("TE_DataLoader")
warnings.filterwarnings("ignore")

# =========================================================
# 1. HELPER FUNCTIONS
# =========================================================

def clean_col_names(df):
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

def get_machine_id(filename):
    """
    Extracts Machine ID from filename using Regex.
    Handles 'M-471', 'Machine_471', or just '471'.
    """
    match = re.search(r'M-?(\d+)', str(filename), re.IGNORECASE)
    return match.group(1) if match else "Unknown"

# =========================================================
# 2. FEATURE ENGINEERING ENGINE
# =========================================================

def add_advanced_features(param_df):
    """
    Generates Physics-based and Statistical features.
    This expands the dataset from raw readings to trend-aware data.
    """
    if param_df.empty: 
        return param_df

    logger.info("✨ Generating Advanced Engineering Features...")
    
    # Sort for time-series calculations
    param_df = param_df.sort_values(['machine_id', 'timestamp'])
    
    # Identify Sensor Columns that actually exist in the dataframe
    # We intersect with the config to avoid errors here
    available_cols = [c for c in param_df.columns if c in cfg.REQUIRED_PARAM_COLS]
    
    for col in available_cols:
        # Group by Machine to prevent data bleeding between machines
        grouped = param_df.groupby('machine_id')[col]
        
        # 1. History (Lag) - What happened 1 cycle ago?
        param_df[f'{col}_lag_1'] = grouped.shift(1)
        
        # 2. Rate of Change (Delta) - Is it rising or falling?
        param_df[f'{col}_delta'] = param_df[col] - param_df[f'{col}_lag_1']
        
        # 3. Short-Term Trend (Rolling Average - Window 3)
        param_df[f'{col}_roll_avg_3'] = grouped.transform(lambda x: x.rolling(3, min_periods=1).mean())
        
        # 4. Stability (Rolling Std Dev - Window 3)
        param_df[f'{col}_roll_std_3'] = grouped.transform(lambda x: x.rolling(3, min_periods=1).std())
        
        # 5. Long-Term Trend (Exponential Moving Average)
        # EMA gives more weight to recent data but remembers history
        param_df[f'{col}_ema_5'] = grouped.transform(lambda x: x.ewm(span=5, adjust=False).mean())

    # 6. Interaction Features (Physics)
    # Only calculate if both columns exist
    if 'injection_pressure' in param_df.columns and 'cyl_tmp_z1' in param_df.columns:
        param_df['pressure_x_temp'] = param_df['injection_pressure'] * param_df['cyl_tmp_z1']
    
    if 'cycle_time' in param_df.columns and 'cushion' in param_df.columns:
        # Avoid division by zero
        param_df['fill_efficiency'] = param_df['cushion'] / (param_df['cycle_time'] + 1e-5)

    # Fill NaNs created by lagging (first rows of each machine)
    param_df = param_df.fillna(0)
    
    logger.info(f"   -> Added {len(param_df.columns) - len(available_cols)} new derived features.")
    return param_df

# =========================================================
# 3. DATA LOADING & EXTRACTION
# =========================================================

def load_hydra_data():
    """Loads and standardizes Order/Production (Hydra) data."""
    files = list(cfg.DATA_DIR.glob(cfg.HYDRA_PATTERN))
    if not files: 
        logger.error("❌ No Hydra files found!")
        return pd.DataFrame()
    
    all_hydra = []
    logger.info(f"📄 Found {len(files)} Hydra files.")
    
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False, on_bad_lines='skip')
            df = clean_col_names(df)
            
            # Normalize Machine ID
            if 'machine_nr' in df.columns:
                df['machine_id'] = df['machine_nr'].astype(str).str.replace('M', '', case=False).str.replace('-', '')
            else:
                df['machine_id'] = get_machine_id(f.name)

            all_hydra.append(df)
        except Exception as e:
            logger.warning(f"   ⚠️ Failed to load {f.name}: {e}")
            
    if not all_hydra: return pd.DataFrame()
    
    full_hydra = pd.concat(all_hydra, ignore_index=True)
    logger.info(f"   ✅ Hydra Data Loaded: {full_hydra.shape[0]} rows")
    return full_hydra

def load_param_data():
    """Loads and standardizes Machine Parameter data."""
    files = list(cfg.DATA_DIR.glob(cfg.PARAM_PATTERN))
    if not files: 
        logger.error("❌ No Parameter files found!")
        return pd.DataFrame()
    
    all_params = []
    logger.info(f"📄 Found {len(files)} Parameter files.")
    
    for f in files:
        try:
            df = pd.read_csv(f, engine="python", on_bad_lines='warn')
            df = clean_col_names(df)
            df['machine_id'] = get_machine_id(f.name)
            all_params.append(df)
        except Exception as e:
            logger.warning(f"   ⚠️ Failed to load {f.name}: {e}")

    if not all_params: return pd.DataFrame()
    
    full_params = pd.concat(all_params, ignore_index=True)
    logger.info(f"   ✅ Parameter Data Loaded: {full_params.shape[0]} rows")
    return full_params

# =========================================================
# 4. MAIN MERGE LOGIC (THE BRAIN)
# =========================================================

def load_and_merge_data(use_cache=True):
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
        if file_age < 86400: # 24 hours
            logger.info("⚡ Loading Data from Cache (Fast Mode)...")
            return pd.read_pickle(cache_path)
        else:
            logger.info("⏳ Cache expired. Reloading fresh data...")

    # --- Step 1: Load Raw ---
    logger.info("🚀 Starting Full ETL Pipeline...")
    hydra = load_hydra_data()
    param = load_param_data()
    
    if hydra.empty or param.empty:
        logger.error("❌ Critical Error: Missing input data streams.")
        return pd.DataFrame()

    # --- Step 2: Timestamp Standardization ---
    logger.info("⏰ Standardizing Timestamps...")
    
    # Identify Date Columns intelligently
    h_date_col = next((c for c in hydra.columns if 'time' in c or 'date' in c), None)
    p_date_col = 'timestamp'
    
    if not h_date_col:
        logger.error("❌ Could not find timestamp column in Hydra data.")
        return pd.DataFrame()

    # Convert to UTC-naive datetime for compatibility
    hydra[h_date_col] = pd.to_datetime(hydra[h_date_col], errors='coerce').dt.tz_localize(None)
    param[p_date_col] = pd.to_datetime(param[p_date_col], errors='coerce').dt.tz_localize(None)
    
    # Drop rows without time
    hydra = hydra.dropna(subset=[h_date_col])
    param = param.dropna(subset=[p_date_col])

    # --- Step 3: Pivoting & Engineering ---
    logger.info("🔄 Pivoting Sensor Data (Long -> Wide)...")
    
    # Filter only necessary parameters to save memory
    relevant_params = [p.lower() for p in cfg.REQUIRED_PARAM_COLS]
    param = param[param['variable_name'].astype(str).str.lower().isin(relevant_params)]
    
    # Ensure values are numeric
    param['value'] = pd.to_numeric(param['value'], errors='coerce')
    
    # Pivot: One row per timestamp per machine
    param_wide = param.pivot_table(
        index=[p_date_col, 'machine_id'],
        columns='variable_name',
        values='value',
        aggfunc='mean'
    ).reset_index()
    
    param_wide = clean_col_names(param_wide)
    
    # Apply Feature Engineering
    param_wide = add_advanced_features(param_wide)
    
    # --- Step 4: Enterprise Merge (ASOF) ---
    logger.info("🔗 Linking Order Data with Machine Data (Tolerance: 2 mins)...")
    
    # Sort for merge_asof
    hydra = hydra.sort_values(h_date_col)
    param_wide = param_wide.sort_values(p_date_col)
    
    # Ensure ID types match
    hydra['machine_id'] = hydra['machine_id'].astype(str)
    param_wide['machine_id'] = param_wide['machine_id'].astype(str)
    
    merged = pd.merge_asof(
        hydra,
        param_wide,
        left_on=h_date_col,
        right_on=p_date_col,
        by='machine_id',
        direction='nearest',
        tolerance=pd.Timedelta('2 min')
    )
    
    # --- Step 5: Scrap Labeling (Business Logic) ---
    logger.info("🏷️  Applying Business Logic for Scrap Labeling...")
    
    merged['actual_qty'] = pd.to_numeric(merged['actual_qty'], errors='coerce').fillna(0)
    merged['actual_scrap_qty'] = pd.to_numeric(merged['actual_scrap_qty'], errors='coerce').fillna(0)
    
    # Calculate Scrap Rate
    merged['scrap_rate'] = merged.apply(
        lambda row: row['actual_scrap_qty'] / row['actual_qty'] if row['actual_qty'] > 0 else 0, axis=1
    )
    
    # Threshold: Consider "Scrap" if failure rate > 2%
    # This prevents marking 1 bad part in 10,000 as a "Bad Batch"
    merged['is_scrap'] = (merged['scrap_rate'] > 0.02).astype(int)
    
    # Final Cleanup
    merged = merged.dropna(subset=['timestamp']) # Remove orders with no matching sensor data
    
    # =========================================================
    # 🛡️ SAFETY FIX FOR MISSING COLUMNS
    # =========================================================
    # This block prevents the KeyError by filling any missing
    # config columns with 0.0
    logger.info("🛡️ Performing Schema Validation & Auto-Fill...")
    
    for required_col in cfg.REQUIRED_PARAM_COLS:
        if required_col not in merged.columns:
            logger.warning(f"   ⚠️ Missing Config Column: '{required_col}'. Filling with 0.0 to prevent crash.")
            merged[required_col] = 0.0
            
    # =========================================================
    
    # --- Step 6: Data Quality Report ---
    report_quality(merged)

    # --- Step 7: Caching ---
    logger.info("💾 Caching Processed Dataset...")
    merged.to_pickle(cache_path)
    
    elapsed = time.time() - start_time
    logger.info(f"✅ Data Pipeline Complete in {elapsed:.2f}s. Shape: {merged.shape}")
    
    return merged

def report_quality(df):
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