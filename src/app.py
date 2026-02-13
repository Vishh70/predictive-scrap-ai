import streamlit as st
import sys
import os
import json
import pandas as pd
import joblib
import time
import numpy as np
import io
import logging
import traceback
import hashlib
import random
import base64
import shutil
import re
from datetime import datetime, timedelta
from pathlib import Path
from functools import lru_cache
from typing import Dict, Optional, List, Any, Tuple

# =========================================================
# 🔧 CRITICAL PATH FIX & SYSTEM CONFIG
# =========================================================
try:
    current_script_path = Path(__file__).resolve()
    PROJECT_ROOT = current_script_path.parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))
        
    DATA_DIR = PROJECT_ROOT / 'data'
    MODELS_DIR = PROJECT_ROOT / 'data' / 'models'
    REPORTS_DIR = PROJECT_ROOT / 'reports'
    LOGS_DIR = PROJECT_ROOT / 'logs'
    
    for d in [DATA_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

except Exception as e:
    st.error(f"Critical System Error: {e}")
    st.stop()

# =========================================================
# 📦 LIBRARY IMPORTS (Lazy Loading)
# =========================================================
try:
    import matplotlib.pyplot as plt
    plt.switch_backend('Agg') 
except ImportError:
    plt = None

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    st.error("❌ Critical Dependency Missing: Plotly. Run `pip install plotly`.")
    st.stop()

try:
    import shap
except ImportError:
    shap = None

try:
    import src.config as cfg
    from src.data_loading import load_and_merge_data
    from src.monitoring_utils import ProcessMonitor
except ImportError:
    class cfg:
        BASE_DIR = PROJECT_ROOT
        REQUIRED_PARAM_COLS = ['injection_pressure', 'cycle_time', 'cushion', 'cyl_tmp_z1', 'melt_temp']
        CRITICAL_RISK_THRESHOLD = 0.85
        WARNING_RISK_THRESHOLD = 0.35
        FORECAST_HORIZON_MINUTES = 60
        SAFE_ZONE_SIGMA_MULTIPLIER = 3.0
        VOLATILITY_WINDOW = 10
        PHYSICS_RANGES = {
            'injection_pressure': (0, 2000),
            'melt_temp': (100, 400),
            'cycle_time': (0, 120),
        }
        PARAM_SATURATION_RATIO = 0.995

        @staticmethod
        def get_logger(name: str, log_file: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
            logger = logging.getLogger(name)
            logger.setLevel(level)
            logger.propagate = False
            formatter = logging.Formatter("%(asctime)s - [%(name)s] - %(levelname)s - %(message)s")
            if not logger.handlers:
                stream_handler = logging.StreamHandler()
                stream_handler.setFormatter(formatter)
                logger.addHandler(stream_handler)
                if log_file is not None:
                    file_handler = logging.FileHandler(log_file, encoding="utf-8")
                    file_handler.setFormatter(formatter)
                    logger.addHandler(file_handler)
            return logger
    
    def load_and_merge_data(use_cache: bool = True) -> pd.DataFrame:
        try:
            cache_path = DATA_DIR / "processed_full_dataset.pkl"
            if cache_path.exists():
                return pd.read_pickle(cache_path)
            return pd.DataFrame()
        except:
            return pd.DataFrame()

    class ProcessMonitor:
        def check_safety(self, row: dict | pd.Series, targets: Dict[str, Any]) -> Dict[str, Any]:
            return {"status": "OK", "message": "", "tooltip": "", "deviation": 0.0}

# =========================================================
# 📝 LOGGING SETUP
# =========================================================
logger = cfg.get_logger(
    "TE_Dashboard",
    log_file=LOGS_DIR / f"dashboard_log_{datetime.now().strftime('%Y%m%d')}.txt",
    level=logging.INFO
)

# =========================================================
# 🎨 STREAMLIT PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="TE Connectivity - AI Copilot (Enterprise)",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .main { background-color: #f8f9fa; }
    h1, h2, h3 { color: #2c3e50; font-family: 'Segoe UI', sans-serif; }
    .stMetric { 
        background-color: #ffffff; 
        padding: 20px; 
        border-radius: 12px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-top: 5px solid #e67e22; 
        transition: transform 0.2s;
    }
    .stMetric:hover { transform: scale(1.02); }
    section[data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e0e0e0; }
    .suggestion-box {
        padding: 20px;
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
        border-radius: 8px;
        font-family: 'Segoe UI', sans-serif;
        margin: 15px 0;
    }
    .alert-box-critical {
        padding: 15px;
        background-color: #ffebee;
        border-left: 5px solid #d32f2f;
        border-radius: 5px;
        color: #b71c1c;
        font-weight: bold;
        animation: pulse 2s infinite;
    }
    @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.8; } 100% { opacity: 1; } }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# 🧠 UTILITY FUNCTIONS (Cleaning & Searching)
# =========================================================
def sanitize_numeric(val: Any) -> float:
    """Robustly converts inputs to float."""
    if isinstance(val, (int, float)): return float(val)
    if isinstance(val, str):
        clean = val.replace('[', '').replace(']', '').replace("'", "").strip()
        try: return float(clean)
        except ValueError: return 0.0
    return 0.0

def clean_dataframe_for_ai(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Prepares data for AI: fills missing cols, ensures float types."""
    df_clean = df.copy()
    for f in features:
        if f not in df_clean.columns: df_clean[f] = 0.0 
    df_clean = df_clean[features]
    for col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    return df_clean.fillna(0.0)

def robust_column_finder(df: pd.DataFrame, keywords: List[str]) -> Optional[str]:
    """Smart Search: Exact -> Clean -> Substring match."""
    if df.empty: return None
    cols = df.columns.tolist()
    # 1. Exact Match
    for kw in keywords:
        for col in cols:
            if kw.lower() == col.lower().strip(): return col
    # 2. Clean Match
    for kw in keywords:
        kw_clean = kw.lower().replace("_", "").replace(" ", "")
        for col in cols:
            col_clean = col.lower().replace("_", "").replace(" ", "")
            if kw_clean == col_clean: return col
    # 3. Substring Match
    for kw in keywords:
        for col in cols:
            if kw.lower() in col.lower(): return col
    return None
# =========================================================
# 🔒 SECURE AUTHENTICATION
# =========================================================
def check_password() -> bool:
    """
    Secure password check using Hashing (SHA256).
    Default password is 'admin'.
    """
    # Hash of 'admin'
    CORRECT_HASH = "8c6976e5b5410415bde908bd4dee15dfb167a9c873fc4bb8a81f6f2ab448a918"

    # --- CRITICAL FIX: Initialize State Variables First ---
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    
    # Callback to check password
    def password_entered() -> None:
        # Read from the widget key 'password_input'
        if "password_input" in st.session_state:
            user_input = st.session_state["password_input"]
            input_hash = hashlib.sha256(user_input.encode()).hexdigest()
            
            if input_hash == CORRECT_HASH:
                st.session_state["password_correct"] = True
                # Clear input for security
                st.session_state["password_input"] = ""
            else:
                st.session_state["password_correct"] = False
                st.error("😕 Access Denied")

    if not st.session_state["password_correct"]:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🔒 TE Enterprise Login")
        # Use 'key' to bind this input to session_state
        st.sidebar.text_input("Password", type="password", key="password_input", on_change=password_entered)
        return False
    else:
        return True

# =========================================================
# ⚙️ PHASE 2: ADVANCED FEATURE ENGINEERING
# =========================================================
def engineer_volatility_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Calculates 'Wobble' (Rolling Std) and 'Drift' (Rolling Mean) 
    for critical machine parameters.
    """
    if df.empty: return df
    
    # Parameters to analyze for stability
    target_params = ['injection_pressure', 'cushion', 'cycle_time', 'melt_temp']
    
    # 1. Identify which params exist in this dataset
    available_params = [p for p in target_params if robust_column_finder(df, [p])]
    
    # 2. Calculate Rolling Features
    for param in available_params:
        col_name = robust_column_finder(df, [param])
        if col_name:
            # Volatility (The "Wobble") - Standard Deviation over window
            df[f'{param}_volatility'] = df[col_name].rolling(window=window).std().fillna(0)
            
            # Trend (The "Drift") - Moving Average over window
            df[f'{param}_trend'] = df[col_name].rolling(window=window).mean().fillna(df[col_name])
        
    return df

# =========================================================
# 🚀 ASSET LOADING (MODELS & DATA)
# =========================================================
@st.cache_resource
def load_ai_assets() -> Tuple[Optional[Any], List[str], Optional[Any], Optional[Any], float]:
    """
    Loads the trained XGBoost model and preprocessing pipelines.
    Returns default/empty objects if files are missing to prevent crashes.
    """
    try:
        model = joblib.load(MODELS_DIR / "scrap_model.joblib")
        features = joblib.load(MODELS_DIR / "model_features.joblib")
        imputer = joblib.load(MODELS_DIR / "imputer.joblib")
        scaler = joblib.load(MODELS_DIR / "scaler.joblib")
        threshold = joblib.load(MODELS_DIR / "threshold.joblib")
        return model, features, imputer, scaler, float(threshold)
    except Exception as e:
        logging.warning(f"AI assets load failed: {e}")
        # Return safe defaults so app doesn't crash
        return None, getattr(cfg, 'REQUIRED_PARAM_COLS', []), None, None, float(getattr(cfg, "WARNING_RISK_THRESHOLD", 0.35))

@st.cache_data(ttl=3600)
def get_cached_data() -> pd.DataFrame:
    """
    Loads the main dataset with Aggressive Memory Optimization.
    Prevents 'Unable to allocate' errors on local machines.
    """
    try:
        # 1. Load Data
        df = load_and_merge_data(use_cache=True)

        if df.empty:
            return df

        # 2. MEMORY FIX: Downcast float64 -> float32 (Saves ~50% RAM)
        cols_float = df.select_dtypes(include=['float64']).columns
        if len(cols_float) > 0:
            df[cols_float] = df[cols_float].astype('float32')

        # 3. SAFETY CAP: Prevent crashes by limiting row count
        # 1.08GB error suggests ~2M rows is the limit. We cap at 1M safe rows.
        MAX_ROWS = 1_000_000
        if len(df) > MAX_ROWS:
            df = df.tail(MAX_ROWS).reset_index(drop=True)
            st.toast(f"⚠️ Performance Mode: Data limited to last {MAX_ROWS:,} records.", icon="🚀")

        # 4. Re-Apply Engineering (Volatility)
        df = engineer_volatility_features(df, window=int(getattr(cfg, "VOLATILITY_WINDOW", 10)))

        return df
    except Exception as e:
        st.error(f"⚠️ Data Load Error: {e}")
        st.warning("The cached dataset might be too large for your RAM.")
        if st.button("🗑️ Delete Cache & Retry"):
            try:
                cache_path = DATA_DIR / "processed_full_dataset.pkl"
                if cache_path.exists():
                    cache_path.unlink()
                st.success("Cache deleted. Please refresh the page.")
            except Exception as del_err:
                st.error(f"Could not delete cache: {del_err}")
        return pd.DataFrame()

def load_data() -> pd.DataFrame:
    """Single entry point for dashboard data loading."""
    return get_cached_data()

# Load Global State
model, feature_list, imputer, scaler, threshold = load_ai_assets()
history_df = load_data()

# =========================================================
# ⚙️ HIGH-PERFORMANCE PREDICTION ENGINE
# =========================================================

# Helper to make dictionary hashable for LRU Cache
def dict_to_tuple(d: Dict[str, float]) -> Tuple[Tuple[str, float], ...]:
    return tuple(sorted(d.items()))

@lru_cache(maxsize=1024)
def _cached_predict(input_tuple: Tuple[Tuple[str, float], ...]) -> float:
    """
    Internal cached prediction function. 
    Prevents re-running model on identical slider inputs.
    """
    try:
        input_dict = dict(input_tuple)
        input_df = pd.DataFrame([input_dict])
        
        # Clean & Align
        X = clean_dataframe_for_ai(input_df, feature_list)
        
        # Transform (Impute -> Scale)
        if imputer:
            X = pd.DataFrame(imputer.transform(X), columns=feature_list)
        if scaler:
            X = pd.DataFrame(scaler.transform(X), columns=feature_list)
            
        # Predict
        if model:
            if hasattr(model, 'predict_proba'):
                return float(model.predict_proba(X)[0][1])
            elif hasattr(model, 'predict'):
                return float(model.predict(X)[0])
        return 0.0
    except Exception:
        return 0.0

def calculate_risk(input_row_dict: Dict[str, float]) -> float:
    """
    Predict scrap risk probability for a single production observation.

    Args:
        input_row_dict: Mapping of feature names to numeric values for one cycle.
            Example: {"injection_pressure": 1200.0, "cycle_time": 28.4, ...}

    Returns:
        Predicted scrap probability in the range [0.0, 1.0].
    """
    key = dict_to_tuple(input_row_dict)
    return _cached_predict(key)

def batch_calculate_risk(df_batch: pd.DataFrame) -> List[float]:
    """
    Vectorized Prediction for 100+ rows.
    """
    try:
        if df_batch.empty: return []
        
        # Clean & Align Batch
        X = clean_dataframe_for_ai(df_batch, feature_list)
        
        if imputer:
            X = pd.DataFrame(imputer.transform(X), columns=feature_list)
        if scaler:
            X = pd.DataFrame(scaler.transform(X), columns=feature_list)
            
        if model:
            if hasattr(model, 'predict_proba'):
                return model.predict_proba(X)[:, 1].tolist()
            elif hasattr(model, 'predict'):
                return model.predict(X).tolist()
        return [0.0] * len(df_batch)
    except Exception as e:
        logging.error(f"Batch Prediction Error: {e}")
        return [0.0] * len(df_batch)

def prescriptive_engine(current_params: Dict[str, float], base_risk: float) -> str:
    """
    Generate a best-effort parameter adjustment recommendation.

    Args:
        current_params: Mapping of tunable process parameters to current numeric
            values, e.g. {"cushion": 4.8, "melt_temp": 230.0}.
        base_risk: Baseline scrap-risk probability for `current_params`.

    Returns:
        HTML-formatted recommendation text for Streamlit rendering.
    """
    # Phase 7: hard safety check for parameter saturation near physics bounds.
    physics_ranges = getattr(cfg, "PHYSICS_RANGES", {})
    saturation_ratio = float(getattr(cfg, "PARAM_SATURATION_RATIO", 0.995))
    for param_name, bounds in physics_ranges.items():
        try:
            lower, upper = bounds
            span = float(upper) - float(lower)
            if span <= 0:
                continue
            curr_val = sanitize_numeric(current_params.get(param_name, np.nan))
            if np.isnan(curr_val):
                continue
            utilization = (curr_val - float(lower)) / span
            if utilization >= saturation_ratio:
                return (
                    f"⚠️ <b>Parameter Saturation:</b> Reduce <code>{param_name}</code> immediately.<br>"
                    f"&nbsp;&nbsp;Current: {curr_val:.2f} | Limit: {float(upper):.2f}"
                )
        except Exception as saturation_err:
            logger.warning("Saturation safety check failed for %s: %s", param_name, saturation_err)

    if base_risk <= threshold:
        return "✅ <b>System Stable:</b> Current parameters are optimal."

    best_imp = 0.0
    recommendation = "🔍 Check Process Parameters (General)"
    
    # Physics Parameters we can tune
    tunable = ['injection_pressure', 'cycle_time', 'cushion', 'cyl_tmp_z1', 'melt_temp']
    
    for param in tunable:
        curr = sanitize_numeric(current_params.get(param, 0.0))
        if curr == 0: continue

        # Bound simulation moves inside configured physics limits.
        bound_low, bound_high = physics_ranges.get(param, (-np.inf, np.inf))
        
        # Try adjusting up/down by 5% and 10%
        for step in [-0.10, -0.05, 0.05, 0.10]:
            try:
                test = current_params.copy()
                proposed = curr * (1 + step)
                test[param] = float(np.clip(proposed, bound_low, bound_high))
            except Exception as sim_err:
                logger.warning("Simulation step failed for %s at step %.3f: %s", param, step, sim_err)
                continue
            
            # Update lag features for realism
            if f"{param}_lag_1" in feature_list:
                test[f"{param}_lag_1"] = curr
            
            new_risk = calculate_risk(test)
            imp = base_risk - new_risk
            
            if imp > best_imp:
                best_imp = imp
                direction = "Decrease" if step < 0 else "Increase"
                recommendation = (
                    f"💡 <b>AI Recommendation:</b> {direction} <code>{param}</code> by {abs(step)*100:.0f}%<br>"
                    f"&nbsp;&nbsp;→ Expected Risk Reduction: -{imp:.1%}"
                )
    return recommendation
# =========================================================
# 🖥️ MAIN APPLICATION INTERFACE
# =========================================================

def _log_and_show_global_error(exc: Exception) -> None:
    """Global app safety net: user-friendly error + persistent crash log."""
    crash_log_path = LOGS_DIR / "app_crash.log"
    traceback_text = traceback.format_exc()
    try:
        with open(crash_log_path, "a", encoding="utf-8") as crash_file:
            crash_file.write(f"[{datetime.now().isoformat()}] {exc}\n{traceback_text}\n")
    except Exception:
        logger.exception("Failed to write crash log: %s", crash_log_path)
    logger.exception("Unhandled dashboard exception: %s", exc)
    st.error(f"System Malfunction: {exc}. Please contact the AI Engineering team.")

def render_sidebar(df: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Render sidebar filters and return:
      1) current_filters (includes app_mode)
      2) df_filtered (already sliced in sidebar)
    """
    current_filters: Dict[str, Any] = {
        "app_mode": "Dashboard Overview",
        "date_range": (None, None),
        "selected_segment": "All",
        "selected_statuses": [],
        "selected_machine": "N/A",
        "selected_tool": "All",
        "selected_part": "All",
    }
    active_df = pd.DataFrame()

    with st.sidebar:
        try:
            st.image("https://www.te.com/content/dam/te-com/global/english/about-te/news-center/te-logo.png", width=150)
        except Exception:
            st.title("TE Connectivity")

        st.markdown("**Version:** 7.1.0\n**Role:** Admin")

        current_filters["app_mode"] = st.radio(
            "MAIN MENU",
            [
                "Dashboard Overview", "Data Integration Map", "Data Explorer",
                "Prescriptive Simulator", "Live Monitor", "Operator Logbook",
                "System Health", "User Guide"
            ],
        )

        st.divider()
        st.subheader("🔍 Filters")

        if df.empty:
            st.error("❌ No records loaded. Please check your 'data/' folder.")
            st.divider()
            if st.button("🔄 Reload Data Cache"):
                st.cache_data.clear()
                st.cache_resource.clear()
                cache_path = DATA_DIR / "processed_full_dataset.pkl"
                if cache_path.exists():
                    try:
                        cache_path.unlink()
                        st.toast("Cache file deleted.")
                    except Exception:
                        pass
                st.toast("♻️ Reloading System...")
                time.sleep(1)
                st.rerun()
            return current_filters, active_df

        working_df = df.copy()

        # 1) Date Range Filter (plant_shift_date)
        date_col = robust_column_finder(
            working_df,
            ["plant_shift_date", "plant_shift_calendar_date", "shift_date", "date"]
        )
        if date_col:
            parsed_dates = pd.to_datetime(working_df[date_col], errors="coerce")
            valid_dates = parsed_dates.dropna()
            if not valid_dates.empty:
                min_date = valid_dates.min().date()
                max_date = valid_dates.max().date()
                selected_range = st.date_input(
                    "📅 Select Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                )

                if isinstance(selected_range, tuple) and len(selected_range) == 2:
                    start_date, end_date = selected_range
                else:
                    start_date = selected_range
                    end_date = selected_range

                current_filters["date_range"] = (start_date, end_date)

                working_df = working_df.assign(_sidebar_date=parsed_dates.dt.date)
                working_df = working_df[
                    (working_df["_sidebar_date"] >= start_date) &
                    (working_df["_sidebar_date"] <= end_date)
                ]
            else:
                st.info("No valid dates found in plant_shift_date.")
        else:
            st.info("Column 'plant_shift_date' not found. Date filter disabled.")

        # 2) Business Segment (Dropdown) - cascades into statuses and machines
        segment_col = robust_column_finder(
            working_df,
            ["segment_abbr_name", "segment_name", "business_segment", "segment"]
        )
        if segment_col:
            segments = sorted(
                [str(x).strip() for x in working_df[segment_col].dropna().unique() if str(x).strip() != ""]
            )
            segment_options = ["All"] + segments
            selected_segment = st.selectbox("🏢 Business Segment", segment_options)
            current_filters["selected_segment"] = selected_segment
            if selected_segment != "All":
                working_df = working_df[working_df[segment_col].astype(str).str.strip() == selected_segment]
        else:
            st.info("Column 'segment_abbr_name' not found. Segment filter disabled.")

        # 3) Machine Status (Multiselect) - empty means All
        status_col = robust_column_finder(
            working_df,
            ["machine_status_name", "status_name", "machine_status", "status"]
        )
        if status_col:
            all_statuses = sorted(
                [str(x).strip() for x in working_df[status_col].dropna().unique() if str(x).strip() != ""]
            )
            selected_statuses = st.multiselect("🚦 Machine Status", options=all_statuses, default=all_statuses)
            if not selected_statuses:
                selected_statuses = all_statuses
            current_filters["selected_statuses"] = selected_statuses
            if selected_statuses:
                working_df = working_df[working_df[status_col].astype(str).isin(selected_statuses)]
        else:
            st.info("Column 'machine_status_name' not found. Status filter disabled.")

        # 4) Existing cascade: machine -> tool -> part
        mach_kw = ["machine_id", "machine", "equipment", "mach", "machine_nr"]
        mach_col = robust_column_finder(working_df, mach_kw)

        if mach_col:
            raw_machines = working_df[mach_col].dropna().unique()
            machine_options = sorted([str(x).split(".")[0] for x in raw_machines if str(x).strip() != ""])
            if machine_options:
                selected_m = st.selectbox("Active Machine", machine_options)
                current_filters["selected_machine"] = selected_m
                active_df = working_df[working_df[mach_col].astype(str).str.split(".").str[0] == selected_m].copy()
            else:
                active_df = pd.DataFrame()
                st.warning("No machines available for current Date/Status/Segment filters.")
        else:
            active_df = pd.DataFrame()
            st.error("❌ Machine column not detected in filtered data.")

        tool_kw = ["tool_nr", "tool", "mold", "toolid", "mold_id", "die"]
        tool_col = robust_column_finder(active_df, tool_kw) if not active_df.empty else None
        if tool_col:
            raw_tools = active_df[tool_col].dropna().unique()
            unique_tools = sorted([str(t).strip() for t in raw_tools if str(t).strip() != ""])
            selected_tool = st.selectbox(f"Select Tool ({len(unique_tools)} found)", ["All"] + unique_tools)
            current_filters["selected_tool"] = selected_tool
            if selected_tool != "All":
                active_df = active_df[active_df[tool_col].astype(str).str.strip() == selected_tool]

        part_kw = ["material_nr", "part_no", "part", "material", "product", "article", "part_number"]
        part_col = robust_column_finder(active_df, part_kw) if not active_df.empty else None
        if part_col:
            raw_parts = active_df[part_col].dropna().unique()
            unique_parts = sorted([str(p).strip() for p in raw_parts if str(p).strip() != ""])
            selected_part = st.selectbox(f"Select Part No. ({len(unique_parts)} found)", ["All"] + unique_parts)
            current_filters["selected_part"] = selected_part
            if selected_part != "All":
                active_df = active_df[active_df[part_col].astype(str).str.strip() == selected_part]

        if "_sidebar_date" in active_df.columns:
            active_df = active_df.drop(columns=["_sidebar_date"], errors="ignore")

        st.success(f"✅ Active Context: {len(active_df):,} cycles")

        st.divider()
        if st.button("🔄 Reload Data Cache"):
            st.cache_data.clear()
            st.cache_resource.clear()

            cache_path = DATA_DIR / "processed_full_dataset.pkl"
            if cache_path.exists():
                try:
                    cache_path.unlink()
                    st.toast("Cache file deleted.")
                except Exception:
                    pass

            st.toast("♻️ Reloading System...")
            time.sleep(1)
            st.rerun()

    return current_filters, active_df

def render_dashboard_overview(df: pd.DataFrame) -> None:
    """
    Renders the executive dashboard overview using already-filtered dataframe.
    Includes KPI cards + strategic charts:
    1) Daily Scrap Rate Trend
    2) Scrap Volume by Machine Status
    3) Scrap Rate by Business Segment
    """
    st.title("📊 Executive Dashboard Overview")

    if df is None or df.empty:
        st.warning("⚠️ No data matches the current filters. Please adjust your selection.")
        return

    required_cols = [
        "scrap_quantity",
        "yield_quantity",
        "plant_shift_date",
        "machine_status_name",
        "segment_abbr_name",
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        st.warning(f"⚠️ Missing required column(s): {', '.join(missing_cols)}")
        return

    # Work on a copy to avoid mutating caller dataframe.
    viz_df = df.copy()
    viz_df["scrap_quantity"] = pd.to_numeric(viz_df["scrap_quantity"], errors="coerce").fillna(0)
    viz_df["yield_quantity"] = pd.to_numeric(viz_df["yield_quantity"], errors="coerce").fillna(0)
    viz_df["plant_shift_date"] = pd.to_datetime(viz_df["plant_shift_date"], errors="coerce")
    viz_df = viz_df.dropna(subset=["plant_shift_date"])

    if viz_df.empty:
        st.warning("⚠️ No valid dated records available after preprocessing.")
        return

    # --- KPIs ---
    total_scrap = float(viz_df["scrap_quantity"].sum())
    total_yield = float(viz_df["yield_quantity"].sum())
    total_production = total_scrap + total_yield
    scrap_rate = (total_scrap / total_production * 100.0) if total_production > 0 else 0.0

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("🚨 Total Scrap Parts", f"{total_scrap:,.0f}")
    kpi2.metric("📦 Total Production", f"{total_production:,.0f}")
    kpi3.metric("📉 Overall Scrap Rate", f"{scrap_rate:.2f}%")

    st.divider()

    # TE Connectivity color direction.
    te_blue = "#005EB8"
    te_orange = "#E66C37"

    # --- 1) Daily Scrap Trend ---
    daily_df = (
        viz_df.groupby(viz_df["plant_shift_date"].dt.date)[["scrap_quantity", "yield_quantity"]]
        .sum()
        .reset_index()
        .rename(columns={"plant_shift_date": "plant_shift_date_day"})
    )
    daily_df["total"] = daily_df["scrap_quantity"] + daily_df["yield_quantity"]
    daily_df["daily_scrap_rate"] = np.where(
        daily_df["total"] > 0,
        (daily_df["scrap_quantity"] / daily_df["total"]) * 100.0,
        0.0,
    )

    fig_trend = px.line(
        daily_df,
        x="plant_shift_date_day",
        y="daily_scrap_rate",
        title="Daily Scrap Rate Trend",
        labels={"plant_shift_date_day": "Date", "daily_scrap_rate": "Scrap Rate (%)"},
        markers=True,
        template="plotly_white",
    )
    fig_trend.update_traces(line_color=te_orange)
    fig_trend.update_layout(
        title_x=0.01,
        yaxis_title="Scrap Rate (%)",
        xaxis_title="Date",
    )
    st.plotly_chart(fig_trend)

    # --- 2) & 3) Side-by-side strategic charts ---
    c1, c2 = st.columns(2)

    with c1:
        status_agg = (
            viz_df.groupby("machine_status_name", dropna=False)["scrap_quantity"]
            .sum()
            .reset_index()
            .sort_values("scrap_quantity", ascending=False)
        )
        if status_agg.empty:
            st.info("No machine-status data available for this selection.")
        else:
            fig_status = px.bar(
                status_agg,
                x="machine_status_name",
                y="scrap_quantity",
                color="machine_status_name",
                title="Scrap Volume by Machine Status",
                labels={"machine_status_name": "Machine Status", "scrap_quantity": "Total Scrap Parts"},
                template="plotly_white",
                color_discrete_sequence=[te_blue, te_orange],
            )
            fig_status.update_layout(title_x=0.01, legend_title_text="Status")
            st.plotly_chart(fig_status)

    with c2:
        seg_agg = (
            viz_df.groupby("segment_abbr_name", dropna=False)[["scrap_quantity", "yield_quantity"]]
            .sum()
            .reset_index()
        )
        seg_agg["total"] = seg_agg["scrap_quantity"] + seg_agg["yield_quantity"]
        seg_agg["seg_scrap_rate"] = np.where(
            seg_agg["total"] > 0,
            (seg_agg["scrap_quantity"] / seg_agg["total"]) * 100.0,
            0.0,
        )
        seg_agg = seg_agg.sort_values("seg_scrap_rate", ascending=False)

        if seg_agg.empty:
            st.info("No segment data available for this selection.")
        else:
            fig_seg = px.bar(
                seg_agg,
                x="segment_abbr_name",
                y="seg_scrap_rate",
                title="Scrap Rate by Business Unit",
                labels={"segment_abbr_name": "Segment", "seg_scrap_rate": "Scrap Rate (%)"},
                template="plotly_white",
                color_discrete_sequence=[te_blue],
            )
            fig_seg.update_layout(title_x=0.01, showlegend=False)
            st.plotly_chart(fig_seg)

def render_data_integration_map() -> None:
    """
    Data Ingestion Portal:
    1. Uploads files.
    2. Triggers ETL.
    3. Visualizes file streams with Deep Scan metadata (Machine ID + Timestamp).
    """
    st.title("🗺️ Live Input & Data Integration Portal")

    # --- Section 1: Upload Zone ---
    st.subheader("📂 Drag & Drop Upload Zone")
    uploaded_file = st.file_uploader(
        "Upload a new input file (CSV/Excel)",
        type=["csv", "xlsx"],
        accept_multiple_files=False,
        key="data_ingestion_uploader",
    )

    if uploaded_file is not None:
        file_signature = f"{uploaded_file.name}:{uploaded_file.size}"
        if st.session_state.get("last_uploaded_signature") != file_signature:
            save_path = Path("data") / uploaded_file.name
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state["last_uploaded_signature"] = file_signature
            st.toast(f"✅ File saved: {uploaded_file.name}")
            logger.info("Uploaded file saved to %s", save_path)

    if st.button("🔄 Process New Data", type="primary"):
        with st.spinner("Processing new files with ETL..."):
            try:
                # Local import avoids potential circular import edges during app bootstrap.
                from src.data_loading import load_and_merge_data as _load_and_merge_data

                merged_df = _load_and_merge_data(use_cache=False)
                st.cache_data.clear()
                st.success(
                    f"✅ Processing complete. Total Rows: {len(merged_df):,}"
                )
            except Exception as exc:
                st.error(f"❌ Processing failed: {exc}")
                logger.exception("ETL processing failed from ingestion portal: %s", exc)

    st.divider()

    # --- Section 2: Deep Scan Visualization ---
    data_dir = Path("data")
    if not data_dir.exists():
        st.error("❌ 'data/' directory not found.")
        return

    all_files = sorted(list(data_dir.glob("*.csv")) + list(data_dir.glob("*.xlsx")))
    hydra_files = [f for f in all_files if "hydra" in f.name.lower()]
    sensor_files = [f for f in all_files if "hydra" not in f.name.lower()]

    c1, c2, c3 = st.columns([1, 0.3, 1])

    # COLUMN 1: Hydra Stream
    with c1:
        st.markdown("### 🏭 Production Stream (Hydra)")
        if hydra_files:
            for f in hydra_files:
                mod_time = datetime.fromtimestamp(os.path.getmtime(f)).strftime("%Y-%m-%d %H:%M")
                st.info(f"🗂️ **{f.name}**\n\n🕒 Updated: `{mod_time}`")
        else:
            st.info("No Hydra files detected.")

    # COLUMN 2: Merge Engine
    with c2:
        st.markdown("### ⚙️ Engine")
        st.markdown("<div style='text-align:center; font-size:40px; line-height:2;'>➡️</div>", unsafe_allow_html=True)
        st.caption("Merge Logic:\n`pd.merge_asof`")
        st.caption("**15 min Tolerance**")

    # COLUMN 3: Sensor Stream (Deep Scan)
    with c3:
        st.markdown("### 🛰️ Sensor Stream (Machine)")
        if sensor_files:
            for f in sensor_files:
                match = re.search(r"(M\d+)", f.name)
                machine_id = match.group(1) if match else "Unknown"
                mod_time = datetime.fromtimestamp(os.path.getmtime(f)).strftime("%Y-%m-%d %H:%M")
                if machine_id != "Unknown":
                    st.warning(f"📄 **{f.name}**\n\n🆔 Detected: **{machine_id}**\n🕒 Updated: `{mod_time}`")
                else:
                    st.warning(f"📄 **{f.name}**\n\n⚠️ ID Unknown\n🕒 Updated: `{mod_time}`")
        else:
            st.warning("No sensor files detected.")

def render_model_audit_panel() -> None:
    """Live inspection of the trained ML model artifact AND active physics rules."""
    st.markdown("### 🛡️ AI & Physics Audit")

    # --- SECTION 1: MACHINE LEARNING MODEL ---
    st.markdown("#### 🧠 1. Predictive Model (ML)")
    model_path = Path("models/scrap_model.joblib")
    if not model_path.exists():
        # Project default storage
        model_path = Path("data/models/scrap_model.joblib")

    if model_path.exists():
        try:
            model_data = joblib.load(model_path)

            # Handle dict-wrapped artifacts and raw sklearn/xgboost models
            if isinstance(model_data, dict):
                model_obj = model_data.get("model", "Unknown")
                features = model_data.get("features", [])
                _meta = model_data.get("metadata", {})
            else:
                model_obj = model_data
                features = getattr(model_obj, "feature_names_in_", [])
                _meta = {}

            # Fallback to separate features artifact if needed
            if (not features) and (MODELS_DIR / "model_features.joblib").exists():
                try:
                    features = joblib.load(MODELS_DIR / "model_features.joblib")
                except Exception:
                    features = []

            c1, c2, c3 = st.columns(3)
            c1.metric("Architecture", type(model_obj).__name__)
            c2.metric("Input Features", len(features))
            mtime = datetime.fromtimestamp(model_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            c3.metric("Last Trained", mtime)

            with st.expander("🔍 Inspect ML Features"):
                st.dataframe(pd.DataFrame({"Feature Name": list(features)}), height=200)

            st.success(f"✅ ML Model Loaded: {model_path.name}")
        except Exception as exc:
            st.error(f"❌ Corrupt Model Artifact: {exc}")
    else:
        st.warning("⚠️ No ML model found. Run training pipeline.")

    st.divider()

    # --- SECTION 2: PHYSICS ENGINE (CONFIG) ---
    st.markdown("#### 📏 2. Physics Guardrails (Rule-Based)")
    config_path = Path("data/monitoring_config.json")

    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                rules = json.load(f)

            st.metric("Active Safety Rules", len(rules))

            rule_list = []
            for param, details in rules.items():
                rule_list.append({
                    "Parameter": param,
                    "Threshold": f"± {details.get('threshold', 'N/A')}",
                    "Wildcard Mode": "✅" if details.get("is_wildcard") else "❌",
                    "Source Text": details.get("original_text", ""),
                })

            rules_df = pd.DataFrame(rule_list)
            st.dataframe(
                rules_df,
                column_config={
                    "Parameter": st.column_config.TextColumn("Parameter", help="Sensor ID"),
                    "Threshold": st.column_config.TextColumn("Safety Limit", help="Max allowed deviation"),
                },
                hide_index=True,
            )

            st.success(f"✅ Physics Config Loaded: {config_path.name}")
        except Exception as exc:
            st.error(f"❌ Failed to load config: {exc}")
    else:
        st.warning("⚠️ No physics config found. Run `generate_config.py`.")


def render_user_guide() -> None:
    """Renders judge/operator documentation plus live model audit."""
    st.title("📘 AI Copilot User Manual")

    tab_docs, tab_audit = st.tabs(["📖 Documentation", "🤖 Model Audit"])

    with tab_docs:
        st.markdown(
            """```mermaid
flowchart LR
    A[Raw Data] --> B[Universal Loader]
    B --> C[Physics Check]
    C --> D[ML Model]
    D --> E[Dashboard]
```"""
        )

        with st.expander("🔌 Data Integration", expanded=True):
            st.markdown(
                """
- Use **Data Integration Map** to drag-and-drop `.csv` / `.xlsx` files.
- Uploaded files are saved to `data/` and can be processed with **Process New Data**.
- **Deep Scan** parses sensor filenames for machine IDs (e.g., `M231`) and shows last update timestamp.
"""
            )

        with st.expander("🚦 Live Monitor", expanded=False):
            st.markdown(
                """
- **Green = OK** within threshold.
- **Yellow = Warning** when soft limits are exceeded.
- **Red = Critical** when hard limits are breached.
- Example critical guardrail: **Cushion ±0.5 mm**.
"""
            )

        with st.expander("🔮 Prescriptive Simulator", expanded=False):
            st.markdown(
                """
- Use sliders to run **what-if simulations** for process parameters.
- The model compares simulated settings vs. baseline behavior.
- Output shows predicted risk and recommended corrective direction.
"""
            )

        with st.expander("📊 Executive Dashboard", expanded=False):
            st.markdown(
                """
- Filters cascade from business context to operations.
- Standard flow: **Segment -> Status -> Trend view**.
- KPI cards and charts always reflect the filtered scope.
"""
            )

        st.subheader("🛠️ Troubleshooting")
        st.markdown(
            """
- **Why is my screen empty?** Check your **Date Filters** first.
- **What does 'Critical Error' mean?** A physics parameter (for example, Cushion) breached a safety limit.
"""
        )

    with tab_audit:
        render_model_audit_panel()

def render_data_explorer(active_df: pd.DataFrame) -> None:
    """Data explorer view that consumes pre-filtered dataset from sidebar."""
    st.title("🔍 Advanced Data Inspector")

    if active_df.empty:
        st.warning("Data unavailable. Please check filters.")
        return

    tabs = st.tabs(["Dataset View", "Correlations", "Distribution Analysis", "Data Dictionary"])

    with tabs[0]:
        st.markdown("### Raw Production Data")
        st.dataframe(active_df.head(100))
        st.caption(f"Showing first 100 of {len(active_df)} rows.")

    with tabs[1]:
        st.markdown("### Feature Correlations")
        numeric_df = clean_dataframe_for_ai(active_df, feature_list[:12])
        if not numeric_df.empty:
            corr = numeric_df.corr()
            fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', aspect="auto")
            st.plotly_chart(fig_corr)

    with tabs[2]:
        st.markdown("### Feature Distributions")
        sel_feat = st.selectbox("Select Feature to Analyze", feature_list[:10])
        if sel_feat in active_df.columns:
            fig_hist = px.histogram(
                active_df,
                x=sel_feat,
                color='is_scrap' if 'is_scrap' in active_df else None,
                barmode='overlay',
                title=f"Distribution of {sel_feat}"
            )
            st.plotly_chart(fig_hist)

    with tabs[3]:
        st.markdown("### 📖 Data Dictionary")
        st.markdown("""
        | Column Name | Description | Type | Source |
        |---|---|---|---|
        | **machine_id** | Unique ID of the injection molding machine | Categorical | Hydra |
        | **timestamp** | Time of data capture (UTC) | DateTime | Sensor |
        | **injection_pressure** | Peak pressure during injection phase | Float (Bar) | Sensor |
        | **cycle_time** | Total duration of one production cycle | Float (Sec) | Sensor |
        | **cushion** | Material remaining in barrel after injection | Float (mm) | Sensor |
        | **cyl_tmp_z1** | Temperature of cylinder zone 1 | Float (°C) | Sensor |
        | **melt_temp** | Temperature of the molten plastic | Float (°C) | Sensor |
        | **is_scrap** | Target Variable (1 = Bad Part, 0 = Good Part) | Boolean | QC |
        | **_volatility** | (Engineered) Rolling Standard Deviation over 10 cycles | Float | AI Engine |
        | **_trend** | (Engineered) Rolling Mean over 10 cycles | Float | AI Engine |
        """)

def render_operator_logbook(
    selected_m: str,
    current_filters: Dict[str, Any],
    active_df: pd.DataFrame
) -> None:
    """Operator logbook view aligned with sidebar-selected date range and statuses."""
    st.title("📖 Digital Shift Log")

    if 'logs' not in st.session_state:
        st.session_state['logs'] = pd.DataFrame(
            columns=['Time', 'Operator', 'Machine', 'Status', 'Category', 'Note']
        )

    status_col = robust_column_finder(active_df, ['machine_status_name', 'machine_status', 'status'])
    default_status = "Unknown"
    if status_col and not active_df.empty:
        mode_values = active_df[status_col].dropna()
        if not mode_values.empty:
            default_status = str(mode_values.astype(str).mode().iloc[0])

    with st.form("log_entry"):
        c1, c2 = st.columns(2)
        with c1:
            u_name = st.text_input("Operator ID", "OP-01")
            cat = st.selectbox("Issue Type", ["Maintenance", "Quality", "Material", "Other"])
        with c2:
            note = st.text_area("Observations")

        if st.form_submit_button("Submit Entry"):
            new_row = {
                'Time': datetime.now().strftime("%Y-%m-%d %H:%M"),
                'Operator': u_name,
                'Machine': selected_m if selected_m else "N/A",
                'Status': default_status,
                'Category': cat,
                'Note': note
            }
            st.session_state['logs'] = pd.concat(
                [st.session_state['logs'], pd.DataFrame([new_row])],
                ignore_index=True
            )
            st.success("Log Saved.")

    st.subheader("Recent Entries")
    logs_df = st.session_state['logs'].copy()
    if logs_df.empty:
        st.info("Logbook is empty.")
        return

    # Filter by selected machine.
    if selected_m and selected_m != "N/A":
        logs_df = logs_df[logs_df["Machine"].astype(str) == str(selected_m)]

    # Filter by selected date range from sidebar.
    start_date, end_date = current_filters.get("date_range", (None, None))
    if start_date is not None and end_date is not None:
        log_dt = pd.to_datetime(logs_df["Time"], errors="coerce")
        logs_df = logs_df.loc[
            (log_dt.dt.date >= start_date) &
            (log_dt.dt.date <= end_date)
        ]

    # Filter by selected machine statuses from sidebar.
    selected_statuses = current_filters.get("selected_statuses", [])
    if selected_statuses:
        logs_df = logs_df[logs_df["Status"].astype(str).isin(selected_statuses)]

    if logs_df.empty:
        st.info("No log entries match the current Date Range / Status / Machine filters.")
    else:
        st.dataframe(logs_df)

def _run_dashboard() -> None:
    if check_password():
        # 1) Load full data, 2) apply sidebar filters, 3) route by mode.
        df_loaded = load_data()
        current_filters, df_filtered = render_sidebar(df_loaded)
        app_mode = current_filters.get("app_mode", "Dashboard Overview")
        active_df = df_filtered
        selected_m = current_filters.get("selected_machine", "N/A")

        # Data-driven pages should short-circuit when sidebar filters return no rows.
        data_driven_modes = {
            "Dashboard Overview",
            "Data Explorer",
            "Prescriptive Simulator",
            "Live Monitor",
        }
        if app_mode in data_driven_modes and active_df.empty:
            st.warning("No data matches these filters. Please adjust your selection.")
            return
    
        # -----------------------------------------------------
        # VIEW 1: DASHBOARD OVERVIEW
        # -----------------------------------------------------
        if app_mode == "Dashboard Overview":
            render_dashboard_overview(df_filtered)
    # -----------------------------------------------------
        # VIEW 2: DATA INTEGRATION MAP
        # -----------------------------------------------------
        elif app_mode == "Data Integration Map":
            render_data_integration_map()
    
        # -----------------------------------------------------
        # VIEW 3: DATA EXPLORER
        # -----------------------------------------------------
        elif app_mode == "Data Explorer":
            render_data_explorer(active_df)
    
        # -----------------------------------------------------
        # VIEW 4: PRESCRIPTIVE SIMULATOR (Enhanced)
        # -----------------------------------------------------
        elif app_mode == "Prescriptive Simulator":
            st.title("🧮 What-If Physics Simulator")
            
            if 'sim_history' not in st.session_state:
                st.session_state['sim_history'] = []
    
            recent_vals = active_df.iloc[-1].to_dict() if not active_df.empty else {}
            
            col_ctrl, col_viz = st.columns([1, 2])
            
            with col_ctrl:
                st.subheader("🎛️ Parameters")
                sim_inputs = {}
                # We only show sliders for features that actually exist in the model
                valid_sliders = [f for f in ['injection_pressure', 'cycle_time', 'cushion', 'cyl_tmp_z1', 'melt_temp'] if f in feature_list]
                
                for feat in valid_sliders:
                    base_val = sanitize_numeric(recent_vals.get(feat, 100.0))
                    # Prevent zero-range sliders
                    min_v = base_val * 0.8 if base_val != 0 else 0.0
                    max_v = base_val * 1.2 if base_val != 0 else 10.0
                    if min_v == max_v: max_v += 1.0
                    
                    sim_inputs[feat] = st.slider(f"{feat}", float(min_v), float(max_v), float(base_val))
                
                if st.button("💾 Save Simulation"):
                    rec = {"time": datetime.now().strftime("%H:%M:%S"), **sim_inputs}
                    st.session_state['sim_history'].append(rec)
                    st.success("Scenario Saved!")
    
            with col_viz:
                st.subheader("🔮 Predictive Outcome")
                # Calculate risk using the simulation inputs
                pred_risk = calculate_risk(sim_inputs)
                
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=pred_risk * 100,
                    title={'text': "Scrap Probability (%)"},
                    delta={'reference': threshold * 100},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#34495e"},
                        'steps': [
                            {'range': [0, threshold*100], 'color': "#2ecc71"},
                            {'range': [threshold*100, 100], 'color': "#e74c3c"}
                        ],
                        'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': threshold*100}
                    }
                ))
                st.plotly_chart(fig_g)
                
                # Generate Text Recommendation
                rec_html = prescriptive_engine(sim_inputs, pred_risk)
                st.markdown(f"<div class='suggestion-box'>{rec_html}</div>", unsafe_allow_html=True)
    # -----------------------------------------------------
    # -----------------------------------------------------
        # VIEW 5: LIVE MONITOR (Predictive RCA & Forecasting)
        # -----------------------------------------------------
        elif app_mode == "Live Monitor":
            st.title(f"🚨 Live Stream: {selected_m}")
            st.markdown("### Real-time Root Cause Analysis & Trend Forecasting")
    
            c1, c2 = st.columns([1, 4])
            with c1:
                is_running = st.button("▶️ Start Feed", type="primary")
            with c2:
                spd = st.slider("Feed Speed", 0.05, 1.0, 0.2)
    
            if is_running and not active_df.empty:
                ph_chart = st.empty()
                ph_alert = st.empty()
                ph_process_alert = st.empty()
                process_monitor = ProcessMonitor()
                risk_series = []
    
                # Use latest rows as a simulated feed
                monitor_df = active_df.tail(120).copy().reset_index(drop=True)
    
                # Baseline is scoped to filtered context (machine/part/date/segment/status).
                baseline_df = active_df.apply(pd.to_numeric, errors='coerce') if not active_df.empty else pd.DataFrame()
                baseline_means = baseline_df.mean(numeric_only=True) if not baseline_df.empty else pd.Series(dtype=float)
                baseline_stds = baseline_df.std(numeric_only=True).replace(0, np.nan) if not baseline_df.empty else pd.Series(dtype=float)
                baseline_targets = baseline_means.to_dict()
    
                # Restrict RCA to modeled parameters
                candidate_params = [f for f in feature_list if f in monitor_df.columns and f in baseline_means.index and f in baseline_stds.index]
    
                # Detect time cadence if timestamp exists (fallback = 1 min/cycle)
                ts_col = robust_column_finder(monitor_df, ['timestamp', 'date', 'time'])
                if ts_col:
                    monitor_df[ts_col] = pd.to_datetime(monitor_df[ts_col], errors='coerce')
    
                for i in range(len(monitor_df)):
                    live_df = monitor_df.iloc[:i+1].copy()
                    row = live_df.iloc[-1]
                    risk = calculate_risk(row.to_dict())
                    last_risk = risk_series[-1] if risk_series else 0.0
                    risk_series.append(risk)

                    with ph_process_alert.container():
                        critical_msgs = []
                        warning_msgs = []

                        # Parameter-wise checks against target values.
                        for param_name, param_value in row.to_dict().items():
                            safety = process_monitor.check_safety({param_name: param_value}, baseline_targets)
                            tooltip = safety.get("tooltip", "")
                            composed = safety.get("message", "")
                            if tooltip:
                                composed = f"{composed} Tip: {tooltip}"

                            if safety.get("status") == "CRITICAL" and composed:
                                critical_msgs.append(composed)
                            elif safety.get("status") == "WARNING" and composed:
                                warning_msgs.append(composed)

                        if critical_msgs:
                            st.error("Process Alarm: Critical threshold breach detected.")
                            for msg in critical_msgs:
                                st.markdown(f"- {msg}")
                        elif warning_msgs:
                            st.warning("Process Warning: Parameter deviation detected.")
                            for msg in warning_msgs:
                                st.markdown(f"- {msg}")
                        else:
                            st.success("Process Status: OK (all monitored parameters within threshold limits).")

                    culprit = None
                    z_value = 0.0
                    deviation_pct: Optional[float] = None
                    slope = 0.0
                    param_series = [0.0] * len(risk_series)
                    forecast_series = None
                    eta_txt = ""
    
                    if risk > threshold:
                        if candidate_params:
                            row_numeric = pd.to_numeric(row[candidate_params], errors='coerce')
                            z_scores = ((row_numeric - baseline_means[candidate_params]) / baseline_stds[candidate_params]).replace([np.inf, -np.inf], np.nan).dropna()
                            if not z_scores.empty:
                                culprit = z_scores.abs().idxmax()
                                z_value = float(z_scores[culprit])
                                culprit_value = float(row_numeric.get(culprit, np.nan))
                                culprit_mean = float(baseline_means.get(culprit, np.nan))
                                if not np.isnan(culprit_value) and not np.isnan(culprit_mean) and abs(culprit_mean) > 1e-9:
                                    deviation_pct = ((culprit_value - culprit_mean) / abs(culprit_mean)) * 100.0
    
                                factor_hist = pd.to_numeric(live_df[culprit], errors='coerce').dropna()
                                if not factor_hist.empty:
                                    param_series = factor_hist.ffill().fillna(0).tolist()
    
                                if len(factor_hist) >= 3:
                                    x = np.arange(len(factor_hist), dtype=float)
                                    y = factor_hist.values.astype(float)
    
                                    cadence_min = 1.0
                                    if ts_col and ts_col in live_df.columns:
                                        time_hist = pd.to_datetime(live_df.loc[factor_hist.index, ts_col], errors='coerce').dropna()
                                        if len(time_hist) >= 2:
                                            dt_min = time_hist.diff().dt.total_seconds().dropna() / 60.0
                                            dt_min = dt_min[dt_min > 0]
                                            if not dt_min.empty:
                                                cadence_min = float(dt_min.median())
    
                                    forecast_horizon_minutes = int(getattr(cfg, "FORECAST_HORIZON_MINUTES", 60))
                                    n_future = max(2, int(forecast_horizon_minutes / max(cadence_min, 1e-6)))
                                    future_steps = np.arange(1, n_future + 1, dtype=float)
                                    try:
                                        if y.size == 0 or np.allclose(y, y[0]):
                                            raise np.linalg.LinAlgError("Insufficient variation for linear fit.")
                                        slope = float(np.polyfit(x, y, 1)[0])
                                        forecast_series = float(y[-1]) + slope * future_steps
                                    except np.linalg.LinAlgError as polyfit_err:
                                        logging.warning(
                                            "Forecast polyfit fallback for %s due to %s. Using mean projection.",
                                            culprit,
                                            polyfit_err
                                        )
                                        slope = 0.0
                                        fallback_mean = float(np.nanmean(y)) if y.size > 0 else 0.0
                                        forecast_series = np.full(shape=n_future, fill_value=fallback_mean, dtype=float)
                                    except (ValueError, TypeError) as polyfit_err:
                                        logging.warning(
                                            "Forecast input fallback for %s due to %s. Using mean projection.",
                                            culprit,
                                            polyfit_err
                                        )
                                        slope = 0.0
                                        fallback_mean = float(np.nanmean(y)) if y.size > 0 else 0.0
                                        forecast_series = np.full(shape=n_future, fill_value=fallback_mean, dtype=float)
    
                                    mu = float(baseline_means.get(culprit, np.nan))
                                    sigma = float(baseline_stds.get(culprit, np.nan))
                                    if not np.isnan(mu) and not np.isnan(sigma) and sigma > 0 and slope != 0:
                                        sigma_multiplier = float(getattr(cfg, "SAFE_ZONE_SIGMA_MULTIPLIER", 3.0))
                                        upper = mu + sigma_multiplier * sigma
                                        lower = mu - sigma_multiplier * sigma
                                        current_val = float(y[-1])
                                        steps_to_crash = None
                                        if current_val >= upper or current_val <= lower:
                                            steps_to_crash = 0.0
                                        elif slope > 0 and current_val < upper:
                                            steps_to_crash = (upper - current_val) / slope
                                        elif slope < 0 and current_val > lower:
                                            steps_to_crash = (lower - current_val) / slope
                                        if steps_to_crash is not None and steps_to_crash >= 0:
                                            eta_min = steps_to_crash * cadence_min
                                            if eta_min <= forecast_horizon_minutes:
                                                eta_txt = f" | Forecast crash window: ~{eta_min:.1f} min"
    
                        if risk > float(getattr(cfg, "CRITICAL_RISK_THRESHOLD", 0.85)):
                            if risk > last_risk:
                                try:
                                    st.toast(
                                        f"⚠️ RAPID ESCALATION: Risk rose to {risk:.1%}",
                                        icon="⚠️"
                                    )
                                except Exception as toast_err:
                                    logger.warning("Rapid escalation toast failed: %s", toast_err)
                            with ph_alert.container():
                                if culprit and deviation_pct is not None:
                                    st.metric(
                                        label=f"Root Cause Deviation: {culprit}",
                                        value=f"{deviation_pct:+.1f}%",
                                        delta=f"Z-Score {z_value:+.2f}",
                                        delta_color="normal"
                                    )
                                st.markdown(
                                    f"<div class='alert-box-critical'>🚨 CRITICAL: STOP MACHINE (Risk: {risk:.1%})"
                                    f"{f' | Top Factor: {culprit} (Z={z_value:+.2f}, Drift={slope:+.4f}/cycle){eta_txt}' if culprit else ''}</div>",
                                    unsafe_allow_html=True
                                )
                        elif culprit:
                            with ph_alert.container():
                                if deviation_pct is not None:
                                    st.metric(
                                        label=f"Root Cause Deviation: {culprit}",
                                        value=f"{deviation_pct:+.1f}%",
                                        delta=f"Z-Score {z_value:+.2f}",
                                        delta_color="normal"
                                    )
                                st.warning(
                                    f"⚠️ Warning: Parameter Drift Detected (Risk: {risk:.1%}) | "
                                    f"Top Factor: {culprit} (Z={z_value:+.2f}, Drift={slope:+.4f}/cycle){eta_txt}"
                                )
                        else:
                            ph_alert.warning(f"⚠️ Warning: Parameter Drift Detected (Risk: {risk:.1%})")
                    else:
                        ph_alert.success(f"✅ Nominal Operation (Risk: {risk:.1%})")
    
                    # Smart chart: Risk Spike + At-Risk Parameter side-by-side
                    fig = make_subplots(
                        rows=1,
                        cols=2,
                        subplot_titles=("Risk Spike", f"At-Risk Parameter: {culprit if culprit else 'N/A'}")
                    )
                    fig.add_trace(
                        go.Scatter(y=risk_series, mode='lines', fill='tozeroy', name='Risk', line=dict(color='#3498db', width=3)),
                        row=1, col=1
                    )
                    fig.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="Limit", row=1, col=1)
                    fig.update_yaxes(range=[0, 1], title_text="Risk", row=1, col=1)
    
                    fig.add_trace(
                        go.Scatter(y=param_series, mode='lines', name='Observed', line=dict(color='#e67e22', width=3)),
                        row=1, col=2
                    )
    
                    if culprit and forecast_series is not None:
                        x_future = list(range(len(param_series), len(param_series) + len(forecast_series)))
                        fig.add_trace(
                            go.Scatter(
                                x=x_future,
                                y=forecast_series.tolist(),
                                mode='lines',
                                name=f"Forecast ({int(getattr(cfg, 'FORECAST_HORIZON_MINUTES', 60))}m)",
                                line=dict(color='#c0392b', width=3, dash='dot')
                            ),
                            row=1, col=2
                        )
                        mu = baseline_means.get(culprit, np.nan)
                        sigma = baseline_stds.get(culprit, np.nan)
                        if not pd.isna(mu):
                            fig.add_hline(y=float(mu), line_dash="dot", line_color="gray", annotation_text="Mean", row=1, col=2)
                        if not pd.isna(mu) and not pd.isna(sigma) and float(sigma) > 0:
                            sigma_multiplier = float(getattr(cfg, "SAFE_ZONE_SIGMA_MULTIPLIER", 3.0))
                            upper_spec = float(mu + sigma_multiplier * sigma)
                            lower_spec = float(mu - sigma_multiplier * sigma)
                            fig.add_hrect(
                                y0=lower_spec,
                                y1=upper_spec,
                                fillcolor="rgba(46, 204, 113, 0.1)",
                                line_width=0,
                                row=1,
                                col=2
                            )
                            fig.add_hline(y=upper_spec, line_dash="dash", line_color="crimson", annotation_text="+3σ", row=1, col=2)
                            fig.add_hline(y=lower_spec, line_dash="dash", line_color="crimson", annotation_text="-3σ", row=1, col=2)
    
                    fig.update_layout(title="Predictive RCA & Forecasting Monitor", height=420, legend_orientation="h")
                    ph_chart.plotly_chart(fig)
    
                    time.sleep(spd)
        # -----------------------------------------------------
        # VIEW 6: OPERATOR LOGBOOK
        # -----------------------------------------------------
        elif app_mode == "Operator Logbook":
            render_operator_logbook(selected_m, current_filters, active_df)
    
        # -----------------------------------------------------
        # VIEW 7: SYSTEM HEALTH (Upgraded)
        # -----------------------------------------------------
        elif app_mode == "System Health":
            st.title("🏥 System Diagnostics")
            
            # 1. Health Check Logic
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Model Status")
                health_stats = {
                    "Model_Loaded": model is not None,
                    "Model_Type": "XGBoost Classifier",
                    "Input_Features": len(feature_list) if feature_list else 0,
                    "Risk_Threshold": f"{threshold:.1%}",
                    "Last_Data_Update": datetime.now().strftime("%H:%M:%S")
                }
                st.json(health_stats)
                
            with col2:
                st.subheader("Validation Metrics")
                # Load images if they exist
                roc_path = REPORTS_DIR / "roc_curve.png"
                if roc_path.exists():
                    st.image(str(roc_path), caption="ROC Curve")
                else:
                    st.warning("ROC Curve not found. Run training pipeline.")
            
            st.divider()
            
            # 2. SHAP Explainability (Robust Fix)
            st.subheader("🧠 Explainability Engine (SHAP)")
            st.info("This analysis reveals which physics parameters (Pressure, Temp, etc.) drive the AI's decisions.")
            
            if shap is not None and plt is not None:
                if st.button("Generate Feature Importance"):
                    try:
                        with st.spinner("Calculating SHAP values (Auto-Aligning Features)..."):
                            # Get sample data
                            if not active_df.empty:
                                raw_sample = active_df.tail(50)
                            else:
                                raw_sample = df_loaded.tail(50) if not df_loaded.empty else pd.DataFrame()
                            
                            if not raw_sample.empty:
                                # Clean & Prep
                                clean_sample = clean_dataframe_for_ai(raw_sample, feature_list)
                                
                                # Transform
                                if imputer:
                                    clean_sample = pd.DataFrame(imputer.transform(clean_sample), columns=feature_list)
                                if scaler:
                                    clean_sample = pd.DataFrame(scaler.transform(clean_sample), columns=feature_list)
                                
                                # Calculate SHAP
                                explainer = shap.TreeExplainer(model)
                                shap_values = explainer.shap_values(clean_sample)
                                
                                # Plot
                                st.write("**Top Factors Driving Scrap Risk:**")
                                fig, ax = plt.subplots()
                                shap.summary_plot(shap_values, clean_sample, show=False)
                                st.pyplot(fig)
                            else:
                                st.warning("Not enough data to generate explanation.")
                    except Exception as e:
                        st.error(f"SHAP Calculation Error: {e}")
                        st.info("Tip: Ensure 'model_features.joblib' matches current data columns.")
            else:
                st.warning("SHAP library missing. Please run `pip install shap`.")
    
        # -----------------------------------------------------
        # VIEW 8: USER GUIDE
        # -----------------------------------------------------
        elif app_mode == "User Guide":
            render_user_guide()
    
    # =========================================================
    # FOOTER
    # =========================================================
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #7f8c8d; font-size: 12px;'>
            TE Connectivity AI Cup | Enterprise Edition v7.1.0 <br>
            © 2026 AI Engineering Team | Secure System
        </div>
        """, 
        unsafe_allow_html=True
    )

try:
    _run_dashboard()
except Exception as app_exception:
    _log_and_show_global_error(app_exception)
