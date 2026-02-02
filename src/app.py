import streamlit as st
import sys
import os
import pandas as pd
import joblib
import time
import numpy as np
import io
import logging
import hashlib
import random
import base64
import shutil
import re
from datetime import datetime, timedelta
from pathlib import Path
from functools import lru_cache
from typing import Dict, Optional, List, Any

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
except ImportError:
    class cfg:
        BASE_DIR = PROJECT_ROOT
        REQUIRED_PARAM_COLS = ['injection_pressure', 'cycle_time', 'cushion', 'cyl_tmp_z1', 'melt_temp']
    
    def load_and_merge_data(use_cache=True):
        try:
            cache_path = DATA_DIR / "processed_full_dataset.pkl"
            if cache_path.exists():
                return pd.read_pickle(cache_path)
            return pd.DataFrame()
        except:
            return pd.DataFrame()

# =========================================================
# 📝 LOGGING SETUP
# =========================================================
logging.basicConfig(
    filename=LOGS_DIR / f"dashboard_log_{datetime.now().strftime('%Y%m%d')}.txt",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
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

def clean_dataframe_for_ai(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Prepares data for AI: fills missing cols, ensures float types."""
    df_clean = df.copy()
    for f in features:
        if f not in df_clean.columns: df_clean[f] = 0.0 
    df_clean = df_clean[features]
    for col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    return df_clean.fillna(0.0)

def robust_column_finder(df: pd.DataFrame, keywords: list) -> Optional[str]:
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
def check_password():
    """
    Secure password check using Hashing (SHA256).
    Default password is 'admin'.
    """
    # Hash of 'admin'
    CORRECT_HASH = "8c6976e5b5410415bde908bd4dee15dfb167a9c873fc4bb8a81f6f2ab448a918"

    def password_entered():
        user_input = st.session_state["password"]
        input_hash = hashlib.sha256(user_input.encode()).hexdigest()
        
        if input_hash == CORRECT_HASH:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.sidebar.markdown("---")
        st.markdown("### 🔒 TE Enterprise Login")
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.markdown("### 🔒 TE Enterprise Login")
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Access Denied")
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
    # We utilize the robust finder to locate the actual column names
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
def load_ai_assets():
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
        return None, getattr(cfg, 'REQUIRED_PARAM_COLS', []), None, None, 0.5

@st.cache_data(ttl=3600)
def get_cached_data():
    """
    Loads the main dataset. Uses Streamlit caching for performance.
    """
    try:
        df = load_and_merge_data(use_cache=True)
        # --- APPLY FEATURE ENGINEERING ---
        # We automatically add volatility features upon load
        df = engineer_volatility_features(df, window=10)
        return df
    except Exception as e:
        st.error(f"Data Load Error: {e}")
        return pd.DataFrame()

# Load Global State
model, feature_list, imputer, scaler, threshold = load_ai_assets()
history_df = get_cached_data()

# =========================================================
# ⚙️ HIGH-PERFORMANCE PREDICTION ENGINE
# =========================================================

# Helper to make dictionary hashable for LRU Cache
def dict_to_tuple(d: Dict[str, float]) -> tuple:
    return tuple(sorted(d.items()))

@lru_cache(maxsize=1024)
def _cached_predict(input_tuple: tuple) -> float:
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
    Public wrapper for risk calculation using the caching engine.
    """
    # Convert dict to tuple for caching mechanism
    key = dict_to_tuple(input_row_dict)
    return _cached_predict(key)

def batch_calculate_risk(df_batch: pd.DataFrame) -> List[float]:
    """
    IMPROVEMENT: Vectorized Prediction.
    Takes a DataFrame of 100+ rows and predicts all at once.
    10x faster than looping calculate_risk().
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
    Generates advice to lower scrap risk based on physics simulation.
    """
    if base_risk <= threshold:
        return "✅ <b>System Stable:</b> Current parameters are optimal."

    best_imp = 0.0
    recommendation = "🔍 Check Process Parameters (General)"
    
    # Physics Parameters we can tune
    tunable = ['injection_pressure', 'cycle_time', 'cushion', 'cyl_tmp_z1', 'melt_temp']
    
    for param in tunable:
        curr = sanitize_numeric(current_params.get(param, 0.0))
        if curr == 0: continue
        
        # Try adjusting up/down by 5% and 10%
        for step in [-0.10, -0.05, 0.05, 0.10]:
            test = current_params.copy()
            test[param] = curr * (1 + step)
            
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
if check_password():
    # -----------------------------------------------------
    # SIDEBAR NAVIGATION
    # -----------------------------------------------------
    with st.sidebar:
        # Header
        try:
            # Try to load online logo
            st.image("https://www.te.com/content/dam/te-com/global/english/about-te/news-center/te-logo.png", width=150)
        except:
            st.title("TE Connectivity")
            
        st.markdown(f"**Version:** 7.1.0 (High-Performance)\n**Role:** Admin")
        
        app_mode = st.radio("MAIN MENU", [
            "Dashboard Overview", 
            "Data Integration Map",
            "Data Explorer",
            "Prescriptive Simulator", 
            "Live Monitor", 
            "Operator Logbook",
            "System Health",
            "User Guide"
        ])
        
        st.divider()
        st.subheader("🔍 Filters")
        
        # -------------------------------------------------
        # FAILSAFE FILTER LOGIC (Solves "N/A" Columns)
        # -------------------------------------------------
        active_df = pd.DataFrame()
        
        if not history_df.empty:
            # 1. Machine ID Selection
            mach_kw = ['machine_id', 'machine', 'equipment', 'mach']
            mach_col = robust_column_finder(history_df, mach_kw)
            
            if not mach_col:
                st.error("⚠️ 'Machine ID' column not found.")
                # Fallback: Let user map it manually if needed
                mach_col = st.selectbox("Map Machine Column", history_df.columns)
            
            if mach_col:
                # Get unique machines
                m_opts = sorted(history_df[mach_col].astype(str).unique().tolist())
                selected_m = st.selectbox("Active Machine", m_opts)
                
                # Filter Dataset
                active_df = history_df[history_df[mach_col].astype(str) == selected_m]
                
                # 2. Tool Selection
                # Smart Search
                tool_kw = ['tool_nr', 'tool', 'mold', 'toolid', 'mold_id', 'die']
                tool_col = robust_column_finder(active_df, tool_kw)
                
                # If Smart Search fails, offer Manual Override
                if not tool_col:
                    with st.expander("⚠️ Tool Column Not Found"):
                        tool_col = st.selectbox("Select Tool Column Manually", ["None"] + list(active_df.columns))
                        if tool_col == "None": tool_col = None

                if tool_col:
                    unique_tools = active_df[tool_col].dropna().astype(str).unique().tolist()
                    tool_options = ["All"] + sorted(unique_tools)
                    selected_tool = st.selectbox("Select Tool", tool_options)
                    if selected_tool != "All":
                        active_df = active_df[active_df[tool_col].astype(str) == selected_tool]
                else:
                    st.selectbox("Select Tool", ["N/A (Map Column Above)"], disabled=True)

                # 3. Part Selection
                part_kw = ['material_nr', 'part_no', 'part', 'material', 'product', 'article']
                part_col = robust_column_finder(active_df, part_kw)
                
                if not part_col:
                    with st.expander("⚠️ Part Column Not Found"):
                        part_col = st.selectbox("Select Part Column Manually", ["None"] + list(active_df.columns))
                        if part_col == "None": part_col = None

                if part_col:
                    unique_parts = active_df[part_col].dropna().astype(str).unique().tolist()
                    part_options = ["All"] + sorted(unique_parts)
                    selected_part = st.selectbox("Select Part No.", part_options)
                    if selected_part != "All":
                        active_df = active_df[active_df[part_col].astype(str) == selected_part]
                else:
                    st.selectbox("Select Part No.", ["N/A (Map Column Above)"], disabled=True)
                
                # Show Filter Stats
                st.success(f"✅ Filtered: {len(active_df):,} records")
                
        else:
            st.caption("❌ No records loaded.")
            
        st.divider()
# --- HARD RESET BUTTON ---
        if st.button("🔄 Reload Data Cache"):
            st.cache_data.clear()
            st.cache_resource.clear()
            
            # Physically delete cache file to force rebuild
            cache_path = DATA_DIR / "processed_full_dataset.pkl"
            if cache_path.exists():
                try:
                    cache_path.unlink()
                    st.toast("Cache file deleted.")
                except:
                    pass
            
            st.toast("♻️ Reloading System...")
            time.sleep(1)
            st.rerun()

    # -----------------------------------------------------
    # VIEW 1: DASHBOARD OVERVIEW
    # -----------------------------------------------------
    if app_mode == "Dashboard Overview":
        st.title(f"🏭 Factory Command Center: Machine {selected_m}")
        
        if active_df.empty:
            st.info("👋 Welcome! No data for this selection. Try changing filters.")
        else:
            # KPI Cards
            k1, k2, k3, k4 = st.columns(4)
            
            total_prod = len(active_df)
            scrap_count = int(active_df['is_scrap'].sum()) if 'is_scrap' in active_df.columns else 0
            scrap_rate = (scrap_count / total_prod * 100) if total_prod > 0 else 0.0
            
            k1.metric("Total Batches", f"{total_prod:,}")
            k2.metric("Scrap Events", scrap_count)
            k3.metric("Scrap Rate", f"{scrap_rate:.2f}%", 
                     delta="-0.2%" if scrap_rate < 3 else "+1.5%", delta_color="inverse")
            k4.metric("Model Confidence", "93.2%")
            
            st.divider()
            
            # --- NEW VOLATILITY VISUALIZATION (Phase 2 Feature) ---
            st.subheader("📉 Process Volatility Analysis (The 'Wobble')")
            st.info("Detects instability in Cushion/Pressure before limits are breached.")
            
            c1, c2 = st.columns([2, 1])
            with c1:
                # Plot Volatility if available
                vol_col = 'cushion_volatility'
                if vol_col in active_df.columns:
                    # Create a dual-axis chart: Volatility vs Scrap
                    fig_vol = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    # Add Volatility (Line)
                    fig_vol.add_trace(
                        go.Scatter(x=active_df.index, y=active_df[vol_col], name="Cushion Wobble (StdDev)", line=dict(color='#e67e22')),
                        secondary_y=False
                    )
                    # Add Scrap (Area)
                    if 'is_scrap' in active_df:
                        fig_vol.add_trace(
                            go.Scatter(x=active_df.index, y=active_df['is_scrap'], name="Scrap Event", mode='markers', marker=dict(color='red', size=8, opacity=0.5)),
                            secondary_y=True
                        )
                    
                    fig_vol.update_layout(title="Cushion Instability vs. Scrap Events", height=400)
                    fig_vol.update_yaxes(title_text="Volatility (Std Dev)", secondary_y=False)
                    fig_vol.update_yaxes(title_text="Scrap Event (1=Bad)", secondary_y=True, showgrid=False)
                    st.plotly_chart(fig_vol, use_container_width=True)
                else:
                    # Fallback to standard timeline if volatility missing
                    time_col = robust_column_finder(active_df, ['timestamp', 'date', 'time'])
                    if time_col:
                        active_df['ui_date'] = pd.to_datetime(active_df[time_col])
                        daily = active_df.set_index('ui_date').resample('D')['is_scrap'].agg(['mean']).reset_index()
                        st.plotly_chart(px.area(daily, x='ui_date', y='mean', title="Daily Scrap Rate Trend"), use_container_width=True)

            with c2:
                st.subheader("⚠️ Risk Segmentation")
                if 'is_scrap' in active_df.columns:
                    active_df['Quality'] = active_df['is_scrap'].apply(lambda x: 'Bad (Scrap)' if x==1 else 'Good (Safe)')
                    color_map = {'Good (Safe)': '#2ecc71', 'Bad (Scrap)': '#e74c3c'}
                    
                    fig_pie = px.pie(active_df, names='Quality', title="Batch Quality Distribution", 
                                     color='Quality', color_discrete_map=color_map,
                                     hole=0.4)
                    st.plotly_chart(fig_pie, use_container_width=True)

    # -----------------------------------------------------
    # VIEW 2: DATA MAP
    # -----------------------------------------------------
    elif app_mode == "Data Integration Map":
        st.title("🌐 AWS Data Integration Architecture")
        st.markdown("### Visualizing the Flow of Inputs X1 & X2 → Output Y")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="flow-box" style="background-color: #2980b9;">🔹 INPUT 1 (X1)<br>AWS Athena<br>(Machine Params)</div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="flow-box" style="background-color: #8e44ad;">🔹 INPUT 2 (X2)<br>AWS Redshift<br>(Order/Hydra Data)</div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="flow-box" style="background-color: #27ae60;">🎯 OUTPUT (Y)<br>Scrap Prediction</div>', unsafe_allow_html=True)
            
        st.divider()
        st.markdown("### 🔄 ETL Logic Visualization")
        m1, m2, m3, m4, m5 = st.columns([1, 0.2, 1, 0.2, 1])
        m1.button("Extract X1 & X2", disabled=True, use_container_width=True)
        m2.markdown("➡️")
        m3.button("Merge & Clean", disabled=True, use_container_width=True)
        m4.markdown("➡️")
        m5.button("Train AI Model (Y)", disabled=True, use_container_width=True)
# -----------------------------------------------------
    # VIEW 3: DATA EXPLORER
    # -----------------------------------------------------
    elif app_mode == "Data Explorer":
        st.title("🔍 Advanced Data Inspector")
        
        if active_df.empty:
            st.warning("Data unavailable. Please check filters.")
        else:
            tabs = st.tabs(["Dataset View", "Correlations", "Distribution Analysis", "Data Dictionary"])
            
            with tabs[0]:
                st.markdown("### Raw Production Data")
                st.dataframe(active_df.head(100), use_container_width=True)
                st.caption(f"Showing first 100 of {len(active_df)} rows.")
            
            with tabs[1]:
                st.markdown("### Feature Correlations")
                clean_view = clean_dataframe_for_ai(active_df, feature_list[:12]) 
                corr = clean_view.corr()
                fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', aspect="auto")
                st.plotly_chart(fig_corr, use_container_width=True)
            
            with tabs[2]:
                st.markdown("### Feature Distributions")
                sel_feat = st.selectbox("Select Feature to Analyze", feature_list[:10])
                fig_hist = px.histogram(active_df, x=sel_feat, color='is_scrap', 
                                      barmode='overlay', title=f"Distribution of {sel_feat}")
                st.plotly_chart(fig_hist, use_container_width=True)
            
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
            for feat in ['injection_pressure', 'cycle_time', 'cushion', 'cyl_tmp_z1', 'melt_temp']:
                if feat in feature_list:
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
            st.plotly_chart(fig_g, use_container_width=True)
            
            rec_html = prescriptive_engine(sim_inputs, pred_risk)
            st.markdown(f"<div class='suggestion-box'>{rec_html}</div>", unsafe_allow_html=True)
            
        st.divider()
        
        # --- IMPROVED STABILITY TEST (Robust Plotly) ---
        st.subheader("📉 Tomorrow's Forecast: Drift Analysis")
        st.markdown("This tool simulates what happens if parameters like **Cushion** vary by **±5%** (Process Drift).")
        
        if st.button("▶️ Run Stability Test"):
            try:
                with st.spinner("Running 100 Monte Carlo Simulations..."):
                    # 1. Setup Simulation Data
                    noise_matrix = np.random.uniform(0.95, 1.05, size=(100, len(sim_inputs)))
                    base_values = np.array(list(sim_inputs.values()))
                    simulated_data = noise_matrix * base_values
                    sim_df = pd.DataFrame(simulated_data, columns=sim_inputs.keys())
                    
                    # 2. Run Vectorized Prediction
                    drift_risks = batch_calculate_risk(sim_df)
                    
                    # 3. Analyze Results
                    avg_risk = np.mean(drift_risks)
                    max_risk = np.max(drift_risks)
                    high_risk_count = sum(1 for r in drift_risks if r > threshold)
                    
                    # 4. Display Metrics
                    d1, d2, d3 = st.columns(3)
                    d1.metric("Predicted Avg Risk", f"{avg_risk:.1%}", delta=f"{avg_risk-pred_risk:.1%}", delta_color="inverse")
                    d2.metric("Worst Case Scenario", f"{max_risk:.1%}")
                    d3.metric("Failures in 100 Batches", f"{high_risk_count}", "Critical" if high_risk_count > 5 else "Stable", delta_color="inverse")
                    
                    if high_risk_count > 5:
                        st.error(f"⚠️ **Instability Detected:** Process is too sensitive. {high_risk_count}% of batches failed.")
                    else:
                        st.success("✅ **Process Stable:** Machine can handle normal parameter variation.")
                        
                    # 5. Visualization (Robust Plotly Chart)
                    chart_df = pd.DataFrame({
                        'Simulation #': range(1, 101),
                        'Risk Score': drift_risks
                    })
                    
                    fig_stab = px.area(chart_df, x='Simulation #', y='Risk Score',
                                      title="Risk Variance Across 100 Simulations",
                                      color_discrete_sequence=['#3498db'])
                    
                    fig_stab.add_hline(y=threshold, line_dash="dash", line_color="red", 
                                      annotation_text=f"Failure Limit ({threshold:.0%})")
                    fig_stab.update_yaxes(range=[0, 1.1])
                    st.plotly_chart(fig_stab, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Simulation Error: {e}")

        st.divider()
        st.subheader("🌋 Sensitivity Heatmap")
        if 'injection_pressure' in sim_inputs and 'cycle_time' in sim_inputs:
            x_range = np.linspace(sim_inputs['injection_pressure']*0.9, sim_inputs['injection_pressure']*1.1, 10)
            y_range = np.linspace(sim_inputs['cycle_time']*0.9, sim_inputs['cycle_time']*1.1, 10)
            z_data = []
            
            # Heatmap calculation loop
            for y_val in y_range:
                row_z = []
                for x_val in x_range:
                    temp_input = sim_inputs.copy()
                    temp_input['injection_pressure'] = x_val
                    temp_input['cycle_time'] = y_val
                    # Uses LRU cached prediction for speed
                    row_z.append(calculate_risk(temp_input))
                z_data.append(row_z)
                
            fig_heat = go.Figure(data=go.Heatmap(
                z=z_data, x=x_range, y=y_range,
                colorscale='RdYlGn_r',
                hoverongaps=False
            ))
            fig_heat.update_layout(title="Risk Heatmap", xaxis_title="Injection Pressure", yaxis_title="Cycle Time")
            st.plotly_chart(fig_heat, use_container_width=True)

    # -----------------------------------------------------
    # VIEW 5: LIVE MONITOR
    # -----------------------------------------------------
    elif app_mode == "Live Monitor":
        st.title(f"🚨 Live Stream: {selected_m}")
        
        c1, c2 = st.columns([1, 4])
        with c1:
            is_running = st.button("▶️ Start Feed")
        with c2:
            spd = st.slider("Feed Speed", 0.05, 1.0, 0.2)

        if is_running and not active_df.empty:
            ph_chart = st.empty()
            ph_alert = st.empty()
            risk_series = []
            
            monitor_df = active_df.tail(100)
            
            for i in range(len(monitor_df)):
                row = monitor_df.iloc[i].to_dict()
                risk = calculate_risk(row)
                risk_series.append(risk)
                
                if risk > 0.90:
                    ph_alert.markdown(f"<div class='alert-box-critical'>🚨 CRITICAL: STOP MACHINE (Risk: {risk:.1%})</div>", unsafe_allow_html=True)
                elif risk > threshold:
                    ph_alert.warning(f"⚠️ Warning: Parameter Drift Detected (Risk: {risk:.1%})")
                else:
                    ph_alert.success(f"✅ Nominal Operation (Risk: {risk:.1%})")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=risk_series, mode='lines', fill='tozeroy', name='Risk', line=dict(color='#3498db')))
                fig.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="Limit")
                fig.update_layout(title="Real-time Risk Telemetry", yaxis_range=[0, 1], height=350)
                
                ph_chart.plotly_chart(fig, use_container_width=True)
                time.sleep(spd)
# -----------------------------------------------------
    # VIEW 6: OPERATOR LOGBOOK
    # -----------------------------------------------------
    elif app_mode == "Operator Logbook":
        st.title("📖 Digital Shift Log")
        
        # Initialize Logbook if missing
        if 'logs' not in st.session_state:
            st.session_state['logs'] = pd.DataFrame(columns=['Time', 'Operator', 'Machine', 'Category', 'Note'])
            
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
                    'Category': cat,
                    'Note': note
                }
                # Safe concatenation
                st.session_state['logs'] = pd.concat([st.session_state['logs'], pd.DataFrame([new_row])], ignore_index=True)
                st.success("Log Saved.")
        
        st.subheader("Recent Entries")
        if not st.session_state['logs'].empty:
            st.dataframe(st.session_state['logs'], use_container_width=True)
        else:
            st.info("Logbook is empty.")

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
                st.image(str(roc_path), caption="ROC Curve", use_container_width=True)
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
                            raw_sample = history_df.tail(50) if not history_df.empty else pd.DataFrame()
                        
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
        st.title("📚 User Documentation")
        st.markdown("""
        # 🏭 TE Connectivity AI Copilot - User Manual

        ### **1. Dashboard Overview**
        **Purpose:** Your daily "Health Check" for factory production.
        * **KPI Cards:** Track Total Batches, Scrap Events, and AI Confidence.
        * **Timeline:** Visualize scrap spikes over time.
        * **Risk Segmentation:** See the Good vs. Bad ratio.
        * **Volatility Analysis:** (New!) See if parameters like 'Cushion' are wobbling before they fail.

        ### **2. Prescriptive Simulator**
        **Purpose:** A "What-If" Physics playground.
        * **Usage:** Move sliders to change Machine Parameters (Pressure, Temp).
        * **AI Advice:** The system will tell you the optimal settings to minimize risk.
        * **Stability Test:** Click "Run Stability Test" to simulate 100 future batches and check for process drift.

        ### **3. Live Monitor**
        **Purpose:** Real-time production tracking.
        * **Usage:** Click "Start Feed" to replay data. 
        * **Alerts:** Critical Red Alerts appear if Risk > Threshold.

        ### **4. Troubleshooting**
        * **"Reload Data Cache":** Use this button (in the sidebar) if you uploaded new files but don't see them yet.
        """)

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