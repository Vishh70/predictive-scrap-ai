import pandas as pd
import numpy as np
import joblib
import sys
import json
import logging
import time
import random
from pathlib import Path
from datetime import datetime

# =========================================================
# CONFIGURATION & SETUP
# =========================================================
# Setup Logging to capture every prediction event
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [INFERENCE] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("TE_Predictor")

# Import Config (Graceful Fallback)
try:
    import src.config as cfg
except ImportError:
    # If running script directly from src/ folder
    sys.path.append(str(Path.cwd().parent))
    import src.config as cfg

# =========================================================
# REAL-TIME INFERENCE CLASS
# =========================================================
class RealTimePredictor:
    """
    Enterprise Inference Engine for Scrap Prediction.
    
    Features:
    - ⚡ One-Time Asset Loading (Singleton)
    - 🛡️ Safe Preprocessing (Transform Only)
    - 🧩 Auto-Feature Completion (Handling missing sensors)
    - 🚦 Traffic Light Risk Status (Safe/Warning/Critical)
    """

    def __init__(self):
        self.model = None
        self.features = None
        self.imputer = None
        self.scaler = None
        self.threshold = 0.5
        self._assets_loaded = False
        
        # Load immediately on instantiation
        self._load_assets()

    def _load_assets(self):
        """
        Loads trained artifacts from disk. 
        Critical: This happens ONCE at startup, not per prediction.
        """
        if self._assets_loaded:
            return

        try:
            logger.info("⏳ Loading AI Brain (Models & Scalers)...")
            
            # Paths to artifacts
            model_path = cfg.MODELS_DIR / "scrap_model.joblib"
            feat_path = cfg.MODELS_DIR / "model_features.joblib"
            imp_path = cfg.MODELS_DIR / "imputer.joblib"
            scl_path = cfg.MODELS_DIR / "scaler.joblib"
            thr_path = cfg.MODELS_DIR / "threshold.joblib"

            if not model_path.exists():
                raise FileNotFoundError("Model file missing. Train the model first!")

            # Load Artifacts
            self.model = joblib.load(model_path)
            self.features = joblib.load(feat_path)
            self.imputer = joblib.load(imp_path)
            self.scaler = joblib.load(scl_path)
            self.threshold = float(joblib.load(thr_path))
            
            self._assets_loaded = True
            logger.info(f"✅ System Ready. Threshold: {self.threshold:.2%}")
            logger.info(f"   Model expects {len(self.features)} features.")

        except Exception as e:
            logger.error(f"❌ FATAL: Could not load AI assets. {e}")
            sys.exit(1)

    def _enrich_features(self, input_df):
        """
        Ensures input matches the exact schema used during training.
        Auto-fills missing engineered features (lags/deltas) with 0.
        """
        # 1. Check for missing columns
        missing_cols = [f for f in self.features if f not in input_df.columns]
        
        if missing_cols:
            # For real-time single points, we cannot calculate lags/history.
            # We safely impute them as 0 (Neutral).
            for col in missing_cols:
                input_df[col] = 0.0
        
        # 2. Enforce Strict Column Order (Crucial for XGBoost)
        return input_df[self.features]

    def predict(self, sensor_data: dict) -> dict:
        """
        Main API Method.
        Args:
            sensor_data (dict): Raw sensor values (e.g. {'temperature': 200, 'pressure': 50})
        Returns:
            dict: JSON-ready response with Risk Score and Status.
        """
        start_t = time.time()
        
        try:
            # Step 1: Data Structuring
            df = pd.DataFrame([sensor_data])
            
            # Step 2: Feature Alignment & Safety Fill
            X_aligned = self._enrich_features(df)
            
            # Step 3: Preprocessing (Impute -> Scale)
            # IMPORTANT: We use transform(), NEVER fit() during inference.
            X_imputed = self.imputer.transform(X_aligned)
            X_scaled = self.scaler.transform(X_imputed)
            
            # Step 4: Inference
            if hasattr(self.model, "predict_proba"):
                # Get probability of Class 1 (Scrap)
                risk_score = float(self.model.predict_proba(X_scaled)[:, 1][0])
            else:
                # Fallback for models without probability
                risk_score = float(self.model.predict(X_scaled)[0])
            
            # Step 5: Status Classification
            status = "SAFE 🟢"
            action = "Monitor"
            
            if risk_score >= 0.85:
                status = "CRITICAL 🛑"
                action = "STOP MACHINE"
            elif risk_score >= self.threshold:
                status = "HIGH RISK ⚠️"
                action = "Inspect Parameters"
                
            latency_ms = (time.time() - start_t) * 1000
            
            # Construct Response
            result = {
                "timestamp": datetime.now().isoformat(),
                "prediction": {
                    "probability": round(risk_score, 4),
                    "is_scrap": bool(risk_score >= self.threshold),
                    "status": status,
                    "recommended_action": action
                },
                "system_info": {
                    "threshold_used": round(self.threshold, 2),
                    "latency_ms": round(latency_ms, 2),
                    "model_version": "v4.2.0"
                }
            }
            
            return result

        except Exception as e:
            logger.error(f"Prediction Failed: {e}")
            return {"error": str(e), "status": "ERROR"}

# =========================================================
# SIMULATION MODE (For Testing)
# =========================================================
def run_simulation_mode(engine):
    """Generates synthetic machine data to test the pipeline."""
    logger.info("🚀 Starting Real-Time Simulation Stream...")
    print("\n" + "="*85)
    print(f"{'TIMESTAMP':<20} | {'RISK %':<8} | {'STATUS':<15} | {'ACTION':<20} | {'LATENCY':<8}")
    print("="*85)
    
    # Base parameters for a "Good" machine state
    base_params = {
        'injection_pressure': 140, 
        'cycle_time': 22.5, 
        'cushion': 5.5, 
        'cyl_tmp_z1': 230,
        'melt_temp': 240
    }
    
    try:
        while True:
            # 1. Randomly decide if this cycle is "Good" or "Bad"
            is_anomaly = random.random() > 0.80 # 20% chance of anomaly
            
            # 2. Jitter the parameters
            input_data = base_params.copy()
            noise_level = 0.25 if is_anomaly else 0.02 # High noise for anomalies
            
            for k, v in input_data.items():
                input_data[k] = v * random.uniform(1 - noise_level, 1 + noise_level)
            
            # 3. Predict
            res = engine.predict(input_data)
            
            # 4. Print Row
            ts = res['timestamp'][11:19] # Time only
            risk = res['prediction']['probability']
            status = res['prediction']['status']
            action = res['prediction']['recommended_action']
            lat = res['system_info']['latency_ms']
            
            print(f"{ts:<20} | {risk:.1%}   | {status:<15} | {action:<20} | {lat}ms")
            
            # Sleep to simulate cycle time
            time.sleep(1.5)
            
    except KeyboardInterrupt:
        print("\n🛑 Simulation Stopped.")

if __name__ == "__main__":
    # Initialize Engine
    engine = RealTimePredictor()
    
    # Run Demo
    run_simulation_mode(engine)