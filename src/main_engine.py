import pandas as pd
import numpy as np
import joblib
import os
import sys
import shutil
import logging
import json
import time
import psutil # Ensure this is installed: pip install psutil
from pathlib import Path
from datetime import datetime, timedelta

# Machine Learning Imports
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, precision_recall_curve, roc_auc_score

# =========================================================
# CONFIGURATION & SETUP
# =========================================================
# Import Local Modules (Graceful Fallback)
try:
    import src.config as cfg
    from src.data_loading import load_and_merge_data
    from src.train_model import ScrapPredictionTrainer # Import the robust trainer we built
except ImportError:
    # If running directly from src/
    sys.path.append(str(Path.cwd().parent))
    import src.config as cfg
    from src.data_loading import load_and_merge_data
    from src.train_model import ScrapPredictionTrainer

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [ORCHESTRATOR] - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline_execution.log", mode='a')
    ]
)
logger = logging.getLogger("TE_Main_Engine")

# =========================================================
# SYSTEM DIAGNOSTICS CLASS
# =========================================================
class SystemSentinel:
    """Monitors Hardware & Environment Health before AI Execution."""
    
    @staticmethod
    def check_resources():
        logger.info("🏥 Running System Health Check...")
        
        # 1. Disk Space Check
        total, used, free = shutil.disk_usage("/")
        free_gb = free // (2**30)
        if free_gb < 2:
            logger.warning(f"⚠️  Low Disk Space: {free_gb}GB available.")
        else:
            logger.info(f"✅ Disk Space: {free_gb}GB available.")

        # 2. Memory Check
        mem = psutil.virtual_memory()
        if mem.percent > 90:
            logger.error(f"❌ Critical Memory Usage: {mem.percent}%")
            return False
        
        logger.info(f"✅ System Ready. CPU: {psutil.cpu_percent()}% | RAM: {mem.percent}%")
        return True

    @staticmethod
    def check_folder_structure():
        required = [cfg.DATA_DIR, cfg.MODELS_DIR, cfg.REPORTS_DIR, cfg.ARCHIVE_DIR]
        for d in required:
            if not d.exists():
                logger.info(f"🛠️  Creating missing directory: {d}")
                d.mkdir(parents=True, exist_ok=True)

# =========================================================
# CORE ORCHESTRATION ENGINE
# =========================================================
class PredictiveScrapEngine:
    """
    The Brain of the Operation.
    Coordinates Data -> Training -> Inference -> Reporting -> Archival.
    """
    
    def __init__(self):
        self.sentinel = SystemSentinel()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.sentinel.check_folder_structure()
        
        # Load Threshold Configuration
        self.threshold_path = cfg.MODELS_DIR / "threshold.joblib"
        self.threshold = 0.5 # Default
        if self.threshold_path.exists():
            self.threshold = float(joblib.load(self.threshold_path))

    def _is_model_stale(self, days=7):
        """Checks if the model is older than 'days'."""
        model_path = cfg.MODELS_DIR / "scrap_model.joblib"
        if not model_path.exists():
            return True # No model = Stale
        
        mod_time = datetime.fromtimestamp(model_path.stat().st_mtime)
        age = (datetime.now() - mod_time).days
        if age > days:
            logger.warning(f"⚠️  Model is {age} days old. Recommendation: Retrain.")
            return True
        return False

    def ingest_data_layer(self):
        """Step 1: Data Loading Wrapper."""
        logger.info("📡 Step 1: Ingesting Data Stream...")
        if not self.sentinel.check_resources():
            raise SystemError("Insufficient resources to load data.")
            
        df = load_and_merge_data()
        if df.empty:
            raise ValueError("❌ No data loaded. Check input folders.")
        
        logger.info(f"✅ Data Ingested. Shape: {df.shape}")
        return df

    def model_management_layer(self, df, force_retrain=False):
        """Step 2: Training Logic (Delegates to ScrapPredictionTrainer)."""
        logger.info("🧠 Step 2: Model Lifecycle Management...")
        
        should_train = force_retrain or self._is_model_stale()
        
        if should_train:
            logger.info("🚀 Initiating Training Sequence...")
            trainer = ScrapPredictionTrainer()
            trainer.train_pipeline()
            
            # Reload threshold after training
            if self.threshold_path.exists():
                self.threshold = float(joblib.load(self.threshold_path))
        else:
            logger.info("✅ Existing model is fresh. Skipping training.")

    def inference_layer(self, df):
        """Step 3: Batch Prediction on Data."""
        logger.info("🔮 Step 3: Running Batch Inference...")
        
        # Load Assets
        try:
            model = joblib.load(cfg.MODELS_DIR / "scrap_model.joblib")
            features = joblib.load(cfg.MODELS_DIR / "model_features.joblib")
            imputer = joblib.load(cfg.MODELS_DIR / "imputer.joblib")
            scaler = joblib.load(cfg.MODELS_DIR / "scaler.joblib")
        except FileNotFoundError as e:
            logger.error(f"❌ Missing AI Artifacts: {e}")
            return pd.DataFrame()

        # Preprocess
        X = df[features]
        # Handle Missing features in new data if any (Align columns)
        missing_cols = set(features) - set(X.columns)
        for c in missing_cols: X[c] = 0
        X = X[features] # Enforce order

        X_imp = imputer.transform(X)
        X_scl = scaler.transform(X_imp)
        
        # Predict
        probs = model.predict_proba(X_scl)[:, 1]
        
        # Append Results to DF
        result_df = df.copy()
        result_df['AI_Scrap_Prob'] = probs
        result_df['AI_Prediction'] = (probs >= self.threshold).astype(int)
        result_df['Risk_Level'] = pd.cut(probs, 
                                         bins=[-0.1, self.threshold, 0.85, 1.1], 
                                         labels=['Safe', 'High Risk', 'CRITICAL'])
        
        logger.info("✅ Inference Complete.")
        return result_df

    def prescriptive_layer(self, df):
        """Step 4: Generate Actionable Advice based on Physics."""
        logger.info("💡 Step 4: Generating Prescriptive Intelligence...")
        
        def generate_advice(row):
            if row['Risk_Level'] == 'Safe':
                return "✅ No Action Required"
            
            advice = []
            
            # Physics Heuristics (Example Logic)
            # In a real app, these thresholds would be dynamic or learnt
            if row.get('injection_pressure', 0) > 1600:
                advice.append("⬇️ Reduce Inj. Pressure (-5%)")
            elif row.get('injection_pressure', 0) < 800:
                advice.append("⬆️ Increase Inj. Pressure (+5%)")
                
            if row.get('cycle_time', 0) < 15:
                advice.append("⚠️ Cycle Time too fast -> Check Cooling")
                
            if row.get('melt_temp', 0) > 280:
                advice.append("❄️ Reduce Melt Temp (-10°C)")
                
            if not advice:
                advice.append("🔍 Inspect Raw Material Quality")
                
            return " | ".join(advice)

        df['AI_Prescription'] = df.apply(generate_advice, axis=1)
        return df

    def reporting_layer(self, df):
        """Step 5: Exporting Intelligence."""
        logger.info("📊 Step 5: Generating Reports...")
        
        # 1. Full Audit Log
        full_report_path = cfg.REPORTS_DIR / f"Full_Audit_{self.timestamp}.csv"
        df.to_csv(full_report_path, index=False)
        
        # 2. Action List (Only Bad Parts)
        action_df = df[df['AI_Prediction'] == 1].copy()
        action_path = cfg.REPORTS_DIR / f"ACTION_REQUIRED_{self.timestamp}.csv"
        
        # Select operational columns for the floor report
        display_cols = ['timestamp', 'machine_id', 'AI_Scrap_Prob', 'Risk_Level', 'AI_Prescription']
        # Add sensor cols if they exist
        for c in ['injection_pressure', 'cycle_time']:
            if c in df.columns: display_cols.append(c)
            
        final_action_df = action_df[display_cols] if not action_df.empty else pd.DataFrame()
        final_action_df.to_csv(action_path, index=False)
        
        # 3. Summary Statistics (JSON)
        summary = {
            "timestamp": self.timestamp,
            "total_batches": len(df),
            "scrap_predicted": len(action_df),
            "scrap_rate_predicted": f"{(len(action_df)/len(df))*100:.2f}%",
            "critical_alerts": len(df[df['Risk_Level'] == 'CRITICAL'])
        }
        
        with open(cfg.REPORTS_DIR / f"Shift_Summary_{self.timestamp}.json", "w") as f:
            json.dump(summary, f, indent=4)
            
        logger.info(f"✅ Reports Saved:\n   - {action_path}")

    def archival_layer(self):
        """Step 6: Cleanup."""
        logger.info("📦 Step 6: Data Archival (Optional)...")
        # In a real system, you would move processed CSVs to cfg.ARCHIVE_DIR
        # Here we just log it to avoid deleting user files during testing.
        logger.info("ℹ️  Skipping file move for safety. Logic is ready.")

    def run_full_pipeline(self, force_retrain=False):
        """The Master Switch."""
        start_t = time.time()
        logger.info("="*60)
        logger.info("🚀 STARTING TE CONNECTIVITY AI PIPELINE")
        logger.info("="*60)
        
        try:
            # 1. Ingest
            df = self.ingest_data_layer()
            
            # 2. Train (if needed)
            self.model_management_layer(df, force_retrain=force_retrain)
            
            # 3. Infer
            scored_df = self.inference_layer(df)
            if scored_df.empty:
                logger.warning("⚠️ No predictions generated.")
                return
            
            # 4. Prescribe
            final_df = self.prescriptive_layer(scored_df)
            
            # 5. Report
            self.reporting_layer(final_df)
            
            # 6. Archive
            self.archival_layer()
            
            duration = time.time() - start_t
            logger.info("="*60)
            logger.info(f"✅ PIPELINE COMPLETED SUCCESSFULLY in {duration:.2f}s")
            logger.info("="*60)
            
        except Exception as e:
            logger.exception(f"❌ PIPELINE CRASHED: {e}")
            sys.exit(1)

# =========================================================
# ENTRY POINT
# =========================================================
if __name__ == "__main__":
    # Check for CLI arguments
    force = False
    if len(sys.argv) > 1 and sys.argv[1] == "--retrain":
        force = True
        
    engine = PredictiveScrapEngine()
    engine.run_full_pipeline(force_retrain=force)