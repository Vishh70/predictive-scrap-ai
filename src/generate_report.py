import pandas as pd
import numpy as np
import joblib
import sys
import os
import logging
import traceback
from datetime import datetime
from pathlib import Path

# =========================================================
# CONFIGURATION & SETUP
# =========================================================
# Configure Logging to standard output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [REPORTING] - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("TE_Reporter")

# Import Config & Data Loader (Graceful Fallback)
# This block allows the script to find the 'src' folder even if run from different locations
try:
    import src.config as cfg
    from src.data_loading import load_and_merge_data
except ImportError:
    # If standard import fails, try appending parent directory to path
    sys.path.append(str(Path.cwd().parent))
    try:
        import src.config as cfg
        from src.data_loading import load_and_merge_data
    except ImportError:
        # Fallback if config cannot be found
        logger.warning("Could not import src.config. Using default paths.")
        class cfg:
            MODELS_DIR = Path('models')
            REPORTS_DIR = Path('reports')
        
        def load_and_merge_data():
            return pd.DataFrame()

# =========================================================
# PRESCRIPTIVE ANALYTICS ENGINE
# =========================================================
class PrescriptiveReporter:
    """
    Generates actionable business intelligence reports.
    Feature: Simulates physics adjustments to find 'Cures' for high-risk batches.
    """

    def __init__(self):
        # Initialize paths based on config
        self.models_dir = getattr(cfg, 'MODELS_DIR', Path('models'))
        self.reports_dir = getattr(cfg, 'REPORTS_DIR', Path('reports'))
        
        # Ensure report directory exists
        if not self.reports_dir.exists():
            logger.info(f"Creating reports directory at: {self.reports_dir}")
            self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Internal State storage
        self.model = None
        self.features = None
        self.imputer = None
        self.scaler = None
        self.threshold = 0.5

    def load_assets(self):
        """
        Loads the trained AI brain from disk.
        Returns True if successful, False otherwise.
        """
        try:
            logger.info("⏳ Loading AI Assets for Reporting...")
            
            # Define paths to assets
            model_path = self.models_dir / "scrap_model.joblib"
            feat_path = self.models_dir / "model_features.joblib"
            imp_path = self.models_dir / "imputer.joblib"
            scl_path = self.models_dir / "scaler.joblib"
            thr_path = self.models_dir / "threshold.joblib"

            # Verify files exist before loading
            for p in [model_path, feat_path, imp_path, scl_path, thr_path]:
                if not p.exists():
                    logger.error(f"❌ Missing critical asset: {p}")
                    return False

            # Load assets
            self.model = joblib.load(model_path)
            self.features = joblib.load(feat_path)
            self.imputer = joblib.load(imp_path)
            self.scaler = joblib.load(scl_path)
            self.threshold = joblib.load(thr_path)
            
            logger.info(f"✅ Assets Loaded Successfully. Active Risk Threshold: {self.threshold:.1%}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load assets: {e}")
            logger.error("   Tip: Run 'python -m src.train_model' to generate them.")
            return False

    def get_prescriptive_advice(self, row, current_prob):
        """
        [The 'Secret Sauce'] 
        Simulates changing machine parameters to see if risk drops.
        Returns the best physics-based advice string.
        """
        # If risk is low, no action needed
        if current_prob < self.threshold:
            return "✅ Optimal Settings"

        best_advice = "🔍 Inspect Material / Mold"
        best_improvement = 0.0
        
        # Parameters we are allowed to tweak (Operator Controls)
        tunable_params = [
            'injection_pressure', 
            'cycle_time', 
            'cushion', 
            'cyl_tmp_z1', 
            'melt_temp', 
            'back_pressure'
        ]
        
        # Simulation Logic
        for param in tunable_params:
            # Skip if parameter data is missing
            if param not in row or row[param] == 0:
                continue
                
            original_val = row[param]
            
            # Scenario A: Decrease by 5%
            val_down = original_val * 0.95
            
            # Scenario B: Increase by 5%
            val_up = original_val * 1.05
            
            # Test both scenarios
            for val, direction in [(val_down, "Decrease"), (val_up, "Increase")]:
                # Create a synthetic row copy
                sim_row = row.copy()
                sim_row[param] = val
                
                # Re-calculate lag features if they exist (Physics Consistency)
                if f"{param}_lag_1" in sim_row:
                    sim_row[f"{param}_lag_1"] = original_val 
                
                # Transform & Predict
                # Create single-row DataFrame
                sim_df = pd.DataFrame([sim_row], columns=self.features)
                
                # Scale using the loaded scaler
                sim_scaled = self.scaler.transform(sim_df)
                
                # Get new prediction probability
                new_prob = self.model.predict_proba(sim_scaled)[0][1]
                
                # Calculate improvement
                improvement = current_prob - new_prob
                
                # If significant improvement found, save it
                if improvement > 0.05 and improvement > best_improvement:
                    best_improvement = improvement
                    best_advice = f"{direction} {param} by 5% (to {val:.1f})"

        if best_improvement > 0:
            return f"💡 {best_advice} (Est. Risk -{best_improvement:.1%})"
        
        return "⚠️ Complex Issue - Technician Required"

    def generate(self):
        """
        Main execution flow.
        Loads data, predicts risk, generates advice, and saves reports.
        """
        # Load Assets
        if not self.load_assets():
            return False

        # 1. Load Data
        logger.info("📥 Loading latest production data...")
        df = load_and_merge_data()
        
        if df.empty:
            logger.warning("❌ No data available for reporting.")
            return False

        # 2. Align Features
        # Ensure input DF has all required columns (fill 0 if missing)
        X = df.copy()
        for f in self.features:
            if f not in X.columns:
                X[f] = 0
        X = X[self.features]

        # 3. Preprocess (Impute Only First)
        # We need unscaled data for the advice engine, but imputed data for validity
        logger.info("🛠️  Running Preprocessing...")
        X_imputed = pd.DataFrame(self.imputer.transform(X), columns=self.features, index=X.index)
        
        # Scale for final prediction
        X_scaled = self.scaler.transform(X_imputed)

        # 4. Batch Prediction
        logger.info(f"🔮 Scoring {len(df)} production records...")
        probs = self.model.predict_proba(X_scaled)[:, 1]
        
        # 5. Classify Risk
        df['Scrap_Probability'] = probs
        df['Risk_Level'] = pd.cut(
            probs, 
            bins=[-0.1, self.threshold - 0.1, self.threshold, 0.85, 1.1], 
            labels=["SAFE 🟢", "WARNING 🟡", "HIGH RISK 🔴", "CRITICAL 🛑"]
        )

        # 6. Generate Advice (Loop)
        # Only run deep simulation for High/Critical risk to save time
        logger.info("🧠 Generating Prescriptive Advice (This may take a moment)...")
        advice_list = []
        
        # Convert to records for faster iteration
        records = X_imputed.to_dict('records')
        
        for idx, row in enumerate(records):
            prob = probs[idx]
            # Only simulate if risk is above threshold to save compute time
            if prob >= self.threshold:
                advice = self.get_prescriptive_advice(row, prob)
            else:
                advice = "✅ Maintain Parameters"
            advice_list.append(advice)
            
        df['AI_Recommendation'] = advice_list

        # 7. Format & Export
        # Select user-friendly columns
        base_cols = ['machine_id', 'timestamp', 'Risk_Level', 'Scrap_Probability', 'AI_Recommendation', 'scrap_rate']
        # Add top 5 features for context
        top_features = self.features[:5]
        export_cols = base_cols + top_features
        
        # Filter existing columns
        final_cols = [c for c in export_cols if c in df.columns]
        
        # Sort by Risk (Worst first)
        final_report = df[final_cols].sort_values('Scrap_Probability', ascending=False)
        
        # Save Report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = self.reports_dir / f"Scrap_Analysis_Report_{timestamp}.csv"
        
        final_report.to_csv(filename, index=False)
        
        # Create "Action List" (Filtered view)
        action_filename = self.reports_dir / f"ACTION_REQUIRED_{timestamp}.csv"
        action_items = final_report[final_report['Scrap_Probability'] >= self.threshold]
        action_items.to_csv(action_filename, index=False)

        # Summary
        logger.info("-" * 40)
        logger.info("📊 REPORT GENERATION COMPLETE")
        logger.info("-" * 40)
        logger.info(f"📄 Full Report:    {filename}")
        logger.info(f"🚨 Action Items:   {len(action_items)} orders require attention.")
        logger.info(f"📂 Saved to:       {action_filename}")
        logger.info("-" * 40)
        
        return True

# =========================================================
# 🌉 BRIDGE FUNCTION (Fixes the Import Error)
# =========================================================
def generate_report():
    """
    Wrapper function that the pipeline expects to call.
    This creates the reporter instance and runs the generation process.
    Returns True if successful, False otherwise.
    """
    try:
        reporter = PrescriptiveReporter()
        success = reporter.generate()
        return success
    except Exception as e:
        logger.error(f"❌ Critical Error in Reporting Engine: {e}")
        traceback.print_exc()
        return False

# =========================================================
# MAIN EXECUTION
# =========================================================
if __name__ == "__main__":
    # If run directly, execute the function
    generate_report()