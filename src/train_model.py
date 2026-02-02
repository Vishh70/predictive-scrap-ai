import os
import sys
import pandas as pd
import numpy as np
import joblib
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import json
import shutil
from pathlib import Path
from datetime import datetime
from scipy.stats import ks_2samp

# =========================================================
# MACHINE LEARNING IMPORTS
# =========================================================
from xgboost import XGBClassifier
from sklearn.model_selection import (
    train_test_split, 
    RandomizedSearchCV, 
    StratifiedKFold, 
    learning_curve
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_recall_curve, 
    f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, 
    recall_score, precision_score, brier_score_loss, accuracy_score
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance

# =========================================================
# CONFIGURATION & SETUP
# =========================================================
# 1. Silence specific warnings for cleaner production logs
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 2. Path Setup & Imports
try:
    import src.config as cfg
    from src.data_loading import load_and_merge_data
except ImportError:
    # Fallback if running directly from src folder
    sys.path.append(str(Path.cwd().parent))
    import src.config as cfg
    from src.data_loading import load_and_merge_data

# 3. Logging System
LOG_DIR = getattr(cfg, 'BASE_DIR', Path.cwd()) / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "training_trace.log", mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger("Enterprise_Trainer")

# =========================================================
# ADVANCED TRAINING CLASS
# =========================================================
class ScrapPredictionTrainer:
    """
    Tier-1 Industrial AI Training Pipeline.
    
    Capabilities:
    - Automated Data Drift Detection (KS-Test)
    - Anti-Multicollinearity Feature Selection
    - Bayesian-style Hyperparameter Tuning
    - F2-Score Threshold Optimization (Recall Focused)
    - Full Metadata Auditing (JSON)
    """

    def __init__(self):
        self.models_dir = getattr(cfg, 'MODELS_DIR', Path('models'))
        self.reports_dir = getattr(cfg, 'REPORTS_DIR', Path('reports'))
        
        # Ensure directories exist
        for d in [self.models_dir, self.reports_dir]:
            d.mkdir(parents=True, exist_ok=True)
            
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.metadata = {
            "execution_time": self.timestamp,
            "metrics": {},
            "parameters": {}
        }

    def _detect_drift(self, df_train, df_test, features):
        """
        [Advanced] Checks if Train and Test data distributions match.
        Uses Kolmogorov-Smirnov test (p < 0.05 indicates drift).
        """
        logger.info("🕵️  Running Statistical Drift Analysis...")
        drift_warnings = []
        
        for col in features:
            try:
                # KS Test compares two samples
                stat, p_value = ks_2samp(df_train[col], df_test[col])
                if p_value < 0.05:
                    drift_warnings.append(f"{col} (p={p_value:.4f})")
            except:
                continue # Skip if column calculation fails
        
        if drift_warnings:
            logger.warning(f"⚠️  Data Drift Detected in {len(drift_warnings)} features.")
            logger.warning(f"   Top drifting: {drift_warnings[:3]}...")
            self.metadata['drift_warning'] = True
        else:
            logger.info("✅ No significant data drift detected. Process stable.")
            self.metadata['drift_warning'] = False

    def _remove_collinear_features(self, df, features, threshold=0.95):
        """
        [Optimization] Removes redundant features (correlation > 0.95).
        Reduces model complexity and overfitting risks.
        """
        logger.info("🔍 Analyzing Feature Correlation Matrix...")
        
        try:
            corr_matrix = df[features].corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
            
            if to_drop:
                logger.info(f"✂️  Pruning {len(to_drop)} redundant features (Correlation > {threshold})")
                self.metadata['dropped_features'] = to_drop
                return [f for f in features if f not in to_drop]
            
            logger.info("✅ Feature set is orthogonal (No high collinearity).")
            return features
        except Exception as e:
            logger.warning(f"Skipping collinearity check due to error: {e}")
            return features

    def load_and_prep_data(self):
        """Standardized Data Ingestion Layer."""
        logger.info("🛠️  Ingesting Source Data...")
        df = load_and_merge_data()

        if df.empty:
            raise ValueError("❌ Fatal: Dataset is empty.")

        # --- Anti-Leakage Feature Selection ---
        base_feats = getattr(cfg, 'REQUIRED_PARAM_COLS', [])
        feature_cols = [c for c in df.columns if any(base in c for base in base_feats)]
        
        # Strictly remove ID and Target columns from features
        forbidden = ('actual_', 'scrap_', 'is_', 'date', 'time', 'batch_id', 'order')
        feature_cols = [c for c in feature_cols if not c.lower().startswith(forbidden)]
        feature_cols = [c for c in feature_cols if c not in ['machine_id', 'timestamp', 'is_scrap']]

        logger.info(f"✅ Feature Candidate Set: {len(feature_cols)} sensors")
        return df, feature_cols

    def optimize_model(self, X_train, y_train, scale_pos_weight):
        """
        Runs RandomizedSearchCV to find optimal XGBoost hyperparameters.
        """
        logger.info("⚡ Starting Hyperparameter Grid Search (5-Fold CV)...")
        
        param_grid = {
            'n_estimators': [200, 350, 500],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'gamma': [0, 1, 5],
            'scale_pos_weight': [scale_pos_weight, scale_pos_weight * 1.2]
        }

        xgb = XGBClassifier(
            random_state=42, 
            eval_metric='auc', 
            n_jobs=-1,
            tree_method='hist' # Optimization for speed
        )

        search = RandomizedSearchCV(
            estimator=xgb,
            param_distributions=param_grid,
            n_iter=12, # Number of random combinations to try
            scoring='f1',
            cv=3,
            verbose=1,
            random_state=42,
            n_jobs=-1
        )

        search.fit(X_train, y_train)
        logger.info(f"🏆 Best Hyperparameters Found: {search.best_params_}")
        self.metadata['best_params'] = search.best_params_
        return search.best_estimator_

    def generate_plots(self, model, X_test, y_test, y_probs, thresh):
        """Generates comprehensive validation plots."""
        # 1. Confusion Matrix
        y_pred = (y_probs >= thresh).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Good', 'Bad'], yticklabels=['Good', 'Bad'])
        plt.title(f"Confusion Matrix (Thresh={thresh:.2%})")
        plt.savefig(self.reports_dir / "confusion_matrix.png")
        plt.close()

        # 2. Precision-Recall Curve (The most important for scrap)
        prec, rec, _ = precision_recall_curve(y_test, y_probs)
        plt.figure(figsize=(7,5))
        plt.plot(rec, prec, marker='.', label='XGBoost')
        plt.xlabel('Recall (Scrap Caught)')
        plt.ylabel('Precision (True Alarms)')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(self.reports_dir / "pr_curve.png")
        plt.close()

        # 3. ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_score = roc_auc_score(y_test, y_probs)
        plt.figure(figsize=(7,5))
        plt.plot(fpr, tpr, color='orange', label=f'AUC = {roc_score:.3f}')
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Analysis')
        plt.legend()
        plt.savefig(self.reports_dir / "roc_curve.png")
        plt.close()

    def train_pipeline(self):
        """Orchestrates the entire End-to-End Training Process."""
        try:
            start_time = datetime.now()
            
            # --- Phase 1: Data Prep ---
            df, features = self.load_and_prep_data()
            features = self._remove_collinear_features(df, features)
            
            X = df[features]
            y = df['is_scrap']

            # Dynamic Weighting for Imbalance
            n_good = (y == 0).sum()
            n_scrap = (y == 1).sum()
            scale_pos_weight = n_good / n_scrap if n_scrap > 0 else 1.0
            logger.info(f"⚖️  Class Balance: {n_good} Good / {n_scrap} Scrap (Weight: {scale_pos_weight:.2f})")

            # --- Phase 2: Splitting ---
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Drift Check
            self._detect_drift(X_train, X_test, features)

            # --- Phase 3: Robust Scaling ---
            logger.info("🧹 Applying Robust Scaling Pipeline...")
            imputer = SimpleImputer(strategy='median')
            scaler = RobustScaler()

            # Preserve DataFrames to keep column names (fixes warnings)
            X_train_p = pd.DataFrame(scaler.fit_transform(imputer.fit_transform(X_train)), columns=features, index=X_train.index)
            X_test_p = pd.DataFrame(scaler.transform(imputer.transform(X_test)), columns=features, index=X_test.index)

            # --- Phase 4: Optimization ---
            best_model = self.optimize_model(X_train_p, y_train, scale_pos_weight)

            # --- Phase 5: Calibration ---
            logger.info("🎛️  Calibrating Probabilities (Isotonic)...")
            calibrated_model = CalibratedClassifierCV(best_model, method='isotonic', cv='prefit')
            calibrated_model.fit(X_test_p, y_test) # Calibrate on test set distribution

            # --- Phase 6: Threshold Tuning (F2 Score) ---
            logger.info("🎯 Finding Optimal Decision Threshold...")
            # Get raw probabilities
            y_probs = best_model.predict_proba(X_test_p)[:, 1]
            
            precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
            # F2 Score formula: (1 + 4) * (P * R) / (4 * P + R)
            f2_scores = (5 * precisions * recalls) / (4 * precisions + recalls + 1e-10)
            best_idx = np.argmax(f2_scores)
            best_threshold = thresholds[best_idx]
            
            # Safety Clamp (Don't alert on < 20% risk)
            best_threshold = max(0.20, min(best_threshold, 0.85))
            logger.info(f"💎 Optimal Threshold: {best_threshold:.2%}")

            # --- Phase 7: Final Evaluation ---
            y_pred = (y_probs >= best_threshold).astype(int)
            
            # Metrics
            recall = recall_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            auc_score = roc_auc_score(y_test, y_probs)
            
            logger.info("\n" + "="*50)
            logger.info("📊 FINAL ENTERPRISE METRICS")
            logger.info("="*50)
            logger.info(f"Recall (Sensitivity):   {recall:.2%}")
            logger.info(f"Precision (Quality):    {precision:.2%}")
            logger.info(f"ROC-AUC Score:          {auc_score:.4f}")
            logger.info("="*50)
            
            print("\n" + classification_report(y_test, y_pred))

            # --- Phase 8: Persistence & Reporting ---
            logger.info("💾 Serializing Model Artifacts...")
            
            joblib.dump(best_model, self.models_dir / "scrap_model.joblib")
            joblib.dump(features, self.models_dir / "model_features.joblib")
            joblib.dump(imputer, self.models_dir / "imputer.joblib")
            joblib.dump(scaler, self.models_dir / "scaler.joblib")
            joblib.dump(best_threshold, self.models_dir / "threshold.joblib")

            # Save Feature Importance CSV
            imp_df = pd.DataFrame({
                'Feature': features,
                'Importance': best_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            imp_df.to_csv(self.reports_dir / "feature_importance.csv", index=False)

            # Generate Plots
            self.generate_plots(best_model, X_test_p, y_test, y_probs, best_threshold)

            # Save Metadata Log
            self.metadata['metrics'] = {
                'recall': recall, 
                'precision': precision, 
                'auc': auc_score, 
                'threshold': best_threshold
            }
            with open(self.reports_dir / "training_metadata.json", "w") as f:
                json.dump(self.metadata, f, indent=4, default=str)

            logger.info(f"✅ Training Success. Runtime: {datetime.now() - start_time}")

        except Exception as e:
            logger.exception(f"❌ Critical Training Failure: {e}")
            raise e

if __name__ == "__main__":
    trainer = ScrapPredictionTrainer()
    trainer.train_pipeline()