from __future__ import annotations

import json
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [REPORTING] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("TE_Reporter")


try:
    import src.config as cfg
    from src.data_loading import load_and_merge_data
except ImportError:
    sys.path.append(str(Path.cwd().parent))
    try:
        import src.config as cfg
        from src.data_loading import load_and_merge_data
    except ImportError:
        logger.warning("Could not import src.config. Using fallback defaults.")

        class cfg:
            MODELS_DIR = Path("models")
            REPORTS_DIR = Path("reports")
            DATA_DIR = Path("data")

        def load_and_merge_data() -> pd.DataFrame:
            return pd.DataFrame()


class PrescriptiveReporter:
    """
    Layer 2 Prescriptive AI:
    Converts scrap-risk predictions into physical, operator-level actions.
    """

    def __init__(self) -> None:
        self.models_dir = getattr(cfg, "MODELS_DIR", Path("models"))
        self.reports_dir = getattr(cfg, "REPORTS_DIR", Path("reports"))
        self.data_dir = getattr(cfg, "DATA_DIR", Path("data"))
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.features: List[str] = []
        self.imputer = None
        self.scaler = None
        self.threshold: float = 0.5

        self.rule_thresholds = self._load_rule_thresholds()
        self.causal_map = {
            "cushion": "Check Switch Position & Holding Pressure",
            "injection_time": "Check Injection Speed profile",
            "dosage_time": "Check Dosing Speed (RPM) and material feeding",
            "plasticizing_time": "Check Dosing Speed (RPM) and material feeding",
            "cycle_time": "Check Mold Movements & cooling/injection speeds",
            "cyl_tmp_z": "Check Heating Zones and barrel thermal stability",
        }

    def _load_rule_thresholds(self) -> Dict[str, float]:
        """Loads dynamic thresholds from data/monitoring_config.json with defaults."""
        defaults = {
            "cushion": 0.5,
            "injection_time": 0.03,
            "dosage_time": 1.0,
            "plasticizing_time": 1.0,
            "injection_pressure": 100.0,
            "switch_pressure": 100.0,
            "cyl_tmp_z": 5.0,
        }
        cfg_path = self.data_dir / "monitoring_config.json"
        if not cfg_path.exists():
            return defaults

        try:
            raw = json.loads(cfg_path.read_text(encoding="utf-8"))
            for key, val in raw.items():
                if isinstance(val, dict) and "threshold" in val:
                    try:
                        defaults[str(key).strip().lower()] = float(val["threshold"])
                    except Exception:
                        continue
            return defaults
        except Exception as exc:
            logger.warning("Failed to read monitoring_config.json (%s). Using defaults.", exc)
            return defaults

    def load_assets(self) -> bool:
        """Loads trained model artifacts required for inference."""
        try:
            model_path = self.models_dir / "scrap_model.joblib"
            feat_path = self.models_dir / "model_features.joblib"
            imp_path = self.models_dir / "imputer.joblib"
            scl_path = self.models_dir / "scaler.joblib"
            thr_path = self.models_dir / "threshold.joblib"

            if not model_path.exists():
                logger.error("Missing model artifact: %s", model_path)
                return False
            self.model = joblib.load(model_path)

            if feat_path.exists():
                self.features = list(joblib.load(feat_path))
            else:
                self.features = list(getattr(self.model, "training_features_", []))

            if not self.features:
                logger.error("Feature list not found in artifacts/model metadata.")
                return False

            if imp_path.exists():
                self.imputer = joblib.load(imp_path)
            if scl_path.exists():
                self.scaler = joblib.load(scl_path)
            if thr_path.exists():
                self.threshold = float(joblib.load(thr_path))

            logger.info("Assets loaded. Threshold: %.2f", self.threshold)
            return True
        except Exception as exc:
            logger.error("Failed to load assets: %s", exc)
            return False

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            numeric = float(value)
            if np.isnan(numeric):
                return None
            return numeric
        except Exception:
            return None

    def _rule_breach(self, row: Dict[str, Any], baseline: Dict[str, float], key: str, threshold: float) -> bool:
        value = self._to_float(row.get(key))
        target = self._to_float(baseline.get(key))
        if value is None or target is None:
            return False
        return abs(value - target) > threshold

    def _cylinder_zone_breach(self, row: Dict[str, Any], baseline: Dict[str, float], threshold: float) -> bool:
        zone_cols = [k for k in row.keys() if str(k).lower().startswith("cyl_tmp_z")]
        for col in zone_cols:
            value = self._to_float(row.get(col))
            target = self._to_float(baseline.get(col))
            if value is None or target is None:
                continue
            if abs(value - target) > threshold:
                return True
        return False

    def get_prescriptive_advice(self, row: Dict[str, Any], risk_score: float, baseline: Dict[str, float]) -> str:
        """
        Causal prescriptive logic:
        - Cushion -> Switch Position & Holding Pressure
        - Injection Time -> Injection Speed
        - Dosage Time -> Dosing Speed
        - Cycle Time -> Mold Movements & Speeds
        - Cyl_tmp -> Heating Zones
        """
        if risk_score < self.threshold:
            return "✅ Maintain Parameters"

        advice: List[str] = []

        if self._rule_breach(row, baseline, "cushion", self.rule_thresholds.get("cushion", 0.5)):
            advice.append("🔴 Cushion Drift: Check Switch Position & Holding Pressure.")

        if self._rule_breach(row, baseline, "injection_time", self.rule_thresholds.get("injection_time", 0.03)):
            advice.append("⚠️ Injection Time Drift: Check Injection Speed profile.")

        dosage_threshold = self.rule_thresholds.get("dosage_time", self.rule_thresholds.get("plasticizing_time", 1.0))
        dosage_breach = self._rule_breach(row, baseline, "dosage_time", dosage_threshold) or self._rule_breach(
            row, baseline, "plasticizing_time", dosage_threshold
        )
        if dosage_breach:
            advice.append("⚠️ Dosage Drift: Check Dosing Speed (RPM) and material feeding.")

        cycle_base = self._to_float(baseline.get("cycle_time"))
        cycle_threshold = max(1.0, (cycle_base or 0.0) * 0.05)
        if self._rule_breach(row, baseline, "cycle_time", cycle_threshold):
            advice.append("ℹ️ Cycle Time Drift: Check Mold Movements & speeds.")

        cyl_threshold = self.rule_thresholds.get("cyl_tmp_z", 5.0)
        if self._cylinder_zone_breach(row, baseline, cyl_threshold):
            advice.append("🌡️ Heating Zone Drift: Check cylinder zone temperatures and heater stability.")

        if hasattr(self.model, "feature_importances_") and self.features:
            try:
                importances = np.asarray(self.model.feature_importances_)
                if len(importances) == len(self.features):
                    top_idx = np.argsort(importances)[::-1][:3]
                    top_factors = [self.features[i] for i in top_idx]
                    advice.append(f"🔍 Top Learned Risk Factors: {', '.join(top_factors)}")
            except Exception:
                pass

        if not advice:
            return "⚠️ High Risk: Investigate Injection Speed, Dosing, and Mold movements."
        return " | ".join(advice)

    def generate(self) -> bool:
        if not self.load_assets():
            return False

        logger.info("Loading latest merged production data...")
        df = load_and_merge_data()
        if df.empty:
            logger.warning("No data available for report generation.")
            return False

        X = df.copy()
        for f in self.features:
            if f not in X.columns:
                X[f] = np.nan
        X = X[self.features]

        if self.imputer is not None:
            X_imputed = pd.DataFrame(self.imputer.transform(X), columns=self.features, index=X.index)
        else:
            X_imputed = X.fillna(0)

        if self.scaler is not None:
            X_scaled = self.scaler.transform(X_imputed)
        else:
            X_scaled = X_imputed.values

        probs = self.model.predict_proba(X_scaled)[:, 1]
        df["Scrap_Probability"] = probs

        safe_limit = max(0.0, self.threshold - 0.1)
        critical_limit = max(0.90, self.threshold + 0.05)
        conditions = [
            (df["Scrap_Probability"] < safe_limit),
            (df["Scrap_Probability"] < self.threshold),
            (df["Scrap_Probability"] < critical_limit),
        ]
        choices = ["SAFE 🟢", "WARNING 🟡", "HIGH RISK 🔴"]
        df["Risk_Level"] = np.select(conditions, choices, default="CRITICAL 🛑")

        numeric_df = df.apply(pd.to_numeric, errors="coerce")
        baseline = numeric_df.median(numeric_only=True).to_dict()
        records = df.to_dict("records")

        advice_list: List[str] = []
        for idx, row in enumerate(records):
            risk = float(probs[idx])
            advice_list.append(self.get_prescriptive_advice(row, risk, baseline))
        df["AI_Recommendation"] = advice_list

        base_cols = [
            "machine_id",
            "timestamp",
            "Risk_Level",
            "Scrap_Probability",
            "AI_Recommendation",
            "scrap_rate",
        ]
        final_cols = [c for c in base_cols + self.features[:5] if c in df.columns]
        final_report = df[final_cols].sort_values("Scrap_Probability", ascending=False)

        ts = datetime.now().strftime("%Y%m%d_%H%M")
        full_path = self.reports_dir / f"Scrap_Analysis_Report_{ts}.csv"
        action_path = self.reports_dir / f"ACTION_REQUIRED_{ts}.csv"

        final_report.to_csv(full_path, index=False)
        action_items = final_report[final_report["Scrap_Probability"] >= self.threshold]
        action_items.to_csv(action_path, index=False)

        logger.info("Report generation complete.")
        logger.info("Full Report: %s", full_path)
        logger.info("Action Items: %d", len(action_items))
        return True


def generate_report() -> bool:
    """Pipeline entrypoint expected by run_pipeline.py."""
    try:
        reporter = PrescriptiveReporter()
        return reporter.generate()
    except Exception as exc:
        logger.error("Critical reporting failure: %s", exc)
        traceback.print_exc()
        return False


if __name__ == "__main__":
    generate_report()

