import os
import logging
from pathlib import Path
from typing import Optional, Union

# =========================================================
# 📂 1. PROJECT DIRECTORY ARCHITECTURE
# =========================================================
# Automatically detects the root folder, no matter where this runs.
# Works on Windows, Linux, and Cloud (AWS/Azure).
BASE_DIR = Path(__file__).resolve().parent.parent

# Define the "Data Lake" structure
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARCHIVE_DIR = DATA_DIR / "archive"

# Define Artifact Storage (Brain of the AI)
MODELS_DIR = DATA_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
LOGS_DIR = BASE_DIR / "logs"

# Automatic Folder Creation (Self-Healing)
# If a folder is missing, the code creates it instantly.
for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR, ARCHIVE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =========================================================
# 📄 2. DATA DISCOVERY PATTERNS
# =========================================================
# The pipeline uses these patterns to find the right CSV files.
# Case-insensitive matching is handled in the loading script.
HYDRA_PATTERN = "*Hydra*.csv"      # Production Order Data
PARAM_PATTERN = "*.csv"            # Scan all CSV files; loader keeps only valid parameter schemas
BACKUP_PATTERN = "*Backup*.csv"    # For archival

# =========================================================
# ⚙️ 3. SENSOR & PHYSICS CONFIGURATION (THE "CONTRACT")
# =========================================================

# 1. TRANSLATION MAP (Excel Header -> Python Name)
# This dictionary maps the strange names in your CSV to clean names for AI.
COLUMN_MAPPING = {
    # Time & Speed
    "Act Cycle Time":       "cycle_time",
    "Act Inj Time":         "injection_time",
    "Act Fill Time":        "fill_time",
    "Act Charge Time":      "plasticizing_time",  # 'Charge' often equals 'Plasticizing'

    # Pressures
    "Max Inj Pres":         "max_injection_pressure",
    "Act Hold Pres":        "holding_pressure",
    "Act Back Pres":        "back_pressure",
    "Act Switch Pres":      "switch_pressure",

    # Geometry / Positions
    "Act Switch Vol":       "switch_over_volume",
    "Act Cushion":          "cushion",
    "Act Charge Pos":       "plasticizing_position",

    # Temperatures
    "Act Melt Temp":        "melt_temp",
    "Act Mold Temp":        "mold_temperature",
    "Act Oil Temp":         "oil_temperature",
    
    # Zone Temperatures (Barrel Heating)
    "Act Zone 1 Temp":      "cyl_tmp_z1",
    "Act Zone 2 Temp":      "cyl_tmp_z2",
    "Act Zone 3 Temp":      "cyl_tmp_z3",
    "Act Zone 4 Temp":      "cyl_tmp_z4",
    "Act Zone 5 Temp":      "cyl_tmp_z5"
}

# 2. THE GOLDEN LIST
# Canonical simplified names expected after cleaning/normalization.
REQUIRED_PARAM_COLS = [
    "injection_pressure",
    "cycle_time",
    "cushion",
    "injection_time",
    "plasticizing_time",
    "max_injection_speed",
    "transfer_pressure",
    "cyl_tmp_z1",
    "cyl_tmp_z2",
    "cyl_tmp_z3",
    "cyl_tmp_z4",
    "melt_temp",
    "mold_temperature",
    "oil_temperature",
    "holding_pressure",
    "back_pressure",
    "switch_pressure",
    "switch_over_volume",
    "plasticizing_position",
    "flange_temperature",
    "nozzle_temperature",
    "mold_protection_force_peak",
    "clamping_force_peak",
]

# =========================================================
# 🔧 4. GLOBAL AI SETTINGS
# =========================================================
# Single source of truth for model behavior.

# Reproducibility (Scientific Standard)
RANDOM_STATE = 42 

# Data Splitting
TEST_SIZE = 0.2          # 20% of data used for final exam (Validation)
VALIDATION_SPLIT = 0.1   # 10% used during training for early stopping

# Scrap Detection Tuning
DEFAULT_THRESHOLD = 0.35 # Conservative starting point. 
                         # The training script will optimize this automatically using F2-Score.

# =========================================================
# 🚨 4B. DASHBOARD SAFETY & MONITORING CONSTANTS
# =========================================================
# Centralized constants to avoid magic numbers across app/ETL code.
CRITICAL_RISK_THRESHOLD = 0.85
WARNING_RISK_THRESHOLD = 0.35
FORECAST_HORIZON_MINUTES = 60
SAFE_ZONE_SIGMA_MULTIPLIER = 3.0
VOLATILITY_WINDOW = 10

# =========================================================
# 🧪 4C. PHYSICS SAFETY GUARDRAILS
# =========================================================
PHYSICS_RANGES = {
    'injection_pressure': (0, 2000),  # Bar
    'melt_temp': (100, 400),          # Celsius
    'cycle_time': (0, 120),           # Seconds
}

# Minimum physically valid cycle time used for post-merge filtering.
MIN_VALID_CYCLE_TIME_SEC = 1.0

# Feature engineering controls
SLOPE_LAG_WINDOW = 5

# Prescriptive safety rule: saturation warning near hard limits
PARAM_SATURATION_RATIO = 0.995

# Business Rules
SCRAP_RATE_THRESHOLD = 0.02  # If >2% of a batch is bad, label the whole batch as "High Risk"

# =========================================================
# 🖥️ 5. DEPLOYMENT FLAGS
# =========================================================
DEBUG_MODE = False       # Set to True to see verbose logs in terminal
ENABLE_CACHE = True      # Speed up reloading by saving processed pickles


def get_logger(
    name: str,
    log_file: Optional[Union[str, Path]] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Build or retrieve a logger with enterprise-standard formatting.

    Format:
        %(asctime)s - [%(name)s] - %(levelname)s - %(message)s
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s - [%(name)s] - %(levelname)s - %(message)s"
    )

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
