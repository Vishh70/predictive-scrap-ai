import os
from pathlib import Path

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
PARAM_PATTERN = "*Param*.csv"      # Machine Sensor Data
BACKUP_PATTERN = "*Backup*.csv"    # For archival

# =========================================================
# ⚙️ 3. SENSOR & PHYSICS CONFIGURATION (THE "CONTRACT")
# =========================================================
# This is the "Golden List". The AI will ONLY look at these columns.
# It protects the model from training on noise or accidental ID columns.

REQUIRED_PARAM_COLS = [
    # -- Times (Speed) --
    "cycle_time",
    "injection_time",
    "plasticizing_time",
    "dosage_time",
    
    # -- Pressures (Force) --
    "max_injection_pressure",
    "injection_pressure",
    "switch_pressure",
    "back_pressure",
    
    # -- Volumes & Positions (Geometry) --
    "switch_over_volume",
    "cushion",               # Critical for part density
    "plasticizing_position",
    
    # -- Temperatures (Thermodynamics) --
    "barrel_temperature",
    "mold_temperature",
    "oil_temperature",
    "melt_temp",
    
    # -- Zone Specific Temps --
    "cyl_tmp_z1", "cyl_tmp_z2", "cyl_tmp_z3", "cyl_tmp_z4", "cyl_tmp_z5"
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

# Business Rules
SCRAP_RATE_THRESHOLD = 0.02  # If >2% of a batch is bad, label the whole batch as "High Risk"

# =========================================================
# 🖥️ 5. DEPLOYMENT FLAGS
# =========================================================
DEBUG_MODE = False       # Set to True to see verbose logs in terminal
ENABLE_CACHE = True      # Speed up reloading by saving processed pickles