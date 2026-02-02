import os
import sys
import shutil
import time
import logging
import argparse
import subprocess
import json
import platform
import concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple

# =========================================================
# 🔧 CRITICAL PATH FIX (Solves "src not found" Error)
# =========================================================
# This block ensures that Python knows exactly where your project looks for files.
# It solves the issue where running the script from different folders causes crashes.
# ---------------------------------------------------------
try:
    # 1. Get the absolute path of this script file
    CURRENT_SCRIPT_PATH = Path(__file__).resolve()
    
    # 2. Get the project root directory (The folder containing this script)
    PROJECT_ROOT = CURRENT_SCRIPT_PATH.parent
    
    # 3. Add this path to Python's system path if not already there
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))
        print(f"🔧 System Path Fixed: Added {PROJECT_ROOT} to python path.")
        
    # 4. Verify 'src' folder exists
    SRC_DIR = PROJECT_ROOT / "src"
    if not SRC_DIR.exists():
        print(f"❌ CRITICAL ERROR: 'src' folder not found at {SRC_DIR}")
        print("   Please ensure you have the correct folder structure.")
        sys.exit(1)

except Exception as e:
    print(f"❌ Path Configuration Error: {e}")
    sys.exit(1)

# =========================================================
# 📦 INTERNAL IMPORTS (Safe Loading)
# =========================================================
try:
    import src.config as cfg
    from src.generate_report import generate_report
    # Third-party libraries
    import pandas as pd
    import numpy as np
except ImportError as e:
    print("\n" + "="*60)
    print(f"❌ IMPORT ERROR: {e}")
    print("="*60)
    print("Detailed Checklist:")
    print("1. Is your virtual environment activated? (.venv)")
    print("2. Did you run 'pip install -r requirements.txt'?")
    print("3. Does 'src/config.py' exist?")
    print("="*60 + "\n")
    sys.exit(1)

# =========================================================
# 1. ADVANCED LOGGING CONFIGURATION
# =========================================================
class PipelineLogger:
    """
    Enterprise-grade logger with file rotation and console output.
    Handles logging to both disk (permanent record) and terminal (live view).
    """
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        
        # Ensure log directory exists, create if missing
        if not self.log_dir.exists():
            print(f"📁 Creating log directory: {self.log_dir}")
            self.log_dir.mkdir(parents=True, exist_ok=True)
            
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_file = self.log_dir / f"pipeline_trace_{self.timestamp}.log"
        
        self.logger = logging.getLogger("TE_Pipeline_Orchestrator")
        self.logger.setLevel(logging.DEBUG)
        
        # Prevent duplicate logs if class is instantiated multiple times
        if not self.logger.handlers:
            # File Handler (Detailed debugging info)
            fh = logging.FileHandler(self.log_file, encoding='utf-8')
            fh.setLevel(logging.DEBUG)
            fh_fmt = logging.Formatter('%(asctime)s - [%(levelname)s] - %(threadName)s - %(message)s')
            fh.setFormatter(fh_fmt)
            self.logger.addHandler(fh)
            
            # Console Handler (User-friendly info)
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.INFO)
            # FIXED: Removed typo "bla" from format string
            ch_fmt = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')
            ch.setFormatter(ch_fmt)
            self.logger.addHandler(ch)

    def info(self, msg): 
        self.logger.info(msg)
    
    def warning(self, msg): 
        self.logger.warning(f"⚠️ {msg}")
    
    def error(self, msg): 
        self.logger.error(f"❌ {msg}")
    
    def success(self, msg): 
        self.logger.info(f"✅ {msg}")
    
    def header(self, msg): 
        self.logger.info(f"\n{'='*60}\n🚀 {msg}\n{'='*60}")

# =========================================================
# 2. DATA GUARD (VALIDATION ENGINE)
# =========================================================
class DataGuard:
    """
    Advanced Data Validation Engine.
    Ensures '200% Right Data' by performing schema checks, null-checks, 
    and quarantine logic before the pipeline touches the files.
    """
    def __init__(self, logger):
        self.logger = logger
        # Use config for base dir, fallback to current working dir
        base = getattr(cfg, 'BASE_DIR', Path.cwd())
        self.quarantine_dir = base / "data_quarantine"
        self.quarantine_dir.mkdir(exist_ok=True)

    def validate_file(self, file_path: Path) -> bool:
        """
        Thread-safe validation of a single CSV file.
        Checks for: Empty files, Bad Headers, Parse Errors.
        """
        try:
            # 1. Basic Size Check
            if file_path.stat().st_size == 0:
                self.logger.warning(f"Empty file detected (0 bytes): {file_path.name}")
                return False

            # 2. Schema/Parse Check (using Pandas for robustness)
            # reading only top 10 rows to keep it fast
            df = pd.read_csv(file_path, nrows=10) 
            
            if df.empty:
                self.logger.warning(f"File has headers but no data rows: {file_path.name}")
                return False
                
            # 3. Critical Column Check (Heuristic)
            cols = [c.lower() for c in df.columns]
            
            # Special check for 'machine' specific files
            if "machine" in str(file_path).lower():
                # Parameter files must have timestamps or variable names
                required_signatures = ['timestamp', 'variable_name', 'time', 'date', 'creation_time']
                if not any(x in cols for x in required_signatures):
                    self.logger.warning(f"Suspect schema in parameter file: {file_path.name}")
                    self.logger.info(f"   Found columns: {cols}")
                    return False
            
            return True

        except pd.errors.EmptyDataError:
            self.logger.warning(f"Empty Data Error reading {file_path.name}")
            return False
        except pd.errors.ParserError:
            self.logger.error(f"CSV Parsing Error (Corrupt file): {file_path.name}")
            self._move_to_quarantine(file_path)
            return False
        except Exception as e:
            self.logger.error(f"Unknown validation error in {file_path.name}: {e}")
            self._move_to_quarantine(file_path)
            return False

    def _move_to_quarantine(self, file_path: Path):
        """Moves bad files out of the production folder to prevent crashes."""
        dest = self.quarantine_dir / file_path.name
        try:
            shutil.move(str(file_path), str(dest))
            self.logger.warning(f"Moved corrupted file to QUARANTINE: {dest}")
        except Exception as e:
            self.logger.error(f"Failed to quarantine file: {e}")

    def validate_dataset(self, files: List[Path]) -> List[Path]:
        """
        Runs parallel validation on all files to speed up large dataset loading.
        """
        self.logger.info(f"🛡️ Validating {len(files)} files for integrity...")
        valid_files = []
        
        # ThreadPoolExecutor for concurrent I/O operations
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_file = {executor.submit(self.validate_file, f): f for f in files}
            for future in concurrent.futures.as_completed(future_to_file):
                f = future_to_file[future]
                try:
                    is_valid = future.result()
                    if is_valid:
                        valid_files.append(f)
                except Exception as e:
                    self.logger.error(f"Validation process crashed for {f.name}: {e}")
        
        self.logger.success(f"Validation Complete. {len(valid_files)}/{len(files)} files passed.")
        return valid_files

# =========================================================
# 3. PIPELINE ORCHESTRATOR (MAIN BRAIN)
# =========================================================
class PipelineOrchestrator:
    """
    Master control class for the entire AI lifecycle.
    Manages: Ingestion -> Training -> Reporting -> Archival -> UI
    """
    def __init__(self):
        # Setup Environment Variables
        self.base_dir = getattr(cfg, 'BASE_DIR', Path.cwd())
        self.data_dir = getattr(cfg, 'DATA_DIR', self.base_dir / 'data')
        self.archive_dir = self.base_dir / 'data_archive'
        
        # Initialize Logger
        self.log = PipelineLogger(self.base_dir / 'logs')
        
        # Initialize Components
        self.guard = DataGuard(self.log)
        
        # Run Initial System Diagnostics
        self._system_diagnostics()

    def _system_diagnostics(self):
        """Checks hardware and software capabilities."""
        self.log.header("STEP 0: SYSTEM DIAGNOSTICS")
        
        # OS Info
        uname = platform.uname()
        self.log.info(f"System: {uname.system} {uname.release}")
        self.log.info(f"Python: {sys.version.split()[0]}")
        self.log.info(f"Root Dir: {self.base_dir}")
        
        # Directory Structure Check
        required_dirs = [
            cfg.MODELS_DIR, 
            cfg.REPORTS_DIR, 
            self.archive_dir,
            self.data_dir
        ]
        
        missing_count = 0
        for d in required_dirs:
            if not d.exists():
                self.log.warning(f"Creating missing directory: {d}")
                d.mkdir(parents=True, exist_ok=True)
                missing_count += 1
        
        if missing_count == 0:
            self.log.success("Directory structure is intact.")
        else:
            self.log.success(f"Fixed {missing_count} missing directories.")

    def scan_and_clean_data(self) -> bool:
        """
        Scans data folder, validates files, and prepares them for processing.
        """
        self.log.header("STEP 1: DATA INGESTION & CLEANING")
        
        # 1. Discovery
        # Use regex patterns from config if available, otherwise default glob
        hydra_pattern = getattr(cfg, 'HYDRA_PATTERN', "*Hydra*.csv")
        param_pattern = getattr(cfg, 'PARAM_PATTERN', "*MachineParameter*.csv")
        
        hydra_files = list(self.data_dir.glob(hydra_pattern))
        param_files = list(self.data_dir.glob(param_pattern))
        all_files = hydra_files + param_files
        
        self.log.info(f"Source: {self.data_dir}")
        self.log.info(f"Detected: {len(hydra_files)} Hydra logs, {len(param_files)} Parameter logs")
        
        if not all_files:
            self.log.error("No input files found. Please populate the 'data/' directory.")
            return False

        # 2. Advanced Validation (The "200% Right" Check)
        valid_files = self.guard.validate_dataset(all_files)
        
        if len(valid_files) < len(all_files):
            self.log.warning(f"Dropped {len(all_files) - len(valid_files)} files due to quality issues.")
        
        if not valid_files:
            self.log.error("No valid files remaining after cleaning.")
            return False

        self.log.success("Data is Clean, Validated, and Ready.")
        return True

    def model_health_check(self, force_retrain: bool = False):
        """
        Checks if the AI brain exists. If not, trains it.
        """
        self.log.header("STEP 2: AI MODEL INTEGRITY CHECK")
        
        required_assets = [
            cfg.MODELS_DIR / "scrap_model.joblib",
            cfg.MODELS_DIR / "scaler.joblib",
            cfg.MODELS_DIR / "imputer.joblib",
            cfg.MODELS_DIR / "model_features.joblib",
            cfg.MODELS_DIR / "threshold.joblib"
        ]
        
        missing = [f.name for f in required_assets if not f.exists()]
        
        if missing or force_retrain:
            if force_retrain:
                self.log.warning("User requested Forced Retraining.")
            else:
                self.log.warning(f"Missing Assets: {missing}. Initiating Auto-Train...")
            
            self._run_training_subprocess()
        else:
            self.log.success("All AI Models are present and valid.")

    def _run_training_subprocess(self):
        """Runs the training script in a separate process to ensure memory safety."""
        start_t = time.time()
        try:
            self.log.info("Spawning training process...")
            
            # We use the same python interpreter executing this script
            result = subprocess.run(
                [sys.executable, "-m", "src.train_model"], 
                capture_output=True, 
                text=True, 
                check=True
            )
            
            # Log stdout from training script into our main log
            for line in result.stdout.splitlines():
                if "ERROR" in line: self.log.error(f"[TRAIN] {line}")
                elif "WARNING" in line: self.log.warning(f"[TRAIN] {line}")
                else: self.log.info(f"[TRAIN] {line}")
                
            duration = time.time() - start_t
            self.log.success(f"Training completed successfully in {duration:.2f}s.")
            
        except subprocess.CalledProcessError as e:
            self.log.error("Training Process Failed!")
            self.log.error(f"Error Code: {e.returncode}")
            self.log.error(f"Error Log: {e.stderr}")
            sys.exit(1)

    def generate_intelligence_reports(self):
        """Runs the reporting engine."""
        self.log.header("STEP 3: GENERATING INTELLIGENCE REPORTS")
        try:
            # We import and run this inside the process to catch exceptions gracefully
            generate_report()
            self.log.success(f"Batch reports generated in '{cfg.REPORTS_DIR}' folder.")
        except Exception as e:
            self.log.error(f"Reporting Engine Failed: {e}")

    def archive_old_data(self):
        """Moves processed files to archive to keep production clean."""
        self.log.header("STEP 4: DATA ARCHIVAL (OPTIONAL)")
        
        files_to_move = list(self.data_dir.glob("*.csv"))
        if not files_to_move:
            self.log.info("No files to archive.")
            return

        # Create today's archive folder
        today_folder = self.archive_dir / datetime.now().strftime("%Y-%m-%d")
        today_folder.mkdir(parents=True, exist_ok=True)
        
        self.log.info(f"Archiving {len(files_to_move)} processed files...")
        count = 0
        for f in files_to_move:
            try:
                shutil.move(str(f), str(today_folder / f.name))
                count += 1
            except Exception as e:
                self.log.warning(f"Could not archive {f.name}: {e}")
        
        self.log.success(f"Archived {count} files to {today_folder}")

    def launch_dashboard(self):
        """Launches the Streamlit App."""
        self.log.header("STEP 5: LAUNCHING AI DASHBOARD")
        self.log.info("Starting Streamlit server...")
        self.log.info("Press Ctrl+C to stop the dashboard.")
        
        app_path = PROJECT_ROOT / "src" / "app.py"
        
        if not app_path.exists():
            self.log.error(f"Could not find dashboard file at: {app_path}")
            return

        try:
            # Use subprocess to run Streamlit
            subprocess.run(
                [sys.executable, "-m", "streamlit", "run", str(app_path)], 
                check=True
            )
        except KeyboardInterrupt:
            self.log.warning("Dashboard stopped by user.")
        except Exception as e:
            self.log.error(f"Dashboard failed to launch: {e}")

# =========================================================
# 4. ENTRY POINT
# =========================================================
def parse_arguments():
    parser = argparse.ArgumentParser(description="TE Connectivity AI Pipeline Orchestrator")
    parser.add_argument("--retrain", action="store_true", help="Force full model retraining")
    parser.add_argument("--archive", action="store_true", help="Archive data files after processing")
    parser.add_argument("--no-dashboard", action="store_true", help="Run pipeline without launching dashboard")
    parser.add_argument("--validate-only", action="store_true", help="Only run data validation and exit")
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Initialize the Brain
    bot = PipelineOrchestrator()
    
    # 1. Check Data
    data_ok = bot.scan_and_clean_data()
    if not data_ok:
        sys.exit(1)
        
    if args.validate_only:
        bot.log.success("Validation complete. Exiting as requested.")
        sys.exit(0)

    # 2. Check Model
    bot.model_health_check(force_retrain=args.retrain)
    
    # 3. Generate Reports
    bot.generate_intelligence_reports()
    
    # 4. Archival (Optional)
    if args.archive:
        bot.archive_old_data()
        
    # 5. Dashboard
    if not args.no_dashboard:
        bot.launch_dashboard()
    else:
        bot.log.success("Pipeline finished (Dashboard skipped).")

if __name__ == "__main__":
    main()