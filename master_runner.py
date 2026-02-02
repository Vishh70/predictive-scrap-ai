import subprocess
import sys
import time
import os
import signal
from pathlib import Path

# =========================================================
# 🏭 MASTER FACTORY CONTROLLER
# =========================================================
# This script runs the entire AI lifecycle automatically.
# It handles infinite loops (like simulations) by timing them out.

def run_step(script_name, description, timeout=None):
    """
    Runs a python script. 
    If 'timeout' is set, it kills the process after N seconds (good for simulations).
    """
    print("\n" + "="*70)
    print(f"🔄 STEP: {description}")
    print(f"📄 Running: src/{script_name}")
    print("="*70)
    
    start_t = time.time()
    module_name = "src." + script_name.replace(".py", "")
    
    try:
        # Start the process
        process = subprocess.Popen(
            [sys.executable, "-m", module_name],
            stdout=sys.stdout, 
            stderr=sys.stderr
        )
        
        # Wait for finish or timeout
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            print(f"\n⏱️  Time Limit ({timeout}s) reached. Stopping {script_name}...")
            process.terminate() # Safe kill
            try:
                process.wait(timeout=2)
            except:
                process.kill() # Force kill if stuck
            print("✅ Process stopped successfully. Moving next...")
            return True

        if process.returncode == 0:
            duration = time.time() - start_t
            print(f"✅ SUCCESS: {script_name} finished in {duration:.2f}s")
            return True
        else:
            print(f"❌ ERROR: {script_name} exited with code {process.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ EXECUTION FAILED: {e}")
        return False

def launch_dashboard():
    """Launches Streamlit in a separate non-blocking window."""
    print("\n" + "="*70)
    print("🚀 FINAL STEP: Launching AI Dashboard")
    print("="*70)
    print("   The dashboard will open in your browser automatically.")
    print("   Press Ctrl+C in this terminal to stop everything.")
    
    app_path = Path("src/app.py")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])
    except KeyboardInterrupt:
        print("\n👋 Factory System Shutting Down.")

def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    print("🏭 TE CONNECTIVITY AI - AUTOMATED DEPLOYMENT SYSTEM")
    print(f"📂 Execution Root: {os.getcwd()}")
    
    # --- 1. CLEAN & LOAD ---
    # Robust loading with auto-fill for missing sensors
    run_step("data_loading.py", "Data Ingestion & Feature Engineering")
    
    # --- 2. TRAIN ---
    # XGBoost optimization with F2-Score tuning
    run_step("train_model.py", "AI Model Training & Optimization")
    
    # --- 3. TEST INFERENCE ---
    # Runs the "Matrix" simulation for 15 seconds, then auto-stops
    run_step("predict_new.py", "Real-Time Simulation (15s Test)", timeout=15)
    
    # --- 4. GENERATE REPORTS ---
    # Creates the Action List CSVs
    run_step("generate_report.py", "Generating Business Intelligence Reports")
    
    # --- 5. ORCHESTRATE ---
    # Runs the full engine logic check
    run_step("main_engine.py", "Full Pipeline Integration Check")
    
    # --- 6. LAUNCH UI ---
    launch_dashboard()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Execution cancelled by user.")