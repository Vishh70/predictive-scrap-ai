# Predictive Scrap System - Architecture & Design

## 1. Problem Statement
In manufacturing, scrap is usually detected **after** production, leading to wasted material and time. Current tools are reactive. This system aims to be **predictive**, allowing engineers to intervene before scrap is produced.

## 2. Core Concept
The system predicts scrap quantity for a Machine–Tool–Part combination using a single AI model that operates in two distinct modes:
1.  **Planning Mode:** Uses historical baselines to forecast scrap before production starts.
2.  **Monitoring Mode:** Uses live runtime data to adapt the forecast during production.

## 3. System Architecture

### A. The AI Model (XGBoost)
* **Role:** Predicts probability of scrap based on sensor inputs.
* **Features:** Uses 39 features, including:
    * **Current Values:** Pressure, Temperature, Cycle Time.
    * **Lag Features:** Sensor values from 5 minutes ago.
    * **Rolling Stats:** Average and StdDev over the last 15 minutes.
* **Training:** Trained offline on historical data. Does **not** retrain in real-time to ensure stability.

### B. Planning Mode (The "Baseline")
* **When:** Before the shift begins.
* **Input:** Machine ID.
* **Process:** System looks up the "Average Behavior" of that machine from history.
* **Output:** A baseline risk score (e.g., "This machine typically runs at 12% risk").
* **User Action:** Engineer decides whether to schedule maintenance or proceed.

### C. Monitoring Mode (The "Live Stream")
* **When:** During production.
* **Input:** Real-time stream of sensor data (Pressure, Temp, etc.).
* **Process:**
    1.  Data enters the system.
    2.  System calculates "Rolling Averages" on the fly.
    3.  Model predicts risk for the *current* state.
* **Output:** Dynamic risk chart. Alerts if risk crosses the 35% threshold.

## 4. Data Lifecycle Strategy
* **Continuous:** Runtime data is logged and scrap outcomes are stored.
* **Discrete:** The historical dataset is updated periodically (e.g., daily).
* **Feedback Loop:** Today's "Monitoring Data" becomes tomorrow's "Planning History".

## 5. Technology Stack
* **Language:** Python 3.10+
* **Model:** XGBoost Classifier
* **Data Handling:** Pandas (Rolling/Lag engineering)
* **Imbalance Handling:** SMOTE
* **Visualization:** Streamlit & Plotly