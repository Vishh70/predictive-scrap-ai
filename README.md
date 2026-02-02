# Predictive Scrap AI

Predictive Scrap AI is a machine learning–based project designed to predict
scrap generation in manufacturing processes using machine parameters and
historical production data.

The main goal of this project is to help manufacturing teams take
**preventive actions before scrap occurs**, reducing cost and improving
process efficiency.

---

## Why This Project?
In manufacturing industries, scrap causes:
- Material wastage
- Increased production cost
- Quality issues

Most systems react **after** scrap happens.
This project focuses on **predicting scrap in advance** using data and ML.

---

## What This Project Does
- Loads and cleans machine/process data
- Trains a machine learning model
- Predicts scrap probability for new data
- Generates structured reports for analysis

---

## Project Structure
```

predictive-scrap-ai/
│
├── src/
│   ├── app.py              # Application entry point
│   ├── config.py           # Configuration settings
│   ├── data_loading.py     # Data loading & preprocessing
│   ├── train_model.py      # Model training logic
│   ├── predict_new.py      # Scrap prediction
│   ├── generate_report.py  # Report generation
│   └── main_engine.py      # Pipeline controller
│
├── run_pipeline.py         # Run full ML pipeline
├── master_runner.py        # High-level execution
├── requirements.txt        # Python dependencies
├── ARCHITECTURE.md         # System design details
└── README.md

````

---

## Technology Used
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Git & GitHub

---

## How to Run
1. Install dependencies
```bash
pip install -r requirements.txt
````

2. Run the pipeline

```bash
python run_pipeline.py
```

---

## Output

* Scrap prediction reports
* Model evaluation metrics
* Feature importance analysis

> Note: Datasets, trained models, and logs are excluded from GitHub
> to maintain confidentiality.

---

## Learning Outcome

This project demonstrates:

* End-to-end ML pipeline development
* Clean project structuring
* Real-world Git usage
* Industry-level coding practices

---

## Author

**Vishnu Aware**
IT Engineering Student

Project developed for **TE Connectivity AI Cup**.

````
