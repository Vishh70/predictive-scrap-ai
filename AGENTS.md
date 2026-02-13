# AGENTS.md

## Project overview
Predictive Scrap AI is an ML project that predicts manufacturing scrap from process data. It provides an offline training pipeline and an online FastAPI service that serves predictions from saved model artifacts.

## Repo layout (key paths)
- run_pipeline.py (full pipeline orchestrator)
- src/ (app code, training, API)
- data/ (datasets and model artifacts)
- data/models/ (saved models and scalers)
- reports/ (generated reports)
- logs/ (pipeline logs)

## Setup
- Python: 3.10+
- Create venv: `python -m venv .venv`
- Activate (PowerShell): `.venv\Scripts\Activate.ps1`
- Install deps: `pip install -r requirements.txt`

## Main commands
- Full pipeline: `python run_pipeline.py`
- Deep model training: `python src/train_deep_model.py`
- API start: `uvicorn src.api_fastapi:app --reload`

## Artifacts and outputs
- XGBoost artifacts: `data/models/*.joblib`
- LSTM artifacts: `data/models/lstm_scrap_model.pth`, `data/models/deep_scaler.joblib`
- Reports: `reports/`
- Logs: `logs/`

## Data expectations
- Input CSVs live in `data/` and should match patterns defined in `src/config.py`.
- Datasets are not committed (see README).

## Testing
- No automated tests are currently present in this repo.

## Safety notes
- Keep files ASCII unless a non-ASCII character is required.
- Do not run long training inside API startup; API should only load saved artifacts.
