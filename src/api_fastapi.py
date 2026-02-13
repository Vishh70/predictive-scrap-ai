# src/api_fastapi.py
import torch
import torch.nn as nn
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from collections import deque
import logging
import random

# -----------------------
# Logging
# -----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("scrap_api")

# -----------------------
# Paths & defaults
# -----------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "data" / "models" / "lstm_scrap_model.pth"
SCALER_PATH = BASE_DIR / "data" / "models" / "deep_scaler.joblib"

SEQUENCE_LENGTH = 30

# -----------------------
# LSTM Model
# -----------------------
class ScrapLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.sigmoid(self.fc(out))

# -----------------------
# FastAPI app
# -----------------------
app = FastAPI(
    title="Scrap Prediction API",
    description="LSTM-based scrap prediction with batch and streaming support",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Load scaler and infer INPUT_DIM
# -----------------------
try:
    scaler = joblib.load(SCALER_PATH)
    INPUT_DIM = scaler.n_features_in_
    logger.info(f"Scaler loaded. INPUT_DIM={INPUT_DIM}")
except Exception as e:
    raise RuntimeError(f"Failed to load scaler: {e}")

# -----------------------
# Load model
# -----------------------
model = ScrapLSTM(INPUT_DIM)
try:
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    logger.info("Model loaded successfully")
except Exception as e:
    raise RuntimeError(
        f"Model loading failed: {e}\n"
        f"Check INPUT_DIM={INPUT_DIM} matches training."
    )

# -----------------------
# Sliding window (streaming)
# -----------------------
_sliding_window = deque(maxlen=SEQUENCE_LENGTH)

# -----------------------
# Schemas
# -----------------------
class MachineData(BaseModel):
    values: List[List[float]] = Field(
        example=[[0.1] * INPUT_DIM] * SEQUENCE_LENGTH
    )

class SingleRow(BaseModel):
    row: List[float] = Field(
        example=[0.1] * INPUT_DIM
    )

# -----------------------
# Routes
# -----------------------
@app.get("/")
def root():
    return {"message": "Scrap Prediction API running", "docs": "/docs"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "input_dim": INPUT_DIM,
        "sequence_length": SEQUENCE_LENGTH
    }

@app.get("/sample")
def sample():
    """Return a valid payload for /predict"""
    return {
        "values": [
            [round(random.random(), 4) for _ in range(INPUT_DIM)]
            for _ in range(SEQUENCE_LENGTH)
        ]
    }

# -----------------------
# Batch prediction
# -----------------------
@app.post("/predict")
def predict_scrap(data: MachineData):
    X = np.array(data.values, dtype=float)

    if X.shape != (SEQUENCE_LENGTH, INPUT_DIM):
        raise HTTPException(
            status_code=400,
            detail=f"Expected shape ({SEQUENCE_LENGTH}, {INPUT_DIM}), got {X.shape}"
        )

    try:
        X_scaled = scaler.transform(X)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scaling failed: {e}")

    X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0)

    with torch.no_grad():
        prob = float(model(X_tensor).item())

    return {
        "scrap_probability": round(prob, 4),
        "risk_level": "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.4 else "LOW"
    }

# -----------------------
# Streaming prediction
# -----------------------
@app.post("/predict/row")
def predict_row_stream(payload: SingleRow):
    row = payload.row

    if len(row) != INPUT_DIM:
        raise HTTPException(
            status_code=400,
            detail=f"Row length must be {INPUT_DIM}, got {len(row)}"
        )

    _sliding_window.append(np.array(row, dtype=float))

    if len(_sliding_window) < SEQUENCE_LENGTH:
        return {
            "status": "window_filling",
            "current_size": len(_sliding_window),
            "required": SEQUENCE_LENGTH
        }

    X = np.vstack(_sliding_window)

    try:
        X_scaled = scaler.transform(X)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scaling failed: {e}")

    X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0)

    with torch.no_grad():
        prob = float(model(X_tensor).item())

    return {
        "scrap_probability": round(prob, 4),
        "risk_level": "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.4 else "LOW",
        "window_size": len(_sliding_window)
    }

# -----------------------
# Reset streaming window
# -----------------------
@app.post("/predict/reset")
def reset_stream():
    _sliding_window.clear()
    return {"status": "reset_done"}

# -----------------------
# Run notes
# -----------------------
# uvicorn src.api_fastapi:app --reload
# Open http://127.0.0.1:8000/docs
