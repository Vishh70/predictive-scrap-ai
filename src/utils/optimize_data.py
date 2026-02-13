import gc
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed_full_dataset.pkl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "demo_ready_data.parquet"
RESAMPLE_FREQ = "1min"


def _downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    float64_cols = df.select_dtypes(include=["float64"]).columns
    if len(float64_cols) > 0:
        df[float64_cols] = df[float64_cols].astype(np.float32)

    int64_cols = df.select_dtypes(include=["int64"]).columns
    if len(int64_cols) > 0:
        df[int64_cols] = df[int64_cols].astype(np.int32)

    return df


def optimize_for_demo() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading giant file: {INPUT_PATH}")
    df = pd.read_pickle(INPUT_PATH)
    print(f"Loaded shape: {df.shape}")

    required_cols: List[str] = ["machine_id", "timestamp"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required columns {missing} in {INPUT_PATH.name}. "
            "Cannot build demo parquet without machine_id and timestamp."
        )

    df = _downcast_numeric(df)

    print("Resampling to 1-minute intervals...")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["machine_id", "timestamp"]).copy()
    df["machine_id"] = df["machine_id"].astype(str)

    context_cols = [
        c for c in [
            "plant_shift_date",
            "machine_status_name",
            "segment_abbr_name",
            "manufacturing_plant_name",
        ] if c in df.columns
    ]

    pieces: List[pd.DataFrame] = []
    for machine_id, grp in df.groupby("machine_id", sort=False):
        grp = grp.sort_values("timestamp")

        resampled = (
            grp.set_index("timestamp")
            .resample(RESAMPLE_FREQ)
            .mean(numeric_only=True)
            .reset_index()
        )
        resampled["machine_id"] = machine_id

        if "is_scrap" in grp.columns:
            scrap_1min = (
                grp.set_index("timestamp")["is_scrap"]
                .resample(RESAMPLE_FREQ)
                .max()
            )
            resampled["is_scrap"] = scrap_1min.values

        if context_cols:
            ctx = (
                grp.set_index("timestamp")[context_cols]
                .resample(RESAMPLE_FREQ)
                .last()
                .ffill()
                .bfill()
                .reset_index(drop=True)
            )
            for col in context_cols:
                resampled[col] = ctx[col].values

        pieces.append(resampled)

    if pieces:
        demo_df = pd.concat(pieces, ignore_index=True)
    else:
        demo_df = df.iloc[0:0].copy()

    del pieces
    del df
    gc.collect()

    demo_df = _downcast_numeric(demo_df)
    print(f"Saving optimized file to: {OUTPUT_PATH}")
    demo_df.to_parquet(OUTPUT_PATH, index=False, compression="snappy")
    print(f"Done. Output shape: {demo_df.shape}")
    print("Optimization complete.")


if __name__ == "__main__":
    optimize_for_demo()
