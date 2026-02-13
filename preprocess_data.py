import glob
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:
    raise SystemExit(
        "pyarrow is required for Parquet export. Install it with: pip install pyarrow"
    ) from exc


# CONFIGURATION
DATA_FOLDER = "data"
OUTPUT_FILE = "data/optimized_scrap_data.parquet"
CHUNK_SIZE = 500_000

# Optional: set required columns if you want to force a fixed schema.
# REQUIRED_COLS = ["timestamp", "actual_cycle_time", "injection_pressure", "cushion_position", "is_scrap"]
REQUIRED_COLS: List[str] = []


def optimize_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Apply memory optimizations on each chunk."""
    if REQUIRED_COLS:
        for col in REQUIRED_COLS:
            if col not in chunk.columns:
                chunk[col] = pd.NA
        chunk = chunk[REQUIRED_COLS]

    for col in chunk.select_dtypes(include=["object"]).columns:
        num_unique_values = chunk[col].nunique(dropna=False)
        num_total_values = len(chunk[col])
        if num_total_values > 0 and (num_unique_values / num_total_values) < 0.5:
            chunk[col] = chunk[col].astype("category")

    for col in chunk.select_dtypes(include=["float64"]).columns:
        chunk[col] = chunk[col].astype("float32")

    for col in chunk.select_dtypes(include=["int64"]).columns:
        chunk[col] = chunk[col].astype("Int32")

    return chunk


def align_to_schema(
    chunk: pd.DataFrame,
    schema_cols: List[str],
    schema_dtypes: Dict[str, str],
) -> pd.DataFrame:
    """Force chunk to the same column order and dtypes as first valid chunk."""
    missing_cols = [c for c in schema_cols if c not in chunk.columns]
    for col in missing_cols:
        chunk[col] = pd.NA

    chunk = chunk[schema_cols]

    for col, dtype_name in schema_dtypes.items():
        if col not in chunk.columns:
            continue
        if dtype_name.startswith("float"):
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce").astype(dtype_name)
        elif dtype_name.lower().startswith("int"):
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce").astype("Int32")
        elif dtype_name == "category":
            chunk[col] = chunk[col].astype("category")
        else:
            chunk[col] = chunk[col].astype(dtype_name, errors="ignore")

    return chunk


def to_arrow_safe(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Convert category columns to string before Arrow conversion.
    This keeps Parquet schema stable across chunks/files.
    """
    cat_cols = chunk.select_dtypes(include=["category"]).columns
    if len(cat_cols) > 0:
        chunk = chunk.copy()
        chunk[cat_cols] = chunk[cat_cols].astype("string")
    return chunk


def main() -> None:
    files = sorted(glob.glob(os.path.join(DATA_FOLDER, "*.csv")))
    print(f"Found {len(files)} files: {[os.path.basename(f) for f in files]}")

    if not files:
        raise SystemExit(
            f"No CSV files found in '{DATA_FOLDER}'. Put raw CSV files there first."
        )

    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    writer: Optional[pq.ParquetWriter] = None
    arrow_schema: Optional[pa.Schema] = None
    schema_cols: Optional[List[str]] = None
    schema_dtypes: Optional[Dict[str, str]] = None
    total_rows = 0
    total_chunks = 0

    try:
        for file in files:
            file_name = os.path.basename(file)
            print(f"Processing {file_name}...")
            chunk_iter = pd.read_csv(file, chunksize=CHUNK_SIZE, low_memory=False)

            for i, chunk in enumerate(chunk_iter, start=1):
                chunk = optimize_chunk(chunk)

                if schema_cols is None:
                    schema_cols = chunk.columns.tolist()
                    schema_dtypes = {c: str(chunk[c].dtype) for c in schema_cols}
                else:
                    chunk = align_to_schema(chunk, schema_cols, schema_dtypes or {})

                chunk_for_write = to_arrow_safe(chunk)
                table = pa.Table.from_pandas(chunk_for_write, preserve_index=False)

                if writer is None:
                    arrow_schema = table.schema
                    writer = pq.ParquetWriter(
                        output_path.as_posix(),
                        arrow_schema,
                        compression="snappy",
                    )
                elif arrow_schema is not None and not table.schema.equals(
                    arrow_schema, check_metadata=False
                ):
                    table = table.cast(arrow_schema, safe=False)

                writer.write_table(table)
                total_rows += len(chunk)
                total_chunks += 1
                print(
                    f"   -> chunk {i} done | rows so far: {total_rows:,} | total chunks: {total_chunks:,}"
                )
    finally:
        if writer is not None:
            writer.close()

    print(f"Saved optimized dataset: {OUTPUT_FILE}")
    print(f"Total rows written: {total_rows:,}")
    print("SUCCESS! Point Streamlit/pipeline to this Parquet file for faster loads.")


if __name__ == "__main__":
    main()
