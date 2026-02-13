from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.utils.rules_engine import load_physics_rules


def resolve_source_csv(data_dir: Path) -> Path | None:
    """Find or create a CSV source from AI Cup parameter sheet."""
    preferred_csv = data_dir / "AI_cup_parameter_info.xlsx - Sheet1.csv"
    if preferred_csv.exists():
        return preferred_csv

    # Fallback: any similarly named CSV already present.
    for csv_file in sorted(data_dir.glob("*AI_cup_parameter_info*.csv")):
        return csv_file

    # Fallback: convert XLSX to CSV.
    xlsx_candidates = sorted(data_dir.glob("*AI_cup_parameter_info*.xlsx"))
    if not xlsx_candidates:
        return None

    xlsx_path = xlsx_candidates[0]
    try:
        df = pd.read_excel(xlsx_path, sheet_name=0)
        df.to_csv(preferred_csv, index=False)
        print(f"🧾 Converted XLSX to CSV: {preferred_csv}")
        return preferred_csv
    except Exception as exc:
        print(f"❌ Failed to convert XLSX to CSV: {exc}")
        return None


def main() -> None:
    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "data"
    output_config = data_dir / "monitoring_config.json"

    source_csv = resolve_source_csv(data_dir)
    if source_csv is None:
        print("❌ Error: Could not find AI Cup parameter source file in data/.")
        print("👉 Place AI_cup_parameter_info.xlsx (or CSV export) in the data/ folder.")
        return

    print(f"📂 Found rules source: {source_csv}")
    rules = load_physics_rules(source_csv)
    if not rules:
        print("⚠️ No rules extracted. Check source file headers and threshold values.")
        return

    output_config.parent.mkdir(parents=True, exist_ok=True)
    with open(output_config, "w", encoding="utf-8") as f:
        json.dump(rules, f, indent=4)

    print(f"✅ Success! Generated {output_config}")
    print(f"📊 Extracted {len(rules)} safety rules.")


if __name__ == "__main__":
    main()

