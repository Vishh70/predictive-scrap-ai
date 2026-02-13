from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Tuple


def _is_row_valid(row: List[str], expected_columns: int | None) -> bool:
    if expected_columns is None:
        return True
    return len(row) == expected_columns


def repair_csv_file(file_path: Path) -> Tuple[int, int]:
    """
    Repairs a CSV file by removing malformed lines.

    A malformed line is one that:
    - raises csv.Error (e.g., unclosed quotes), or
    - has a different number of parsed columns than the header row.
    """
    repaired_rows: List[List[str]] = []
    expected_columns: int | None = None
    bad_lines = 0

    with file_path.open("r", encoding="utf-8", errors="replace", newline="") as src:
        for raw_line in src:
            if not raw_line.strip():
                continue

            try:
                parsed = next(csv.reader([raw_line], strict=True))
            except (csv.Error, StopIteration):
                bad_lines += 1
                continue

            if not _is_row_valid(parsed, expected_columns):
                bad_lines += 1
                continue

            if expected_columns is None:
                expected_columns = len(parsed)

            repaired_rows.append(parsed)

    clean_path = file_path.with_name(f"{file_path.stem}_clean{file_path.suffix}")
    with clean_path.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.writer(dst)
        writer.writerows(repaired_rows)

    clean_path.replace(file_path)
    return len(repaired_rows), bad_lines


def repair_all_csv_files(data_dir: Path) -> None:
    csv_files = sorted(data_dir.rglob("*.csv"))
    for csv_file in csv_files:
        _, removed = repair_csv_file(csv_file)
        print(f"✅ Repaired {csv_file.name}: Removed {removed} bad lines.")


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    repair_all_csv_files(data_dir)


if __name__ == "__main__":
    main()
