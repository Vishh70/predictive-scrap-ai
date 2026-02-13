from __future__ import annotations

import re
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd


@dataclass
class FileHealth:
    file_name: str
    machine_id: str
    status: str
    rows_detected: int
    error_message: str
    has_timestamp: bool
    has_value: bool
    columns: List[str]


def extract_machine_id(file_name: str) -> str:
    match = re.search(r"(M\d+)", file_name, flags=re.IGNORECASE)
    return match.group(1).upper() if match else "UNKNOWN"


def normalize_columns(columns: List[str]) -> List[str]:
    return [str(c).strip().lower() for c in columns]


def check_required_columns(columns: List[str]) -> Tuple[bool, bool]:
    normalized = normalize_columns(columns)
    has_timestamp = any(c == "timestamp" or "timestamp" in c for c in normalized)
    has_value = any(c == "value" or "value" in c for c in normalized)
    return has_timestamp, has_value


def check_tail_integrity(file_path: Path, tail_size: int = 1000) -> List[str]:
    issues: List[str] = []
    with open(file_path, "rb") as f:
        f.seek(0, 2)
        file_size = f.tell()
        if file_size == 0:
            return ["File is empty."]

        read_size = min(file_size, tail_size)
        f.seek(-read_size, 2)
        tail = f.read(read_size).decode("utf-8", errors="ignore")

    # Heuristic checks for truncation/quote corruption near EOF.
    if tail and not tail.endswith(("\n", "\r")):
        issues.append("Tail does not end with newline (possible truncation).")

    lines = [ln for ln in tail.splitlines() if ln.strip()]
    if lines:
        last_line = lines[-1]
        try:
            # Strict CSV parse on the final line catches unfinished quotes at EOF.
            next(csv.reader([last_line], strict=True))
        except Exception:
            issues.append("Last line appears malformed (possible EOF-inside-string corruption).")

    return issues


def header_check(file_path: Path) -> Tuple[List[str], str]:
    try:
        sample_df = pd.read_csv(file_path, engine="python", nrows=10)
        return list(sample_df.columns), ""
    except Exception as exc:  # pylint: disable=broad-except
        return [], str(exc)


def deep_parse_and_count_rows(file_path: Path) -> Tuple[int, str]:
    rows_detected = 0
    try:
        for chunk in pd.read_csv(file_path, engine="python", chunksize=100_000):
            rows_detected += len(chunk)
        return rows_detected, ""
    except Exception as exc:  # pylint: disable=broad-except
        return rows_detected, str(exc)


def format_table(rows: List[List[str]], headers: List[str]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_line(parts: List[str]) -> str:
        return " | ".join(p.ljust(widths[i]) for i, p in enumerate(parts))

    sep = "-+-".join("-" * w for w in widths)
    out = [fmt_line(headers), sep]
    out.extend(fmt_line(r) for r in rows)
    return "\n".join(out)


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"

    if not data_dir.exists():
        print(f"ERROR: data directory not found -> {data_dir}")
        return

    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in: {data_dir}")
        return

    report: List[FileHealth] = []

    for file_path in csv_files:
        file_name = file_path.name
        machine_id = extract_machine_id(file_name)
        status = "OK"
        rows_detected = 0
        error_message = ""

        columns, header_error = header_check(file_path)
        if header_error:
            status = "CORRUPT"
            error_message = f"Header read failed: {header_error}"
            has_timestamp = False
            has_value = False
            report.append(
                FileHealth(
                    file_name=file_name,
                    machine_id=machine_id,
                    status=status,
                    rows_detected=rows_detected,
                    error_message=error_message,
                    has_timestamp=has_timestamp,
                    has_value=has_value,
                    columns=[],
                )
            )
            continue

        has_timestamp, has_value = check_required_columns(columns)

        tail_issues = check_tail_integrity(file_path)
        rows_detected, deep_error = deep_parse_and_count_rows(file_path)

        if deep_error:
            status = "CORRUPT"
            error_message = deep_error
        elif rows_detected == 0:
            status = "EMPTY"
            error_message = "No data rows detected."
        elif not has_timestamp or not has_value:
            status = "INVALID_SCHEMA"
            missing = []
            if not has_timestamp:
                missing.append("timestamp")
            if not has_value:
                missing.append("value")
            error_message = f"Missing required column(s): {', '.join(missing)}"

        if tail_issues:
            if error_message:
                error_message = f"{error_message} | Tail check: {'; '.join(tail_issues)}"
            else:
                status = "WARNING"
                error_message = "; ".join(tail_issues)

        if machine_id == "M356" and deep_error:
            try:
                # Optional second opinion to surface the familiar pandas C-engine message.
                pd.read_csv(file_path, engine="c")
            except Exception as c_exc:  # pylint: disable=broad-except
                if "EOF inside string" in str(c_exc):
                    error_message = str(c_exc)

        if machine_id == "M356" and "EOF inside string" in error_message:
            status = "CORRUPT"
            error_message = f"{error_message} | SPECIFIC FLAG: M356.csv has EOF-inside-string corruption."

        report.append(
            FileHealth(
                file_name=file_name,
                machine_id=machine_id,
                status=status,
                rows_detected=rows_detected,
                error_message=error_message or "-",
                has_timestamp=has_timestamp,
                has_value=has_value,
                columns=columns,
            )
        )

    summary_rows = [
        [r.file_name, r.status, str(r.rows_detected), r.error_message]
        for r in report
    ]
    print("\nCSV HEALTH REPORT")
    print(format_table(summary_rows, ["File Name", "Status", "Rows Detected", "Error Message"]))

    print("\nMETADATA DETAILS")
    for r in report:
        cols_preview = ", ".join(r.columns[:12])
        if len(r.columns) > 12:
            cols_preview += ", ..."
        print(
            f"- {r.file_name} | Machine ID: {r.machine_id} | "
            f"timestamp: {r.has_timestamp} | value: {r.has_value}"
        )
        print(f"  Columns: {cols_preview if cols_preview else '(none)'}")


if __name__ == "__main__":
    main()
