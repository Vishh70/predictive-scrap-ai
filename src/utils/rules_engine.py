from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.utils.logger import get_logger


logger = get_logger(__name__)


def _normalize_param_name(raw_name: str) -> tuple[str, bool]:
    text = str(raw_name).strip().lower()
    cleaned = (
        text.replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(".", "")
    )

    if "cyl_tmp_z" in cleaned and ("1x" in cleaned or "*" in cleaned):
        return "cyl_tmp_z", True
    if "cyl_tmp_z" in cleaned and "1_x" in cleaned:
        return "cyl_tmp_z", True

    return cleaned, False


def _parse_threshold(raw_threshold: Any) -> float | None:
    text = str(raw_threshold).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    if not any(ch.isdigit() for ch in text):
        return None

    match = re.search(r"(\d+(?:[.,]\d+)?)", text)
    if not match:
        return None

    try:
        return float(match.group(1).replace(",", "."))
    except ValueError:
        return None


def load_physics_rules(csv_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Parse AI Cup threshold sheet exported as CSV and return normalized rules.
    Handles comma decimals and wildcard cylinder-zone rule parsing.
    """
    rules: Dict[str, Dict[str, Any]] = {}

    if not csv_path.exists():
        logger.warning("Rules file not found at %s. Returning empty rules.", csv_path)
        return rules

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        logger.error("Failed to read rules CSV at %s: %s", csv_path, exc)
        return rules

    df.columns = [str(c).lower().strip() for c in df.columns]
    target_col = next((c for c in df.columns if "threshold" in c), None)
    param_col = next((c for c in df.columns if "variable" in c), None)

    if not target_col or not param_col:
        logger.warning("Could not find threshold/variable columns in %s.", csv_path)
        return rules

    for _, row in df.iterrows():
        param_raw = str(row.get(param_col, "")).strip()
        threshold_raw = row.get(target_col, "")

        threshold = _parse_threshold(threshold_raw)
        if threshold is None:
            continue

        param_key, is_wildcard = _normalize_param_name(param_raw)
        if not param_key:
            continue

        rules[param_key] = {
            "threshold": threshold,
            "original_text": str(threshold_raw),
            "is_wildcard": is_wildcard,
        }

    logger.info("Successfully parsed %d physics rules from %s", len(rules), csv_path)
    return rules

