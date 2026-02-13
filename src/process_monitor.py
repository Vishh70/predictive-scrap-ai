from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd


PROCESS_RULES: Dict[str, Dict[str, Any]] = {
    "cushion": {
        "threshold": 0.5,
        "level": "CRITICAL",
        "message": "Check Part Filling/Quality",
    },
    "injection_time": {
        "threshold": 0.03,
        "level": "CRITICAL",
        "message": "Process Instability Detected",
    },
    "dosage_time": {
        "threshold": 1.0,
        "level": "CRITICAL",
        "message": "Check Material Feeding/Dosing Speed",
    },
    "injection_pressure": {
        "threshold": 100.0,
        "level": "WARNING",
        "message": "Injection pressure drift detected.",
    },
    "switch_pressure": {
        "threshold": 100.0,
        "level": "WARNING",
        "message": "Switch pressure drift detected.",
    },
    "cyl_tmp_z": {
        "threshold": 5.0,
        "level": "WARNING",
        "message": "Cylinder temperature drift detected.",
    },
    "switch_position": {
        "threshold": 0.05,
        "level": "WARNING",
        "message": "Switch-over position drift detected.",
    },
}

IGNORED_PARAMS = {
    "extruder_torque",
    "peak_pressure_time",
    "peak_pressure_position",
    "ejector_fix_deviation_torque",
}

PARAM_ALIASES = {
    "plasticizing_time": "dosage_time",
    "switch_over_volume": "switch_position",
}

FALLBACK_REMARKS = {
    "cushion": "Resulting value from Switch position and holding pressure.",
    "injection_time": "Resulting value when setting the injection speed (mm/s).",
    "dosage_time": "Resulting value when setting the dosing speed (rpm).",
    "plasticizing_time": "Resulting value when setting the dosing speed (rpm).",
    "injection_pressure": "Resulting value when setting the injection speed (mm/s).",
    "switch_pressure": "Resulting value when setting the injection speed (mm/s).",
    "cyl_tmp_z": "Machine regulates each heating zone to set value; small deviations are expected.",
    "switch_position": "Setting when injection stops and process switches to holding pressure.",
    "switch_over_volume": "Setting when injection stops and process switches to holding pressure.",
}


def _normalize_param(name: Any) -> str:
    text = str(name).strip().lower()
    if not text:
        return ""
    cleaned = (
        text.replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(".", "")
    )
    if cleaned.startswith("cyl_tmp_z"):
        return cleaned
    return cleaned


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip().replace(",", ".")
            if value == "":
                return None
        numeric = float(value)
        if pd.isna(numeric):
            return None
        return numeric
    except Exception:
        return None


def _rule_key_for_param(param_name: str) -> Optional[str]:
    if not param_name:
        return None
    if param_name in IGNORED_PARAMS:
        return None
    if param_name.startswith("cyl_tmp_z"):
        return "cyl_tmp_z"
    if param_name in PROCESS_RULES:
        return param_name
    return PARAM_ALIASES.get(param_name)


def _candidate_keys(observed_param: str, rule_key: str) -> Iterable[str]:
    keys = [observed_param]
    if rule_key not in keys:
        keys.append(rule_key)
    if rule_key == "dosage_time":
        keys.append("plasticizing_time")
    if rule_key == "switch_position":
        keys.append("switch_over_volume")
    if rule_key == "cyl_tmp_z":
        keys.append("cyl_tmp_z")
    deduped = []
    seen = set()
    for key in keys:
        if key and key not in seen:
            deduped.append(key)
            seen.add(key)
    return deduped


def _resolve_baseline(
    observed_param: str,
    rule_key: str,
    row_values: Dict[str, Any],
    target_values: Dict[str, Any],
    moving_avg_values: Dict[str, Any],
) -> Optional[float]:
    keys = list(_candidate_keys(observed_param, rule_key))

    for key in keys:
        value = _to_float(target_values.get(key))
        if value is not None:
            return value

    for key in keys:
        value = _to_float(moving_avg_values.get(key))
        if value is not None:
            return value

    for key in keys:
        for suffix in ("_target", "_ma", "_rolling_mean"):
            value = _to_float(row_values.get(f"{key}{suffix}"))
            if value is not None:
                return value

    return None


def _resolve_tip(observed_param: str, rule_key: str, remarks_map: Dict[str, str]) -> str:
    for key in _candidate_keys(observed_param, rule_key):
        tip = remarks_map.get(key)
        if tip:
            return tip
    return ""


def _evaluate_single_parameter(
    observed_param: str,
    observed_value: Any,
    row_values: Dict[str, Any],
    target_values: Dict[str, Any],
    moving_avg_values: Dict[str, Any],
    remarks_map: Dict[str, str],
) -> Optional[Dict[str, Any]]:
    rule_key = _rule_key_for_param(observed_param)
    if rule_key is None:
        return None

    value = _to_float(observed_value)
    if value is None:
        return None

    baseline = _resolve_baseline(
        observed_param=observed_param,
        rule_key=rule_key,
        row_values=row_values,
        target_values=target_values,
        moving_avg_values=moving_avg_values,
    )
    if baseline is None:
        return None

    rule = PROCESS_RULES[rule_key]
    threshold = float(rule["threshold"])
    deviation = abs(value - baseline)
    if deviation <= threshold:
        return None

    tip = _resolve_tip(observed_param, rule_key, remarks_map)
    message = str(rule["message"])
    error_msg = (
        f"{rule['level']}: {observed_param} deviation {deviation:.2f} "
        f"exceeds +/-{threshold:.2f}. {message}"
    )
    if tip:
        error_msg = f"{error_msg} Tip: {tip}"

    return {
        "level": rule["level"],
        "parameter": observed_param,
        "value": value,
        "target": baseline,
        "deviation": deviation,
        "threshold": threshold,
        "message": message,
        "tip": tip,
        "error": error_msg,
    }


@lru_cache(maxsize=8)
def _load_parameter_remarks_cached(data_dir_str: str) -> Dict[str, str]:
    remarks = dict(FALLBACK_REMARKS)
    manual_path = Path(data_dir_str) / "AI_cup_parameter_info.xlsx"
    if not manual_path.exists():
        return remarks

    try:
        manual_df = pd.read_excel(manual_path)
        if manual_df.empty:
            return remarks

        if "variable_name" not in manual_df.columns or "Remark" not in manual_df.columns:
            return remarks

        for _, row in manual_df[["variable_name", "Remark"]].dropna(subset=["variable_name"]).iterrows():
            key = _normalize_param(row["variable_name"])
            if key.startswith("cyl_tmp_z"):
                key = "cyl_tmp_z"
            tip_text = str(row.get("Remark", "")).strip()
            if key and tip_text:
                remarks[key] = tip_text
    except Exception:
        return remarks

    return remarks


def load_parameter_remarks(data_dir: Optional[Path] = None) -> Dict[str, str]:
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent / "data"
    return dict(_load_parameter_remarks_cached(str(Path(data_dir).resolve())))


def check_process_status(
    data_row: dict | pd.Series,
    target_values: Optional[Dict[str, float]] = None,
    moving_avg_values: Optional[Dict[str, float]] = None,
    remarks_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    row_dict = data_row.to_dict() if isinstance(data_row, pd.Series) else dict(data_row)
    normalized_row = {_normalize_param(k): v for k, v in row_dict.items()}

    target_values = {_normalize_param(k): v for k, v in (target_values or {}).items()}
    moving_avg_values = {_normalize_param(k): v for k, v in (moving_avg_values or {}).items()}
    remarks_map = {_normalize_param(k): str(v) for k, v in (remarks_map or load_parameter_remarks()).items()}

    alerts = []

    is_long_form = (
        "variable_name" in normalized_row
        and "value" in normalized_row
        and str(normalized_row.get("variable_name", "")).strip() != ""
    )

    if is_long_form:
        observed_param = _normalize_param(normalized_row["variable_name"])
        if observed_param in IGNORED_PARAMS:
            return {"status": "OK", "errors": [], "parameter": observed_param, "alerts": []}
        alert = _evaluate_single_parameter(
            observed_param=observed_param,
            observed_value=normalized_row.get("value"),
            row_values=normalized_row,
            target_values=target_values,
            moving_avg_values=moving_avg_values,
            remarks_map=remarks_map,
        )
        if alert:
            alerts.append(alert)
    else:
        for observed_param, observed_value in normalized_row.items():
            if observed_param in IGNORED_PARAMS:
                continue
            alert = _evaluate_single_parameter(
                observed_param=observed_param,
                observed_value=observed_value,
                row_values=normalized_row,
                target_values=target_values,
                moving_avg_values=moving_avg_values,
                remarks_map=remarks_map,
            )
            if alert:
                alerts.append(alert)

    errors = [a["error"] for a in alerts]
    has_critical = any(a["level"] == "CRITICAL" for a in alerts)
    has_warning = any(a["level"] == "WARNING" for a in alerts)

    if has_critical:
        status = "CRITICAL"
    elif has_warning:
        status = "WARNING"
    else:
        status = "OK"

    if not alerts:
        parameter = "none"
    elif len(alerts) == 1:
        parameter = alerts[0]["parameter"]
    else:
        parameter = "multiple"

    return {
        "status": status,
        "errors": errors,
        "parameter": parameter,
        "alerts": [
            {
                "level": a["level"],
                "parameter": a["parameter"],
                "value": a["value"],
                "target": a["target"],
                "deviation": a["deviation"],
                "threshold": a["threshold"],
                "message": a["message"],
                "tip": a["tip"],
            }
            for a in alerts
        ],
    }
