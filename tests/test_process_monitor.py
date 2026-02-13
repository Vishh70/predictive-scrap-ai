import pytest

from src.monitoring_utils import ProcessMonitor


monitor = ProcessMonitor()


def test_critical_alert_cushion():
    """Test CRITICAL alert when Cushion deviates > 0.5mm."""
    row = {"variable_name": "Cushion", "value": 5.6}
    targets = {"Cushion": 5.0}

    result = monitor.check_safety(row, targets)

    assert result["status"] == "CRITICAL"
    assert "Check Part Filling" in result["message"]
    assert result["deviation"] == pytest.approx(0.6)


def test_warning_alert_pressure():
    """Test WARNING alert when Pressure deviates > 100 bar."""
    row = {"variable_name": "Injection_pressure", "value": 1150}
    targets = {"Injection_pressure": 1000}

    result = monitor.check_safety(row, targets)

    assert result["status"] == "WARNING"
    assert "Deviation" in result["message"]


def test_wildcard_temperature():
    """Test wildcard logic for Cyl_tmp_z*."""
    row = {"variable_name": "Cyl_tmp_z1", "value": 210}
    targets = {"Cyl_tmp_z1": 200}

    result = monitor.check_safety(row, targets)

    assert result["status"] == "WARNING"
    assert "Heating Zone" in result["message"]


def test_ignored_parameter():
    """Test that Extruder_torque is explicitly ignored."""
    row = {"variable_name": "Extruder_torque", "value": 5000}
    targets = {"Extruder_torque": 100}

    result = monitor.check_safety(row, targets)

    assert result["status"] == "SKIPPED"


def test_normal_operation():
    """Test that small deviations return OK."""
    row = {"variable_name": "Cushion", "value": 5.1}
    targets = {"Cushion": 5.0}

    result = monitor.check_safety(row, targets)

    assert result["status"] == "OK"
