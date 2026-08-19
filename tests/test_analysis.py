import numpy as np
import pytest

from vfr.analysis import analyze_loop
from vfr.models import ControllerModel, FrequencyResponse


def response_from_db_phase(frequency_hz, magnitude_db, phase_deg) -> FrequencyResponse:
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    magnitude_db = np.asarray(magnitude_db, dtype=float)
    phase_deg = np.asarray(phase_deg, dtype=float)
    complex_response = 10.0 ** (magnitude_db / 20.0) * np.exp(
        1j * np.radians(phase_deg)
    )
    return FrequencyResponse(2.0 * np.pi * frequency_hz, complex_response, phase_deg)


def test_exact_gain_crossover_and_phase_margin() -> None:
    plant = response_from_db_phase(
        [10.0, 100.0, 1000.0],
        [20.0, 0.0, -20.0],
        [-30.0, -60.0, -120.0],
    )
    analysis = analyze_loop(plant, ControllerModel())
    crossing = analysis.metrics.primary_gain_crossover
    assert crossing is not None
    assert crossing.frequency_hz == pytest.approx(100.0)
    assert crossing.phase_deg == pytest.approx(-60.0)
    assert crossing.phase_margin_deg == pytest.approx(120.0)


def test_log_frequency_interpolation() -> None:
    plant = response_from_db_phase(
        [10.0, 1000.0], [20.0, -20.0], [-30.0, -130.0]
    )
    crossing = analyze_loop(plant, ControllerModel()).metrics.primary_gain_crossover
    assert crossing is not None
    assert crossing.frequency_hz == pytest.approx(100.0)
    assert crossing.phase_deg == pytest.approx(-80.0)


def test_multiple_and_missing_crossovers_are_reported() -> None:
    multiple = response_from_db_phase(
        [10.0, 100.0, 1000.0, 10000.0],
        [-10.0, 10.0, -10.0, 10.0],
        [-20.0, -80.0, -140.0, -200.0],
    )
    metrics = analyze_loop(multiple, ControllerModel()).metrics
    assert len(metrics.gain_crossovers) == 3
    assert any("3 个" in message for message in metrics.messages)

    no_crossing = response_from_db_phase(
        [10.0, 100.0], [10.0, 5.0], [-20.0, -30.0]
    )
    metrics = analyze_loop(no_crossing, ControllerModel()).metrics
    assert metrics.primary_gain_crossover is None
    assert any("未找到 0 dB" in message for message in metrics.messages)
