from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .models import ControllerModel, FrequencyResponse


FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class GainCrossover:
    frequency_hz: float
    phase_deg: float
    phase_margin_deg: float


@dataclass(frozen=True, slots=True)
class PhaseCrossover:
    frequency_hz: float
    gain_db: float
    gain_margin_db: float
    phase_level_deg: float


@dataclass(slots=True)
class LoopMetrics:
    gain_crossovers: list[GainCrossover] = field(default_factory=list)
    phase_crossovers: list[PhaseCrossover] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)

    @property
    def primary_gain_crossover(self) -> GainCrossover | None:
        return self.gain_crossovers[0] if self.gain_crossovers else None

    @property
    def worst_phase_margin(self) -> GainCrossover | None:
        if not self.gain_crossovers:
            return None
        return min(self.gain_crossovers, key=lambda item: item.phase_margin_deg)

    @property
    def worst_gain_margin(self) -> PhaseCrossover | None:
        if not self.phase_crossovers:
            return None
        return min(self.phase_crossovers, key=lambda item: item.gain_margin_db)


@dataclass(slots=True)
class LoopAnalysis:
    controller_response: FrequencyResponse
    loop_response: FrequencyResponse
    metrics: LoopMetrics


def _interpolate_log_x(x1: float, x2: float, y1: float, y2: float, target: float) -> float:
    if y2 == y1:
        return x1
    log_x1 = np.log10(x1)
    log_x2 = np.log10(x2)
    fraction = (target - y1) / (y2 - y1)
    return float(10.0 ** (log_x1 + fraction * (log_x2 - log_x1)))


def _interpolate_y(frequency_hz: FloatArray, values: FloatArray, target_hz: float) -> float:
    log_frequency = np.log10(frequency_hz)
    return float(np.interp(np.log10(target_hz), log_frequency, values))


def _crossings(frequency_hz: FloatArray, values: FloatArray, target: float) -> list[float]:
    result: list[float] = []
    shifted = values - target
    for index in range(len(values) - 1):
        left = shifted[index]
        right = shifted[index + 1]
        if left == 0.0:
            result.append(float(frequency_hz[index]))
        if left * right < 0.0:
            result.append(
                _interpolate_log_x(
                    float(frequency_hz[index]),
                    float(frequency_hz[index + 1]),
                    float(values[index]),
                    float(values[index + 1]),
                    target,
                )
            )
    if shifted[-1] == 0.0:
        result.append(float(frequency_hz[-1]))
    deduplicated: list[float] = []
    for value in result:
        if not deduplicated or not np.isclose(value, deduplicated[-1], rtol=1e-12):
            deduplicated.append(value)
    return deduplicated


def analyze_loop(plant: FrequencyResponse, controller: ControllerModel) -> LoopAnalysis:
    omega = plant.omega_rad_s
    frequency_hz = plant.frequency_hz
    controller_complex = controller.response(omega)
    controller_phase = controller.phase_deg(omega)
    loop_complex = plant.response_complex * controller_complex
    loop_phase = plant.phase_deg + controller_phase

    controller_response = FrequencyResponse(omega, controller_complex, controller_phase)
    loop_response = FrequencyResponse(omega, loop_complex, loop_phase)
    metrics = LoopMetrics()

    for crossing_hz in _crossings(frequency_hz, loop_response.magnitude_db, 0.0):
        phase = _interpolate_y(frequency_hz, loop_phase, crossing_hz)
        metrics.gain_crossovers.append(
            GainCrossover(crossing_hz, phase, 180.0 + phase)
        )

    phase_min = float(np.min(loop_phase))
    phase_max = float(np.max(loop_phase))
    first_level = int(np.ceil((phase_min + 180.0) / 360.0))
    last_level = int(np.floor((phase_max + 180.0) / 360.0))
    for level_index in range(first_level, last_level + 1):
        phase_level = -180.0 + 360.0 * level_index
        for crossing_hz in _crossings(frequency_hz, loop_phase, phase_level):
            gain_db = _interpolate_y(
                frequency_hz, loop_response.magnitude_db, crossing_hz
            )
            metrics.phase_crossovers.append(
                PhaseCrossover(crossing_hz, gain_db, -gain_db, phase_level)
            )
    metrics.phase_crossovers.sort(key=lambda item: item.frequency_hz)

    if not metrics.gain_crossovers:
        metrics.messages.append("当前测量频率范围内未找到 0 dB 增益交越")
    elif len(metrics.gain_crossovers) > 1:
        metrics.messages.append(f"检测到 {len(metrics.gain_crossovers)} 个 0 dB 增益交越")
    if not metrics.phase_crossovers:
        metrics.messages.append("当前测量频率范围内未找到相位交越")
    return LoopAnalysis(controller_response, loop_response, metrics)
