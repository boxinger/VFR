from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
REFERENCE_OMEGA_RAD_S = 1.0


class ElementKind(StrEnum):
    REAL_ZERO = "real_zero"
    REAL_POLE = "real_pole"
    ORIGIN_ZERO = "origin_zero"
    ORIGIN_POLE = "origin_pole"
    COMPLEX_ZERO_PAIR = "complex_zero_pair"
    COMPLEX_POLE_PAIR = "complex_pole_pair"

    @property
    def display_name(self) -> str:
        return {
            ElementKind.REAL_ZERO: "实零点",
            ElementKind.REAL_POLE: "实极点",
            ElementKind.ORIGIN_ZERO: "原点零点",
            ElementKind.ORIGIN_POLE: "原点极点",
            ElementKind.COMPLEX_ZERO_PAIR: "复共轭零点对",
            ElementKind.COMPLEX_POLE_PAIR: "复共轭极点对",
        }[self]

    @property
    def needs_frequency(self) -> bool:
        return self not in {ElementKind.ORIGIN_ZERO, ElementKind.ORIGIN_POLE}


@dataclass(slots=True)
class FrequencyResponse:
    omega_rad_s: FloatArray
    response_complex: ComplexArray
    phase_deg: FloatArray

    def __post_init__(self) -> None:
        self.omega_rad_s = np.asarray(self.omega_rad_s, dtype=np.float64)
        self.response_complex = np.asarray(self.response_complex, dtype=np.complex128)
        self.phase_deg = np.asarray(self.phase_deg, dtype=np.float64)
        if self.omega_rad_s.ndim != 1:
            raise ValueError("频率数组必须是一维")
        if not (
            len(self.omega_rad_s)
            == len(self.response_complex)
            == len(self.phase_deg)
        ):
            raise ValueError("频率、复响应和相位长度必须一致")
        if len(self.omega_rad_s) < 2:
            raise ValueError("频率响应至少需要两个频点")
        if not np.all(np.isfinite(self.omega_rad_s)) or np.any(self.omega_rad_s <= 0):
            raise ValueError("频率必须是有限正数")
        if not np.all(np.isfinite(self.response_complex)):
            raise ValueError("复响应必须是有限数值")
        if not np.all(np.isfinite(self.phase_deg)):
            raise ValueError("相位必须是有限数值")
        if np.any(np.diff(self.omega_rad_s) <= 0):
            raise ValueError("频率必须严格递增且不能重复")

    @property
    def frequency_hz(self) -> FloatArray:
        return self.omega_rad_s / (2.0 * np.pi)

    @property
    def magnitude_linear(self) -> FloatArray:
        return np.abs(self.response_complex)

    @property
    def magnitude_db(self) -> FloatArray:
        with np.errstate(divide="ignore"):
            return 20.0 * np.log10(self.magnitude_linear)


@dataclass(slots=True)
class ControllerElement:
    kind: ElementKind
    omega_rad_s: float | None = None
    multiplicity: int = 1
    damping_ratio: float | None = None
    element_id: str = field(default_factory=lambda: uuid4().hex)

    def __post_init__(self) -> None:
        if self.multiplicity < 1:
            raise ValueError("重数必须是正整数")
        if self.kind.needs_frequency:
            if self.omega_rad_s is None or not np.isfinite(self.omega_rad_s):
                raise ValueError("有限零极点必须提供角频率")
            if self.omega_rad_s <= 0:
                raise ValueError("角频率必须为正数")
        if self.kind in {
            ElementKind.COMPLEX_ZERO_PAIR,
            ElementKind.COMPLEX_POLE_PAIR,
        }:
            if self.damping_ratio is None or not np.isfinite(self.damping_ratio):
                raise ValueError("复共轭对必须提供阻尼比")
            if self.damping_ratio <= 0:
                raise ValueError("阻尼比必须为正数")

    @property
    def frequency_hz(self) -> float | None:
        if self.omega_rad_s is None:
            return None
        return self.omega_rad_s / (2.0 * np.pi)

    def response(self, omega_rad_s: FloatArray) -> ComplexArray:
        omega = np.asarray(omega_rad_s, dtype=np.float64)
        s = 1j * omega
        if self.kind == ElementKind.REAL_ZERO:
            base = 1.0 + s / float(self.omega_rad_s)
        elif self.kind == ElementKind.REAL_POLE:
            base = 1.0 / (1.0 + s / float(self.omega_rad_s))
        elif self.kind == ElementKind.ORIGIN_ZERO:
            base = s / REFERENCE_OMEGA_RAD_S
        elif self.kind == ElementKind.ORIGIN_POLE:
            base = REFERENCE_OMEGA_RAD_S / s
        else:
            ratio = s / float(self.omega_rad_s)
            section = 1.0 + 2.0 * float(self.damping_ratio) * ratio + ratio**2
            if self.kind == ElementKind.COMPLEX_ZERO_PAIR:
                base = section
            else:
                base = 1.0 / section
        return np.asarray(base**self.multiplicity, dtype=np.complex128)

    def phase_deg(self, omega_rad_s: FloatArray) -> FloatArray:
        omega = np.asarray(omega_rad_s, dtype=np.float64)
        if self.kind == ElementKind.REAL_ZERO:
            return self.multiplicity * np.degrees(
                np.arctan(omega / float(self.omega_rad_s))
            )
        if self.kind == ElementKind.REAL_POLE:
            return -self.multiplicity * np.degrees(
                np.arctan(omega / float(self.omega_rad_s))
            )
        if self.kind == ElementKind.ORIGIN_ZERO:
            return np.full_like(omega, 90.0 * self.multiplicity)
        if self.kind == ElementKind.ORIGIN_POLE:
            return np.full_like(omega, -90.0 * self.multiplicity)
        phase = np.unwrap(np.angle(self.response(omega)))
        return np.degrees(phase)


@dataclass(slots=True)
class ControllerModel:
    gain_k: float = 1.0
    elements: list[ControllerElement] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.set_gain_k(self.gain_k)

    @property
    def gain_db(self) -> float:
        return 20.0 * float(np.log10(abs(self.gain_k)))

    def set_gain_k(self, gain_k: float) -> None:
        if not np.isfinite(gain_k) or gain_k == 0:
            raise ValueError("K 必须是非零有限数值")
        self.gain_k = float(gain_k)

    def set_gain_db(self, gain_db: float) -> None:
        if not np.isfinite(gain_db):
            raise ValueError("增益 dB 必须是有限数值")
        sign = -1.0 if self.gain_k < 0 else 1.0
        self.gain_k = sign * 10.0 ** (float(gain_db) / 20.0)

    def response(self, omega_rad_s: FloatArray) -> ComplexArray:
        omega = np.asarray(omega_rad_s, dtype=np.float64)
        result = np.full(omega.shape, complex(self.gain_k), dtype=np.complex128)
        for element in self.elements:
            result *= element.response(omega)
        return result

    def phase_deg(self, omega_rad_s: FloatArray) -> FloatArray:
        omega = np.asarray(omega_rad_s, dtype=np.float64)
        result = np.full_like(omega, -180.0 if self.gain_k < 0 else 0.0)
        for element in self.elements:
            result += element.phase_deg(omega)
        return result

    def magnitude_db(self, omega_rad_s: FloatArray) -> FloatArray:
        with np.errstate(divide="ignore"):
            return 20.0 * np.log10(np.abs(self.response(omega_rad_s)))

    def add_element(self, element: ControllerElement) -> None:
        self.elements.append(element)

    def remove_element(self, element_id: str) -> None:
        self.elements = [item for item in self.elements if item.element_id != element_id]

    def get_element(self, element_id: str) -> ControllerElement:
        for element in self.elements:
            if element.element_id == element_id:
                return element
        raise KeyError(element_id)
