import numpy as np
import pytest

from vfr.models import ControllerElement, ControllerModel, ElementKind


def test_real_zero_is_normalized_and_has_expected_corner_response() -> None:
    omega_corner = 2.0 * np.pi * 100.0
    element = ControllerElement(ElementKind.REAL_ZERO, omega_corner)
    response = element.response(np.array([omega_corner]))[0]
    assert abs(response) == pytest.approx(np.sqrt(2.0))
    assert element.phase_deg(np.array([omega_corner]))[0] == pytest.approx(45.0)


def test_real_pole_multiplicity() -> None:
    omega_corner = 100.0
    element = ControllerElement(ElementKind.REAL_POLE, omega_corner, multiplicity=2)
    response = element.response(np.array([omega_corner]))[0]
    assert abs(response) == pytest.approx(0.5)
    assert element.phase_deg(np.array([omega_corner]))[0] == pytest.approx(-90.0)


def test_origin_elements_use_fixed_reference_omega() -> None:
    omega = np.array([10.0])
    integrator = ControllerElement(ElementKind.ORIGIN_POLE)
    differentiator = ControllerElement(ElementKind.ORIGIN_ZERO)
    assert abs(integrator.response(omega)[0]) == pytest.approx(0.1)
    assert integrator.phase_deg(omega)[0] == pytest.approx(-90.0)
    assert abs(differentiator.response(omega)[0]) == pytest.approx(10.0)
    assert differentiator.phase_deg(omega)[0] == pytest.approx(90.0)


def test_gain_k_and_db_are_bidirectionally_synchronized() -> None:
    controller = ControllerModel(gain_k=-10.0)
    assert controller.gain_db == pytest.approx(20.0)
    controller.set_gain_db(40.0)
    assert controller.gain_k == pytest.approx(-100.0)
    assert controller.phase_deg(np.array([1.0]))[0] == pytest.approx(-180.0)
    controller.set_gain_k(0.1)
    assert controller.gain_db == pytest.approx(-20.0)
    with pytest.raises(ValueError):
        controller.set_gain_k(0.0)


def test_complex_pair_extension_point_is_functional() -> None:
    omega_n = 1000.0
    pole_pair = ControllerElement(
        ElementKind.COMPLEX_POLE_PAIR,
        omega_n,
        damping_ratio=1.0 / np.sqrt(2.0),
    )
    magnitude = abs(pole_pair.response(np.array([omega_n]))[0])
    assert magnitude == pytest.approx(1.0 / np.sqrt(2.0))
