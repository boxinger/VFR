from pathlib import Path

import numpy as np
import pytest

from vfr.csv_import import ResponseRepresentation
from vfr.models import ElementKind, FrequencyResponse
from vfr.ui.controller_panel import ControllerPanel, ControllerStore
from vfr.ui.import_dialog import CsvImportDialog
from vfr.ui.main_window import MainWindow


def test_import_dialog_requires_explicit_mapping(qtbot, tmp_path: Path) -> None:
    path = tmp_path / "response.csv"
    path.write_text("f,db,phase,re,im\n1,2,3,4,5\n2,3,4,5,6\n", encoding="utf-8")
    dialog = CsvImportDialog(path)
    qtbot.addWidget(dialog)
    assert not dialog.ok_button.isEnabled()

    dialog.frequency_combo.setCurrentIndex(dialog.frequency_combo.findData("f"))
    dialog.representation_combo.setCurrentIndex(
        dialog.representation_combo.findData(ResponseRepresentation.DB_PHASE.value)
    )
    dialog.first_combo.setCurrentIndex(dialog.first_combo.findData("db"))
    dialog.second_combo.setCurrentIndex(dialog.second_combo.findData("phase"))
    assert dialog.ok_button.isEnabled()


def test_controller_panel_synchronizes_fields(qtbot) -> None:
    store = ControllerStore()
    panel = ControllerPanel(store)
    qtbot.addWidget(panel)
    panel.show()
    element_id = store.add_element(ElementKind.REAL_ZERO, 100.0)
    panel._selected_id = element_id
    panel.refresh()
    assert panel.omega_spin.value() == pytest.approx(2.0 * np.pi * 100.0)

    panel.frequency_hz_spin.setValue(1000.0)
    panel.frequency_hz_spin.editingFinished.emit()
    assert store.model.get_element(element_id).omega_rad_s == pytest.approx(
        2.0 * np.pi * 1000.0
    )
    store.set_gain_k(10.0)
    assert panel.gain_db_spin.value() == pytest.approx(20.0)


def test_main_window_updates_curves_markers_and_gain(qtbot) -> None:
    frequency = np.array([10.0, 100.0, 1000.0])
    phase = np.array([-10.0, -60.0, -120.0])
    response = 10.0 ** (np.array([20.0, 0.0, -20.0]) / 20.0) * np.exp(
        1j * np.radians(phase)
    )
    plant = FrequencyResponse(2.0 * np.pi * frequency, response, phase)
    window = MainWindow()
    qtbot.addWidget(window)
    window.set_plant_response(plant, "test.csv")
    assert len(window.plant_panel.gain_curve.xData) == 3
    assert window.loop_analysis is not None

    element_id = window.store.add_element(ElementKind.REAL_ZERO, 100.0)
    assert element_id in window.controller_bode._markers
    line = window.controller_bode._markers[element_id][0]
    line.setValue(np.log10(200.0))
    assert window.store.model.get_element(element_id).frequency_hz == pytest.approx(200.0)

    window._gain_drag_started()
    window._gain_dragged(6.0)
    assert window.store.model.gain_db == pytest.approx(6.0)
