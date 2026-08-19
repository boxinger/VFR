from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import (
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QToolBar,
    QWidget,
)

from vfr.analysis import LoopAnalysis, analyze_loop
from vfr.csv_import import CsvImportError
from vfr.models import FrequencyResponse
from vfr.ui.controller_panel import ControllerPanel, ControllerStore
from vfr.ui.import_dialog import CsvImportDialog
from vfr.ui.widgets import BodePanel


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("VFR 控制器频响设计器")
        self.resize(1680, 920)
        self.plant_response: FrequencyResponse | None = None
        self.loop_analysis: LoopAnalysis | None = None
        self._gain_drag_start_db = 0.0

        self.store = ControllerStore(self)
        self.controller_panel = ControllerPanel(self.store)
        self.plant_panel = BodePanel("功率通路  Gp(jω)")
        self.controller_bode = BodePanel("控制器  C(jω)", interactive_gain=True)
        self.loop_panel = BodePanel("环路  L(jω)")
        self.controller_bode.link_x_to(self.plant_panel)
        self.loop_panel.link_x_to(self.plant_panel)

        plots = QSplitter()
        plots.addWidget(self.plant_panel)
        plots.addWidget(self.controller_bode)
        plots.addWidget(self.loop_panel)
        plots.setSizes([500, 500, 500])

        main_splitter = QSplitter()
        main_splitter.addWidget(self.controller_panel)
        main_splitter.addWidget(plots)
        main_splitter.setSizes([330, 1350])
        self.setCentralWidget(main_splitter)

        self._create_toolbar()
        self.statusBar().showMessage("请打开包含功率通路频率响应的 CSV")

        for panel in (self.plant_panel, self.controller_bode, self.loop_panel):
            panel.crosshairMoved.connect(self._sync_crosshair)
        self.controller_bode.gain_plot.gainDragStarted.connect(
            self._gain_drag_started
        )
        self.controller_bode.gain_plot.gainDragged.connect(self._gain_dragged)
        self.store.changed.connect(self._refresh_all)
        self._set_mode("pan")
        self._refresh_all()

    def _create_toolbar(self) -> None:
        toolbar = QToolBar("主工具栏")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        open_action = QAction("打开 CSV", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_csv)
        toolbar.addAction(open_action)
        toolbar.addSeparator()

        group = QActionGroup(self)
        group.setExclusive(True)
        for label, mode, checked in (
            ("缩放/平移", "pan", True),
            ("编辑零极点", "elements", False),
            ("调整增益", "gain", False),
        ):
            action = QAction(label, self)
            action.setCheckable(True)
            action.setChecked(checked)
            action.triggered.connect(lambda _checked, value=mode: self._set_mode(value))
            group.addAction(action)
            toolbar.addAction(action)

    def _set_mode(self, mode: str) -> None:
        pan = mode == "pan"
        for panel in (self.plant_panel, self.controller_bode, self.loop_panel):
            panel.set_pan_enabled(pan)
        self.controller_bode.set_marker_movable(mode == "elements")
        self.controller_bode.set_gain_drag_enabled(mode == "gain")
        self.statusBar().showMessage(
            {
                "pan": "缩放/平移模式",
                "elements": "零极点编辑模式：拖动控制器图中的竖线",
                "gain": "增益调整模式：在控制器增益图中上下拖动",
            }[mode]
        )

    def open_csv(self) -> None:
        start = Path.cwd() / "data"
        path, _ = QFileDialog.getOpenFileName(
            self, "打开频率响应 CSV", str(start), "CSV 文件 (*.csv);;所有文件 (*)"
        )
        if not path:
            return
        try:
            dialog = CsvImportDialog(path, self)
        except CsvImportError:
            return
        if dialog.exec() != CsvImportDialog.DialogCode.Accepted:
            return
        if dialog.response is None:
            return
        self.set_plant_response(dialog.response, Path(path).name)

    def set_plant_response(self, response: FrequencyResponse, source_name: str = "") -> None:
        self.plant_response = response
        minimum = float(response.frequency_hz[0])
        maximum = float(response.frequency_hz[-1])
        self.controller_panel.default_frequency_hz = float(np.sqrt(minimum * maximum))
        for panel in (self.plant_panel, self.controller_bode, self.loop_panel):
            panel.set_frequency_range(minimum, maximum)
        self.statusBar().showMessage(
            f"已导入 {len(response.frequency_hz)} 个频点：{source_name}"
        )
        self._refresh_all()

    def _refresh_all(self) -> None:
        if self.plant_response is None:
            display_frequency = np.logspace(0.0, 5.0, 800)
            omega = 2.0 * np.pi * display_frequency
            self.controller_bode.set_data(
                display_frequency,
                self.store.model.magnitude_db(omega),
                self.store.model.phase_deg(omega),
            )
            self.plant_panel.set_data(np.array([]), np.array([]), np.array([]))
            self.loop_panel.set_data(np.array([]), np.array([]), np.array([]))
            self.controller_panel.set_metrics(None)
        else:
            plant = self.plant_response
            self.plant_panel.set_data(
                plant.frequency_hz, plant.magnitude_db, plant.phase_deg
            )
            dense_frequency = np.logspace(
                np.log10(plant.frequency_hz[0]),
                np.log10(plant.frequency_hz[-1]),
                1000,
            )
            dense_omega = 2.0 * np.pi * dense_frequency
            self.controller_bode.set_data(
                dense_frequency,
                self.store.model.magnitude_db(dense_omega),
                self.store.model.phase_deg(dense_omega),
            )
            self.loop_analysis = analyze_loop(plant, self.store.model)
            loop = self.loop_analysis.loop_response
            self.loop_panel.set_data(loop.frequency_hz, loop.magnitude_db, loop.phase_deg)
            self.controller_panel.set_metrics(self.loop_analysis.metrics)

        self.controller_bode.sync_markers(
            self.store.model.elements, self._marker_frequency_changed
        )

    def _marker_frequency_changed(self, element_id: str, frequency_hz: float) -> None:
        try:
            self.store.update_element(
                element_id, omega_rad_s=2.0 * np.pi * frequency_hz
            )
        except (KeyError, ValueError) as exc:
            QMessageBox.warning(self, "无法修改零极点", str(exc))

    def _sync_crosshair(self, log_frequency: float) -> None:
        for panel in (self.plant_panel, self.controller_bode, self.loop_panel):
            panel.set_crosshair(log_frequency)
        self.statusBar().showMessage(f"光标频率：{10.0**log_frequency:.6e} Hz")

    def _gain_drag_started(self) -> None:
        self._gain_drag_start_db = self.store.model.gain_db

    def _gain_dragged(self, delta_db: float) -> None:
        self.store.set_gain_db(self._gain_drag_start_db + delta_db)
