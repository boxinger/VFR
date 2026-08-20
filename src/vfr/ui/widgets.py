from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pyqtgraph as pg
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QEvent, QPointF, Qt, Signal
from PySide6.QtGui import QMouseEvent, QPalette, QValidator
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QGridLayout,
    QLabel,
    QApplication,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from vfr.models import ControllerElement, ElementKind


class TransferFunctionFormula(FigureCanvasQTAgg):
    formula_latex = (
        r"$C(s)=K\frac{\prod_i\left(1+s/\omega_{z_i}\right)^{n_i}}"
        r"{s^{n_0}\prod_k\left(1+s/\omega_{p_k}\right)^{m_k}}$"
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        figure = Figure(figsize=(5.6, 1.25), dpi=100)
        super().__init__(figure)
        self.setParent(parent)
        self.setMinimumHeight(92)
        self.setMaximumHeight(120)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.axes = figure.add_axes((0.0, 0.0, 1.0, 1.0))
        self.axes.set_axis_off()
        self.formula_artist = self.axes.text(
            0.5,
            0.5,
            self.formula_latex,
            ha="center",
            va="center",
            fontsize=21,
            math_fontfamily="stix",
        )
        self._apply_qt_palette()

    def _apply_qt_palette(self) -> None:
        parent = self.parentWidget()
        application = QApplication.instance()
        if parent is not None:
            palette = parent.palette()
        elif application is not None:
            palette = application.palette()
        else:
            palette = self.palette()
        background = palette.color(QPalette.ColorRole.Window).name()
        foreground = palette.color(QPalette.ColorRole.WindowText).name()
        self.figure.set_facecolor(background)
        self.axes.set_facecolor(background)
        self.formula_artist.set_color(foreground)
        self.draw_idle()

    def changeEvent(self, event: QEvent) -> None:  # noqa: N802
        super().changeEvent(event)
        if event.type() in {
            QEvent.Type.PaletteChange,
            QEvent.Type.ApplicationPaletteChange,
            QEvent.Type.StyleChange,
            QEvent.Type.ThemeChange,
            QEvent.Type.ParentChange,
        } and hasattr(self, "formula_artist"):
            self._apply_qt_palette()


class ScientificSpinBox(QDoubleSpinBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setDecimals(12)
        self.setRange(-1.0e200, 1.0e200)
        self.setKeyboardTracking(False)
        self.setAccelerated(True)

    def textFromValue(self, value: float) -> str:  # noqa: N802
        return f"{value:.6e}"

    def valueFromText(self, text: str) -> float:  # noqa: N802
        try:
            return float(text.strip())
        except ValueError:
            return 0.0

    def validate(self, text: str, position: int):  # type: ignore[override]
        stripped = text.strip().lower()
        if stripped in {"", "+", "-", ".", "+.", "-.", "e", "+e", "-e"}:
            return QValidator.State.Intermediate, text, position
        try:
            value = float(stripped)
        except ValueError:
            return QValidator.State.Intermediate, text, position
        if not np.isfinite(value) or value < self.minimum() or value > self.maximum():
            return QValidator.State.Invalid, text, position
        return QValidator.State.Acceptable, text, position

    def stepBy(self, steps: int) -> None:  # noqa: N802
        value = self.value()
        if value == 0.0:
            self.setValue(10.0 ** (steps / 10.0))
        else:
            self.setValue(value * 10.0 ** (steps / 10.0))


class FrequencyAxisItem(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):  # noqa: N802
        labels: list[str] = []
        for value in values:
            fraction = value - np.floor(value)
            useful = any(
                abs(fraction - candidate) < 1.0e-3
                for candidate in (0.0, np.log10(2.0), np.log10(5.0))
            )
            labels.append(f"{10.0**value:.1e}" if useful else "")
        return labels


class InteractivePlotWidget(pg.PlotWidget):
    gainDragStarted = Signal()
    gainDragged = Signal(float)
    gainDragFinished = Signal()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.gain_drag_enabled = False
        self._drag_start_y: float | None = None

    def _view_position(self, event: QMouseEvent) -> QPointF:
        scene_position = self.mapToScene(event.position().toPoint())
        return self.plotItem.vb.mapSceneToView(scene_position)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self.gain_drag_enabled and event.button() == Qt.MouseButton.LeftButton:
            self._drag_start_y = self._view_position(event).y()
            self.gainDragStarted.emit()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._drag_start_y is not None:
            delta_db = self._view_position(event).y() - self._drag_start_y
            self.gainDragged.emit(delta_db)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._drag_start_y is not None:
            self._drag_start_y = None
            self.gainDragFinished.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)


class BodePanel(QWidget):
    crosshairMoved = Signal(float)

    def __init__(self, title: str, interactive_gain: bool = False, parent=None) -> None:
        super().__init__(parent)
        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("font-size: 15px; font-weight: 600;")

        axis_gain = FrequencyAxisItem(orientation="bottom")
        axis_phase = FrequencyAxisItem(orientation="bottom")
        self.gain_plot = InteractivePlotWidget(axisItems={"bottom": axis_gain})
        self.phase_plot = InteractivePlotWidget(axisItems={"bottom": axis_phase})
        self.gain_plot.gain_drag_enabled = interactive_gain
        self.phase_plot.setXLink(self.gain_plot)

        self.gain_plot.setLabel("left", "增益", units="dB")
        self.phase_plot.setLabel("left", "相位", units="°")
        self.phase_plot.setLabel("bottom", "频率", units="Hz")
        self.gain_plot.getAxis("left").setWidth(62)
        self.phase_plot.getAxis("left").setWidth(62)
        self.gain_plot.showGrid(x=True, y=True, alpha=0.25)
        self.phase_plot.showGrid(x=True, y=True, alpha=0.25)
        self.gain_plot.getAxis("bottom").setStyle(showValues=False)

        self.gain_curve = self.gain_plot.plot(pen=pg.mkPen("#4da3ff", width=2))
        self.phase_curve = self.phase_plot.plot(pen=pg.mkPen("#ffb454", width=2))
        self.zero_db_line = pg.InfiniteLine(0.0, angle=0, pen=pg.mkPen("#777", width=1))
        self.minus_180_line = pg.InfiniteLine(
            -180.0, angle=0, pen=pg.mkPen("#777", width=1)
        )
        self.gain_plot.addItem(self.zero_db_line)
        self.phase_plot.addItem(self.minus_180_line)

        self._crosshair_gain = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen("#888", style=Qt.PenStyle.DotLine)
        )
        self._crosshair_phase = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen("#888", style=Qt.PenStyle.DotLine)
        )
        self.gain_plot.addItem(self._crosshair_gain, ignoreBounds=True)
        self.phase_plot.addItem(self._crosshair_phase, ignoreBounds=True)
        self._crosshair_gain.hide()
        self._crosshair_phase.hide()

        self._mouse_proxies = [
            pg.SignalProxy(
                self.gain_plot.scene().sigMouseMoved,
                rateLimit=60,
                slot=lambda event: self._scene_mouse_moved(self.gain_plot, event),
            ),
            pg.SignalProxy(
                self.phase_plot.scene().sigMouseMoved,
                rateLimit=60,
                slot=lambda event: self._scene_mouse_moved(self.phase_plot, event),
            ),
        ]
        self._markers: dict[str, tuple[pg.InfiniteLine, pg.InfiniteLine]] = {}
        self._marker_callback: Callable[[str, float], None] | None = None
        self._markers_movable = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.title_label)
        layout.addWidget(self.gain_plot, 1)
        layout.addWidget(self.phase_plot, 1)

    def _scene_mouse_moved(self, plot: pg.PlotWidget, event) -> None:
        position = event[0] if isinstance(event, tuple) else event
        if plot.sceneBoundingRect().contains(position):
            view_position = plot.plotItem.vb.mapSceneToView(position)
            self.crosshairMoved.emit(float(view_position.x()))

    def set_crosshair(self, log_frequency: float) -> None:
        self._crosshair_gain.setPos(log_frequency)
        self._crosshair_phase.setPos(log_frequency)
        self._crosshair_gain.show()
        self._crosshair_phase.show()

    def set_data(
        self,
        frequency_hz: np.ndarray,
        magnitude_db: np.ndarray,
        phase_deg: np.ndarray,
    ) -> None:
        if len(frequency_hz) == 0:
            self.gain_curve.setData([], [])
            self.phase_curve.setData([], [])
            return
        log_frequency = np.log10(frequency_hz)
        self.gain_curve.setData(log_frequency, magnitude_db)
        self.phase_curve.setData(log_frequency, phase_deg)

    def set_frequency_range(self, minimum_hz: float, maximum_hz: float) -> None:
        left = np.log10(minimum_hz)
        right = np.log10(maximum_hz)
        padding = max((right - left) * 0.03, 0.02)
        self.gain_plot.setXRange(left - padding, right + padding, padding=0.0)

    def link_x_to(self, other: BodePanel) -> None:
        self.gain_plot.setXLink(other.gain_plot)

    def set_pan_enabled(self, enabled: bool) -> None:
        self.gain_plot.setMouseEnabled(x=enabled, y=enabled)
        self.phase_plot.setMouseEnabled(x=enabled, y=enabled)

    def set_gain_drag_enabled(self, enabled: bool) -> None:
        self.gain_plot.gain_drag_enabled = enabled

    def set_marker_movable(self, movable: bool) -> None:
        self._markers_movable = movable
        for lines in self._markers.values():
            for line in lines:
                line.setMovable(movable)

    def sync_markers(
        self,
        elements: list[ControllerElement],
        callback: Callable[[str, float], None],
    ) -> None:
        self._marker_callback = callback
        visible = {
            element.element_id: element
            for element in elements
            if element.kind.needs_frequency and element.frequency_hz is not None
        }
        for element_id in list(self._markers):
            if element_id not in visible:
                for line in self._markers.pop(element_id):
                    line.scene().removeItem(line)
        for index, element in enumerate(visible.values(), start=1):
            log_frequency = float(np.log10(element.frequency_hz))
            if element.element_id not in self._markers:
                lines: list[pg.InfiniteLine] = []
                for plot in (self.gain_plot, self.phase_plot):
                    line = pg.InfiniteLine(
                        log_frequency,
                        angle=90,
                        movable=self._markers_movable,
                        pen=pg.mkPen("#ffffff", width=2),
                        hoverPen=pg.mkPen("#ffffff", width=3),
                        label="",
                        labelOpts={"position": 0.92, "color": "#ffffff"},
                    )
                    line.sigPositionChanged.connect(
                        lambda moved_line, eid=element.element_id: self._marker_moved(
                            eid, moved_line
                        )
                    )
                    plot.addItem(line)
                    lines.append(line)
                self._markers[element.element_id] = (lines[0], lines[1])
            is_zero = element.kind in {
                ElementKind.REAL_ZERO,
                ElementKind.COMPLEX_ZERO_PAIR,
            }
            color = "#32c788" if is_zero else "#ff5f6d"
            prefix = "Z" if is_zero else "P"
            label = f"{prefix}{index} ×{element.multiplicity}"
            for line in self._markers[element.element_id]:
                blocked = line.blockSignals(True)
                line.setPen(pg.mkPen(color, width=2))
                if line.label is not None:
                    line.label.setText(label)
                    line.label.setColor(color)
                line.setPos(log_frequency)
                line.blockSignals(blocked)

    def _marker_moved(self, element_id: str, line: pg.InfiniteLine) -> None:
        pair = self._markers[element_id]
        position = float(line.value())
        for other in pair:
            if other is not line:
                blocked = other.blockSignals(True)
                other.setPos(position)
                other.blockSignals(blocked)
        if self._marker_callback is not None:
            self._marker_callback(element_id, 10.0**position)
