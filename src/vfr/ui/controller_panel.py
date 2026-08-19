from __future__ import annotations

import numpy as np
from PySide6.QtCore import QObject, QSignalBlocker, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from vfr.analysis import LoopMetrics
from vfr.models import ControllerElement, ControllerModel, ElementKind
from vfr.ui.widgets import ScientificSpinBox


class ControllerStore(QObject):
    changed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.model = ControllerModel()

    def set_gain_k(self, value: float) -> None:
        self.model.set_gain_k(value)
        self.changed.emit()

    def set_gain_db(self, value: float) -> None:
        self.model.set_gain_db(value)
        self.changed.emit()

    def add_element(self, kind: ElementKind, frequency_hz: float = 1000.0) -> str:
        omega = 2.0 * np.pi * frequency_hz if kind.needs_frequency else None
        element = ControllerElement(kind=kind, omega_rad_s=omega)
        self.model.add_element(element)
        self.changed.emit()
        return element.element_id

    def remove_element(self, element_id: str) -> None:
        self.model.remove_element(element_id)
        self.changed.emit()

    def update_element(
        self,
        element_id: str,
        *,
        kind: ElementKind | None = None,
        omega_rad_s: float | None = None,
        multiplicity: int | None = None,
    ) -> None:
        element = self.model.get_element(element_id)
        if kind is not None:
            element.kind = kind
            if kind.needs_frequency and element.omega_rad_s is None:
                element.omega_rad_s = 2.0 * np.pi * 1000.0
        if omega_rad_s is not None:
            if not np.isfinite(omega_rad_s) or omega_rad_s <= 0:
                raise ValueError("角频率必须是有限正数")
            element.omega_rad_s = float(omega_rad_s)
        if multiplicity is not None:
            if multiplicity < 1:
                raise ValueError("重数必须是正整数")
            element.multiplicity = int(multiplicity)
        self.changed.emit()


class ControllerPanel(QWidget):
    def __init__(self, store: ControllerStore, parent=None) -> None:
        super().__init__(parent)
        self.store = store
        self.default_frequency_hz = 1000.0
        self._selected_id: str | None = None
        self._refreshing = False

        gain_group = QGroupBox("整体增益")
        self.gain_k_spin = ScientificSpinBox()
        self.gain_db_spin = ScientificSpinBox()
        self.gain_db_spin.setRange(-600.0, 600.0)
        gain_form = QFormLayout(gain_group)
        gain_form.addRow("K", self.gain_k_spin)
        gain_form.addRow("增益（dB）", self.gain_db_spin)

        elements_group = QGroupBox("控制器元素")
        self.element_list = QListWidget()
        add_zero = QPushButton("+ 实零点")
        add_pole = QPushButton("+ 实极点")
        add_integrator = QPushButton("+ 原点极点")
        add_differentiator = QPushButton("+ 原点零点")
        self.remove_button = QPushButton("删除选中元素")
        button_grid = QVBoxLayout()
        row_one = QHBoxLayout()
        row_one.addWidget(add_zero)
        row_one.addWidget(add_pole)
        row_two = QHBoxLayout()
        row_two.addWidget(add_integrator)
        row_two.addWidget(add_differentiator)
        button_grid.addLayout(row_one)
        button_grid.addLayout(row_two)
        button_grid.addWidget(self.remove_button)
        elements_layout = QVBoxLayout(elements_group)
        elements_layout.addWidget(self.element_list)
        elements_layout.addLayout(button_grid)

        detail_group = QGroupBox("元素参数")
        self.kind_combo = QComboBox()
        for kind in (
            ElementKind.REAL_ZERO,
            ElementKind.REAL_POLE,
            ElementKind.ORIGIN_ZERO,
            ElementKind.ORIGIN_POLE,
        ):
            self.kind_combo.addItem(kind.display_name, kind.value)
        self.frequency_hz_spin = ScientificSpinBox()
        self.frequency_hz_spin.setRange(1.0e-200, 1.0e200)
        self.omega_spin = ScientificSpinBox()
        self.omega_spin.setRange(1.0e-200, 1.0e200)
        self.multiplicity_spin = QSpinBox()
        self.multiplicity_spin.setRange(1, 99)
        detail_form = QFormLayout(detail_group)
        detail_form.addRow("类型", self.kind_combo)
        detail_form.addRow("频率 f（Hz）", self.frequency_hz_spin)
        detail_form.addRow("角频率 ω（rad/s）", self.omega_spin)
        detail_form.addRow("重数", self.multiplicity_spin)

        metrics_group = QGroupBox("环路指标")
        self.metrics_label = QLabel("请先导入功率通路 CSV")
        self.metrics_label.setWordWrap(True)
        metrics_layout = QVBoxLayout(metrics_group)
        metrics_layout.addWidget(self.metrics_label)

        layout = QVBoxLayout(self)
        layout.addWidget(gain_group)
        layout.addWidget(elements_group, 1)
        layout.addWidget(detail_group)
        layout.addWidget(metrics_group)

        self.gain_k_spin.editingFinished.connect(self._gain_k_edited)
        self.gain_db_spin.editingFinished.connect(self._gain_db_edited)
        add_zero.clicked.connect(
            lambda: self._add_element(ElementKind.REAL_ZERO)
        )
        add_pole.clicked.connect(
            lambda: self._add_element(ElementKind.REAL_POLE)
        )
        add_integrator.clicked.connect(
            lambda: self._add_element(ElementKind.ORIGIN_POLE)
        )
        add_differentiator.clicked.connect(
            lambda: self._add_element(ElementKind.ORIGIN_ZERO)
        )
        self.remove_button.clicked.connect(self._remove_selected)
        self.element_list.currentItemChanged.connect(self._selection_changed)
        self.kind_combo.currentIndexChanged.connect(self._kind_edited)
        self.frequency_hz_spin.editingFinished.connect(self._frequency_hz_edited)
        self.omega_spin.editingFinished.connect(self._omega_edited)
        self.multiplicity_spin.valueChanged.connect(self._multiplicity_edited)
        self.store.changed.connect(self.refresh)
        self.refresh()

    def _add_element(self, kind: ElementKind) -> None:
        element_id = self.store.add_element(kind, self.default_frequency_hz)
        self._selected_id = element_id
        self.refresh()

    def _remove_selected(self) -> None:
        if self._selected_id is None:
            return
        self.store.remove_element(self._selected_id)
        self._selected_id = None

    def _selection_changed(self, current: QListWidgetItem | None) -> None:
        self._selected_id = (
            str(current.data(Qt.ItemDataRole.UserRole)) if current is not None else None
        )
        self._refresh_detail()

    def _gain_k_edited(self) -> None:
        if self._refreshing:
            return
        try:
            self.store.set_gain_k(self.gain_k_spin.value())
        except ValueError:
            self.refresh()

    def _gain_db_edited(self) -> None:
        if not self._refreshing:
            self.store.set_gain_db(self.gain_db_spin.value())

    def _kind_edited(self) -> None:
        if self._refreshing or self._selected_id is None:
            return
        self.store.update_element(
            self._selected_id, kind=ElementKind(self.kind_combo.currentData())
        )

    def _frequency_hz_edited(self) -> None:
        if self._refreshing or self._selected_id is None:
            return
        self.store.update_element(
            self._selected_id,
            omega_rad_s=2.0 * np.pi * self.frequency_hz_spin.value(),
        )

    def _omega_edited(self) -> None:
        if self._refreshing or self._selected_id is None:
            return
        self.store.update_element(
            self._selected_id, omega_rad_s=self.omega_spin.value()
        )

    def _multiplicity_edited(self, value: int) -> None:
        if self._refreshing or self._selected_id is None:
            return
        self.store.update_element(self._selected_id, multiplicity=value)

    def refresh(self) -> None:
        self._refreshing = True
        with QSignalBlocker(self.gain_k_spin), QSignalBlocker(self.gain_db_spin):
            self.gain_k_spin.setValue(self.store.model.gain_k)
            self.gain_db_spin.setValue(self.store.model.gain_db)

        previous = self._selected_id
        with QSignalBlocker(self.element_list):
            self.element_list.clear()
            selected_row = -1
            for row, element in enumerate(self.store.model.elements):
                if element.kind.needs_frequency:
                    frequency = f" @ {element.frequency_hz:.3e} Hz"
                else:
                    frequency = ""
                item = QListWidgetItem(
                    f"{element.kind.display_name} ×{element.multiplicity}{frequency}"
                )
                item.setData(Qt.ItemDataRole.UserRole, element.element_id)
                self.element_list.addItem(item)
                if element.element_id == previous:
                    selected_row = row
            if selected_row >= 0:
                self.element_list.setCurrentRow(selected_row)
            elif self.element_list.count() > 0 and previous is None:
                self.element_list.setCurrentRow(0)
                self._selected_id = str(
                    self.element_list.item(0).data(Qt.ItemDataRole.UserRole)
                )
        self._refresh_detail()
        self._refreshing = False

    def _refresh_detail(self) -> None:
        self._refreshing = True
        element = None
        if self._selected_id is not None:
            try:
                element = self.store.model.get_element(self._selected_id)
            except KeyError:
                self._selected_id = None
        enabled = element is not None
        for widget in (
            self.kind_combo,
            self.frequency_hz_spin,
            self.omega_spin,
            self.multiplicity_spin,
            self.remove_button,
        ):
            widget.setEnabled(enabled)
        if element is not None:
            with (
                QSignalBlocker(self.kind_combo),
                QSignalBlocker(self.frequency_hz_spin),
                QSignalBlocker(self.omega_spin),
                QSignalBlocker(self.multiplicity_spin),
            ):
                self.kind_combo.setCurrentIndex(
                    self.kind_combo.findData(element.kind.value)
                )
                frequency_enabled = element.kind.needs_frequency
                self.frequency_hz_spin.setEnabled(frequency_enabled)
                self.omega_spin.setEnabled(frequency_enabled)
                self.frequency_hz_spin.setValue(element.frequency_hz or 1.0)
                self.omega_spin.setValue(element.omega_rad_s or 1.0)
                self.multiplicity_spin.setValue(element.multiplicity)
        self._refreshing = False

    def set_metrics(self, metrics: LoopMetrics | None) -> None:
        if metrics is None:
            self.metrics_label.setText("请先导入功率通路 CSV")
            return
        lines: list[str] = []
        primary = metrics.primary_gain_crossover
        if primary is None:
            lines.append("增益交越：未找到")
            lines.append("相位裕度：无法计算")
        else:
            lines.append(f"首次增益交越：{primary.frequency_hz:.6e} Hz")
            lines.append(f"相位裕度：{primary.phase_margin_deg:.3f}°")
        worst_pm = metrics.worst_phase_margin
        if worst_pm is not None and len(metrics.gain_crossovers) > 1:
            lines.append(
                f"最差 PM：{worst_pm.phase_margin_deg:.3f}° @ "
                f"{worst_pm.frequency_hz:.6e} Hz"
            )
        worst_gm = metrics.worst_gain_margin
        if worst_gm is None:
            lines.append("增益裕度：无法计算")
        else:
            lines.append(f"最差增益裕度：{worst_gm.gain_margin_db:.3f} dB")
        lines.extend(metrics.messages)
        lines.append("指标基于用户原始相位，不执行相位展开")
        self.metrics_label.setText("\n".join(lines))
