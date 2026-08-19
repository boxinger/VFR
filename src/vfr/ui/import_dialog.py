from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from vfr.csv_import import (
    CsvImportError,
    CsvMapping,
    ResponseRepresentation,
    inspect_csv,
    load_frequency_response,
)
from vfr.models import FrequencyResponse
from vfr.templates import ImportTemplate, TemplateStore


class CsvImportDialog(QDialog):
    def __init__(self, path: str | Path, parent=None) -> None:
        super().__init__(parent)
        self.path = Path(path)
        self.response: FrequencyResponse | None = None
        self.mapping: CsvMapping | None = None
        self.store = TemplateStore()
        self.templates = self.store.load()

        try:
            inspection = inspect_csv(self.path)
        except CsvImportError as exc:
            QMessageBox.critical(parent, "CSV 导入失败", str(exc))
            raise
        self.headers = list(inspection.headers)

        self.setWindowTitle("导入频率响应")
        self.resize(580, 420)
        file_label = QLabel(str(self.path))
        file_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        rows_label = QLabel(f"数据行：{inspection.data_row_count}")

        self.template_combo = QComboBox()
        self.template_combo.addItem("不使用模板", None)
        for template in self.templates:
            self.template_combo.addItem(template.name, template.name)

        self.frequency_combo = self._column_combo()
        self.representation_combo = QComboBox()
        self.representation_combo.addItem("请选择响应形式", None)
        for representation in ResponseRepresentation:
            self.representation_combo.addItem(
                representation.display_name, representation.value
            )
        self.first_label = QLabel("第一响应列")
        self.second_label = QLabel("第二响应列")
        self.first_combo = self._column_combo()
        self.second_combo = self._column_combo()

        form = QFormLayout()
        form.addRow("文件", file_label)
        form.addRow("", rows_label)
        form.addRow("导入模板", self.template_combo)
        form.addRow("频率列（Hz）", self.frequency_combo)
        form.addRow("响应形式", self.representation_combo)
        form.addRow(self.first_label, self.first_combo)
        form.addRow(self.second_label, self.second_combo)

        self.save_template_button = QPushButton("保存当前映射为模板")
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        self.ok_button = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.ok_button.setText("导入")
        self.ok_button.setEnabled(False)

        button_row = QHBoxLayout()
        button_row.addWidget(self.save_template_button)
        button_row.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addStretch(1)
        layout.addLayout(button_row)
        layout.addWidget(self.buttons)

        self.template_combo.currentIndexChanged.connect(self._apply_template)
        self.representation_combo.currentIndexChanged.connect(self._update_labels)
        for combo in (
            self.frequency_combo,
            self.representation_combo,
            self.first_combo,
            self.second_combo,
        ):
            combo.currentIndexChanged.connect(self._update_validity)
        self.save_template_button.clicked.connect(self._save_template)
        self.buttons.accepted.connect(self._accept_import)
        self.buttons.rejected.connect(self.reject)
        self._update_labels()

    def _column_combo(self) -> QComboBox:
        combo = QComboBox()
        combo.addItem("请选择列", None)
        for header in self.headers:
            combo.addItem(header, header)
        return combo

    def _current_mapping(self) -> CsvMapping | None:
        frequency = self.frequency_combo.currentData()
        representation = self.representation_combo.currentData()
        first = self.first_combo.currentData()
        second = self.second_combo.currentData()
        if not all((frequency, representation, first, second)):
            return None
        return CsvMapping(
            str(frequency),
            ResponseRepresentation(str(representation)),
            str(first),
            str(second),
        )

    def _update_labels(self) -> None:
        value = self.representation_combo.currentData()
        if value == ResponseRepresentation.DB_PHASE.value:
            self.first_label.setText("增益列（dB）")
            self.second_label.setText("相位列（degree）")
        elif value == ResponseRepresentation.LINEAR_PHASE.value:
            self.first_label.setText("线性幅值列")
            self.second_label.setText("相位列（degree）")
        elif value == ResponseRepresentation.REAL_IMAGINARY.value:
            self.first_label.setText("实部列")
            self.second_label.setText("虚部列")
        else:
            self.first_label.setText("第一响应列")
            self.second_label.setText("第二响应列")
        self._update_validity()

    def _update_validity(self) -> None:
        valid = self._current_mapping() is not None
        self.ok_button.setEnabled(valid)
        self.save_template_button.setEnabled(valid)

    def _set_combo_data(self, combo: QComboBox, value: str) -> bool:
        index = combo.findData(value)
        if index < 0:
            return False
        combo.setCurrentIndex(index)
        return True

    def _apply_template(self) -> None:
        name = self.template_combo.currentData()
        if name is None:
            return
        template = next((item for item in self.templates if item.name == name), None)
        if template is None:
            return
        mapping = template.mapping
        values_exist = all(
            value in self.headers
            for value in (
                mapping.frequency_column,
                mapping.first_response_column,
                mapping.second_response_column,
            )
        )
        if not values_exist:
            QMessageBox.warning(self, "模板不匹配", "模板映射的列不在当前 CSV 中")
            self.template_combo.setCurrentIndex(0)
            return
        self._set_combo_data(self.frequency_combo, mapping.frequency_column)
        self._set_combo_data(self.representation_combo, mapping.representation.value)
        self._set_combo_data(self.first_combo, mapping.first_response_column)
        self._set_combo_data(self.second_combo, mapping.second_response_column)

    def _save_template(self) -> None:
        mapping = self._current_mapping()
        if mapping is None:
            return
        name, accepted = QInputDialog.getText(self, "保存导入模板", "模板名称")
        name = name.strip()
        if not accepted or not name:
            return
        self.store.upsert(ImportTemplate(name, mapping))
        self.templates = self.store.load()
        self.template_combo.blockSignals(True)
        self.template_combo.clear()
        self.template_combo.addItem("不使用模板", None)
        for template in self.templates:
            self.template_combo.addItem(template.name, template.name)
        self.template_combo.setCurrentIndex(self.template_combo.findData(name))
        self.template_combo.blockSignals(False)

    def _accept_import(self) -> None:
        mapping = self._current_mapping()
        if mapping is None:
            return
        try:
            response = load_frequency_response(self.path, mapping)
        except CsvImportError as exc:
            QMessageBox.critical(self, "CSV 导入失败", str(exc))
            return
        self.mapping = mapping
        self.response = response
        self.accept()
