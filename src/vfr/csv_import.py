from __future__ import annotations

import csv
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np

from .models import FrequencyResponse


class CsvImportError(ValueError):
    pass


class ResponseRepresentation(StrEnum):
    DB_PHASE = "db_phase"
    LINEAR_PHASE = "linear_phase"
    REAL_IMAGINARY = "real_imaginary"

    @property
    def display_name(self) -> str:
        return {
            ResponseRepresentation.DB_PHASE: "增益 dB + 相位 degree",
            ResponseRepresentation.LINEAR_PHASE: "线性幅值 + 相位 degree",
            ResponseRepresentation.REAL_IMAGINARY: "实部 + 虚部",
        }[self]


@dataclass(frozen=True, slots=True)
class CsvMapping:
    frequency_column: str
    representation: ResponseRepresentation
    first_response_column: str
    second_response_column: str


@dataclass(frozen=True, slots=True)
class CsvInspection:
    headers: tuple[str, ...]
    data_row_count: int


def _read_rows(path: str | Path) -> tuple[list[str], list[tuple[int, list[str]]]]:
    file_path = Path(path)
    try:
        with file_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, delimiter=",")
            numbered_rows = [
                (line_number, row)
                for line_number, row in enumerate(reader, start=1)
                if row and any(cell.strip() for cell in row)
            ]
    except UnicodeDecodeError as exc:
        raise CsvImportError("CSV 必须使用 UTF-8 编码") from exc
    except OSError as exc:
        raise CsvImportError(f"无法读取 CSV：{exc}") from exc

    if not numbered_rows:
        raise CsvImportError("CSV 为空或没有表头")
    _, header_row = numbered_rows[0]
    headers = [cell.strip() for cell in header_row]
    if any(not header for header in headers):
        raise CsvImportError("CSV 表头不能为空")
    if len(headers) != len(set(headers)):
        raise CsvImportError("CSV 表头不能重复")
    return headers, numbered_rows[1:]


def inspect_csv(path: str | Path) -> CsvInspection:
    headers, rows = _read_rows(path)
    return CsvInspection(tuple(headers), len(rows))


def _parse_number(value: str, line_number: int, column: str) -> float:
    text = value.strip()
    try:
        number = float(text)
    except ValueError as exc:
        raise CsvImportError(f"第 {line_number} 行的 {column} 不是合法数值") from exc
    if not np.isfinite(number):
        raise CsvImportError(f"第 {line_number} 行的 {column} 必须是有限数值")
    return number


def load_frequency_response(
    path: str | Path,
    mapping: CsvMapping,
) -> FrequencyResponse:
    headers, rows = _read_rows(path)
    required = (
        mapping.frequency_column,
        mapping.first_response_column,
        mapping.second_response_column,
    )
    missing = [name for name in required if name not in headers]
    if missing:
        raise CsvImportError(f"缺少已映射的列：{', '.join(missing)}")
    indices = {name: headers.index(name) for name in required}

    frequencies: list[float] = []
    responses: list[complex] = []
    phases: list[float] = []
    frequency_lines: dict[float, int] = {}

    for line_number, row in rows:
        needed_index = max(indices.values())
        if len(row) <= needed_index:
            raise CsvImportError(f"第 {line_number} 行缺少已映射的字段")
        frequency = _parse_number(
            row[indices[mapping.frequency_column]], line_number, mapping.frequency_column
        )
        if frequency <= 0:
            raise CsvImportError(f"第 {line_number} 行频率必须为正数")
        if frequency in frequency_lines:
            first_line = frequency_lines[frequency]
            raise CsvImportError(
                f"重复频率 {frequency:.6e} Hz，位于第 {first_line}、{line_number} 行"
            )
        frequency_lines[frequency] = line_number

        first = _parse_number(
            row[indices[mapping.first_response_column]],
            line_number,
            mapping.first_response_column,
        )
        second = _parse_number(
            row[indices[mapping.second_response_column]],
            line_number,
            mapping.second_response_column,
        )
        if mapping.representation == ResponseRepresentation.DB_PHASE:
            magnitude = 10.0 ** (first / 20.0)
            phase = second
            response = magnitude * np.exp(1j * np.radians(phase))
        elif mapping.representation == ResponseRepresentation.LINEAR_PHASE:
            phase = second
            response = first * np.exp(1j * np.radians(phase))
        else:
            response = complex(first, second)
            phase = float(np.degrees(np.arctan2(second, first)))
        frequencies.append(frequency)
        responses.append(response)
        phases.append(phase)

    if len(frequencies) < 2:
        raise CsvImportError("CSV 至少需要两个有效数据行")

    order = np.argsort(np.asarray(frequencies, dtype=np.float64), kind="stable")
    frequency_hz = np.asarray(frequencies, dtype=np.float64)[order]
    response_complex = np.asarray(responses, dtype=np.complex128)[order]
    phase_deg = np.asarray(phases, dtype=np.float64)[order]
    return FrequencyResponse(2.0 * np.pi * frequency_hz, response_complex, phase_deg)
