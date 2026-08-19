from pathlib import Path

import numpy as np
import pytest

from vfr.csv_import import (
    CsvImportError,
    CsvMapping,
    ResponseRepresentation,
    inspect_csv,
    load_frequency_response,
)
from vfr.templates import ImportTemplate, TemplateStore


def write_csv(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_db_phase_import_sorts_frequency_and_preserves_raw_phase(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path / "response.csv",
        "frequency_hz,gain_db,phase_deg,status\n"
        "1e3,-20,-270,bad\n"
        "1e1,20,170,ok\n"
        "1e2,0,-190,anything\n",
    )
    response = load_frequency_response(
        path,
        CsvMapping(
            "frequency_hz",
            ResponseRepresentation.DB_PHASE,
            "gain_db",
            "phase_deg",
        ),
    )
    np.testing.assert_allclose(response.frequency_hz, [10.0, 100.0, 1000.0])
    np.testing.assert_allclose(response.phase_deg, [170.0, -190.0, -270.0])
    np.testing.assert_allclose(
        response.magnitude_db, [20.0, 0.0, -20.0], atol=1.0e-12
    )


def test_linear_phase_and_real_imaginary_modes(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path / "response.csv",
        "f,linear,phase,real,imag\n"
        "1,2,-90,0,-2\n"
        "2,4,90,0,4\n",
    )
    linear = load_frequency_response(
        path,
        CsvMapping("f", ResponseRepresentation.LINEAR_PHASE, "linear", "phase"),
    )
    np.testing.assert_allclose(linear.magnitude_linear, [2.0, 4.0])
    np.testing.assert_allclose(linear.phase_deg, [-90.0, 90.0])

    real_imag = load_frequency_response(
        path,
        CsvMapping("f", ResponseRepresentation.REAL_IMAGINARY, "real", "imag"),
    )
    np.testing.assert_allclose(real_imag.phase_deg, [-90.0, 90.0])


def test_duplicate_frequency_is_rejected_with_line_numbers(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path / "duplicate.csv",
        "f,db,phase\n10,1,2\n1e1,3,4\n",
    )
    with pytest.raises(CsvImportError, match="第 2、3 行"):
        load_frequency_response(
            path,
            CsvMapping("f", ResponseRepresentation.DB_PHASE, "db", "phase"),
        )


@pytest.mark.parametrize(
    "content, message",
    [
        ("f,f,phase\n1,2,3\n", "表头不能重复"),
        ("f,db,phase\n0,2,3\n1,2,3\n", "频率必须为正数"),
        ("f,db,phase\n1,nan,3\n2,2,3\n", "必须是有限数值"),
    ],
)
def test_structural_errors_are_rejected(tmp_path: Path, content: str, message: str) -> None:
    path = write_csv(tmp_path / "bad.csv", content)
    mapping = CsvMapping("f", ResponseRepresentation.DB_PHASE, "db", "phase")
    if "表头" in message:
        with pytest.raises(CsvImportError, match=message):
            inspect_csv(path)
    else:
        with pytest.raises(CsvImportError, match=message):
            load_frequency_response(path, mapping)


def test_invalid_utf8_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "invalid.csv"
    path.write_bytes(b"f,db,phase\n1,2,\xff\n")
    with pytest.raises(CsvImportError, match="UTF-8"):
        inspect_csv(path)


def test_template_round_trip(tmp_path: Path) -> None:
    store = TemplateStore(tmp_path / "templates.json")
    template = ImportTemplate(
        "My FRA",
        CsvMapping("f", ResponseRepresentation.REAL_IMAGINARY, "re", "im"),
    )
    store.upsert(template)
    assert store.load() == [template]
