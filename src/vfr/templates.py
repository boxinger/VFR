from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from PySide6.QtCore import QStandardPaths

from .csv_import import CsvMapping, ResponseRepresentation


@dataclass(frozen=True, slots=True)
class ImportTemplate:
    name: str
    mapping: CsvMapping

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["mapping"]["representation"] = self.mapping.representation.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ImportTemplate:
        mapping_data = dict(data["mapping"])
        mapping_data["representation"] = ResponseRepresentation(
            mapping_data["representation"]
        )
        return cls(str(data["name"]), CsvMapping(**mapping_data))


class TemplateStore:
    def __init__(self, path: str | Path | None = None) -> None:
        if path is None:
            config_dir = Path(
                QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppConfigLocation)
            )
            path = config_dir / "import_templates.json"
        self.path = Path(path)

    def load(self) -> list[ImportTemplate]:
        if not self.path.exists():
            return []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            return [ImportTemplate.from_dict(item) for item in raw]
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return []

    def save(self, templates: list[ImportTemplate]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = [template.to_dict() for template in templates]
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def upsert(self, template: ImportTemplate) -> None:
        templates = [item for item in self.load() if item.name != template.name]
        templates.append(template)
        templates.sort(key=lambda item: item.name.casefold())
        self.save(templates)
