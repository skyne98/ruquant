import json
import os
from pathlib import Path

from .model import Note


def default_path() -> Path:
    return Path(os.environ.get("NOTES_FILE", Path.home() / ".notes.json"))


class Store:
    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else default_path()

    def load(self) -> list[Note]:
        if not self.path.exists():
            return []
        return [Note.from_json(d) for d in json.loads(self.path.read_text())]

    def save(self, notes: list[Note]) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps([n.to_json() for n in notes], indent=1))
        tmp.replace(self.path)

    def next_id(self, notes: list[Note]) -> int:
        return max((n.id for n in notes), default=0) + 1
