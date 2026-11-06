from dataclasses import asdict, dataclass, field
from datetime import datetime


@dataclass
class Note:
    id: int
    title: str
    body: str = ""
    tags: list[str] = field(default_factory=list)
    created: datetime = field(default_factory=datetime.now)

    def to_json(self) -> dict:
        d = asdict(self)
        d["created"] = self.created.isoformat(timespec="seconds")
        return d

    @classmethod
    def from_json(cls, d: dict) -> "Note":
        return cls(d["id"], d["title"], d.get("body", ""), list(d.get("tags", [])), datetime.fromisoformat(d["created"]))

    def line(self) -> str:
        tags = f"  [{', '.join(self.tags)}]" if self.tags else ""
        return f"{self.id:>3}  {self.created:%Y-%m-%d}  {self.title}{tags}"
