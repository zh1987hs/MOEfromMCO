from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    """YAML-backed configuration wrapper."""

    raw: dict[str, Any]

    @classmethod
    def from_file(cls, path: str | Path) -> "Config":
        with Path(path).open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return cls(raw=data)

    def get(self, *keys: str, default: Any = None) -> Any:
        cur: Any = self.raw
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    def dump(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.raw, handle, sort_keys=False)
