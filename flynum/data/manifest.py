"""Derived-artefact manifest.

Every processed artefact (edge lists, subgraphs, shuffled controls, retina
maps, stimulus sets) registers itself here with a hash and shape summary, so a
run can always be traced back to the exact inputs it used.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .. import paths

MANIFEST_PATH = paths.DATA_PROCESSED / "manifest.json"


def _file_digest(path: Path, limit: int = 1 << 20) -> str:
    """Digest of size + head/tail bytes (fast fingerprint for large arrays)."""
    size = path.stat().st_size
    h = hashlib.sha256(str(size).encode())
    with path.open("rb") as fh:
        h.update(fh.read(limit))
        if size > 2 * limit:
            fh.seek(-limit, 2)
            h.update(fh.read(limit))
    return h.hexdigest()[:16]


class Manifest:
    """Append-only record of derived artefacts."""

    def __init__(self, path: Path = MANIFEST_PATH):
        self.path = path
        self.data: dict[str, Any] = {"entries": {}, "updated": None}
        if path.exists():
            try:
                self.data = json.loads(path.read_text(encoding="utf-8"))
                self.data.setdefault("entries", {})
            except json.JSONDecodeError:
                pass

    # ------------------------------------------------------------------ #
    def add(self, name: str, path: Path | None = None, **meta: Any) -> dict[str, Any]:
        rec: dict[str, Any] = {"meta": meta}
        if path is not None and path.exists():
            rec.update(
                {
                    "path": str(path.relative_to(paths.ROOT))
                    if path.is_relative_to(paths.ROOT)
                    else str(path),
                    "size_bytes": path.stat().st_size,
                    "digest": _file_digest(path),
                }
            )
        elif path is not None:
            rec["path"] = str(path)
            rec["missing"] = True
        self.data["entries"][name] = rec
        return rec

    def get(self, name: str) -> dict[str, Any] | None:
        return self.data["entries"].get(name)

    def save(self) -> Path:
        self.data["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.data, indent=2, default=str), encoding="utf-8"
        )
        return self.path

    def __contains__(self, name: str) -> bool:
        return name in self.data["entries"]
