"""Run journal: everything an episode produced, written to disk as it happens.

Each episode gets its own directory with a manifest, an append-only event log, the archived
camera frames and, once the episode ends, a summary. Replays, the monitor page and evaluation
all read from here; nothing in it is ever returned to the Agent.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .imaging import encode_frame


@dataclass
class RunManifest:
    """Metadata of one run. A run with `privileged` set may only be reported as an oracle ceiling."""

    run_id: str
    arm: str
    rung: str
    split_digest: str
    frozen_digest: str | None
    privileged: bool
    privileged_reason: str = ""
    sim: str = "unknown"
    decision_every_k: int | None = None
    model: dict[str, Any] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    notes: str = ""


class Journal:
    """Writes everything a run produces into one directory.

    Layout:
        <root>/<run_id>/manifest.json
        <root>/<run_id>/events.jsonl        one event per line
        <root>/<run_id>/images/<name>.<ext> archived camera frames
        <root>/<run_id>/summary.json        written when the episode ends
    """

    def __init__(self, root: Path, manifest: RunManifest) -> None:
        self.dir = Path(root) / manifest.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "images").mkdir(exist_ok=True)
        self.manifest = manifest
        self._events = self.dir / "events.jsonl"
        self._write_manifest()

    # -- writing -----------------------------------------------------------

    def _write_manifest(self) -> None:
        (self.dir / "manifest.json").write_text(
            json.dumps(asdict(self.manifest), ensure_ascii=False, indent=2, default=_encode),
            encoding="utf-8",
        )

    def event(self, kind: str, **payload: Any) -> None:
        record = {"t": time.time(), "kind": kind, **payload}
        with self._events.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=_encode) + "\n")

    def save_frame(self, name: str, rgb: np.ndarray) -> Path:
        """Save a replay frame as lossy WebP (PNG when Pillow is unavailable)."""
        data, ext = encode_frame(rgb)
        path = self.dir / "images" / f"{name}.{ext}"
        path.write_bytes(data)
        return path

    def finish(self, summary: dict[str, Any]) -> Path:
        path = self.dir / "summary.json"
        path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=_encode), encoding="utf-8"
        )
        self.event("finish")
        return path

    # -- reading (for replay and analysis) -----------------------------------

    def read_events(self) -> list[dict[str, Any]]:
        if not self._events.exists():
            return []
        return [json.loads(line) for line in self._events.read_text(encoding="utf-8").splitlines() if line]


def _encode(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    return repr(obj)
