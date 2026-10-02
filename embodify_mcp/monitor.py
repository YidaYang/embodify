"""Embodify monitor: a read-only local web page showing what the Agent is doing and has done.

It **only reads the disk**: all data comes from the journals the MCP server has already written
(`~/.embodify/runs/<timestamp>/<rN>/` by default). It never talks to the MCP process or touches the simulator, which
has three benefits:

- a crashed, stuck or ten-tabs-open monitor cannot slow down or interrupt a running episode;
- a running episode and one that finished days ago are the same thing, so live view and replay
  share one code path;
- it needs neither LIBERO nor numpy; any Python 3.8+ can run it.

**It is for people, not for the Agent.** The page shows evaluator-side success, which the Agent
must not see during evaluation. It listens on 127.0.0.1 by default; isolating an evaluation also
needs operating-system permission boundaries.

Usage:
    python -m embodify_mcp.monitor                  # reads ~/.embodify/runs, serves http://127.0.0.1:8765
    python -m embodify_mcp.monitor --open           # also opens a browser
    python -m embodify_mcp.mcp --monitor-port 8765   # let the MCP server run one itself
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse

PAGE = Path(__file__).with_name("monitor_page.html")

#: The only characters allowed in each path segment. Run directories are timestamps and rN, frame
#: labels look like `move_0012`, cameras like `robot0_eye_in_hand`; none has a reason to contain dots
#: or slashes.
_SAFE = re.compile(r"^[A-Za-z0-9_\-]+$")

#: The most bytes of events returned per request. A 700-step episode is about 300 KB, so one
#: request is enough; the cap only keeps an unusually large file from bloating a single response,
#: and the page asks for the rest.
_MAX_CHUNK = 4 * 1024 * 1024

#: For the run list, only this many leading bytes are read to find the scene event, not the whole events.jsonl.
_HEAD_BYTES = 16 * 1024


def _clean(obj: Any) -> Any:
    """Replace NaN / Inf with None.

    Python's json writes NaN as a bare `NaN` by default, which the browser's JSON.parse rejects;
    a single bad number would make the whole episode impossible to open in the page.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return _clean(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return None


def read_events(path: Path, offset: int = 0, limit: int = _MAX_CHUNK) -> Tuple[List[Dict[str, Any]], int]:
    """Read complete event lines from byte offset `offset`; returns the events and where to read next.

    The MCP server keeps appending to this file, so the last line may be half written at any moment.
    Reading stops at the last newline and leaves a partial line for next time; the offset counts
    bytes rather than lines precisely so reading can resume in the middle of that line.
    """
    try:
        with path.open("rb") as handle:
            handle.seek(offset)
            chunk = handle.read(limit)
    except OSError:
        return [], offset
    end = chunk.rfind(b"\n")
    if end < 0:
        return [], offset
    events: List[Dict[str, Any]] = []
    for line in chunk[: end + 1].splitlines():
        if not line.strip():
            continue
        try:
            events.append(_clean(json.loads(line.decode("utf-8"))))
        except (UnicodeDecodeError, ValueError):
            # One bad line must not hide the hundreds after it. Mark it; the page shows it.
            events.append({"kind": "unreadable", "raw": line[:200].decode("utf-8", "replace")})
    return events, offset + end + 1


class RunIndex:
    """Scans the output directory and lists every episode.

    Each run's metadata (scene, summary) is cached by the file's (mtime, size): the list refreshes
    every few seconds, and with hundreds of runs on disk they should not all be reread each time.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._cache: Dict[str, Tuple[Tuple[float, int, bool], Dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def run_dir(self, stamp: str, run: str) -> Optional[Path]:
        if not (_SAFE.match(stamp) and _SAFE.match(run)):
            return None
        path = self.root / stamp / run
        return path if (path / "events.jsonl").is_file() else None

    def list_runs(self) -> List[Dict[str, Any]]:
        if not self.root.is_dir():
            return []
        runs: List[Dict[str, Any]] = []
        for stamp_dir in self.root.iterdir():
            if not (stamp_dir.is_dir() and _SAFE.match(stamp_dir.name)):
                continue
            for run_dir in stamp_dir.iterdir():
                events = run_dir / "events.jsonl"
                if _SAFE.match(run_dir.name) and events.is_file():
                    runs.append(self._describe(stamp_dir.name, run_dir.name, run_dir, events))
        runs.sort(key=lambda item: item["updated_at"], reverse=True)
        return runs

    def _describe(self, stamp: str, run: str, run_dir: Path, events: Path) -> Dict[str, Any]:
        stat = events.stat()
        summary_path = run_dir / "summary.json"
        key = (stat.st_mtime, stat.st_size, summary_path.is_file())
        run_id = f"{stamp}/{run}"
        with self._lock:
            hit = self._cache.get(run_id)
        if hit is not None and hit[0] == key:
            info = dict(hit[1])
        else:
            head, _ = read_events(events, 0, _HEAD_BYTES)
            scene = next((e for e in head if e.get("kind") == "scene"), {})
            summary = _read_json(summary_path) if key[2] else None
            info = {
                "id": run_id,
                "stamp": stamp,
                "run": run,
                "session": scene.get("session"),
                "backend": scene.get("backend"),
                "suite": scene.get("suite"),
                "task_index": scene.get("task_index"),
                "name": scene.get("name"),
                "instruction": scene.get("instruction"),
                "init_state_index": scene.get("init_state_index"),
                "started_at": head[0].get("t") if head else stat.st_mtime,
                "finished": summary is not None,
                "steps": summary.get("steps") if summary else None,
                "success": summary.get("success") if summary else None,
                "size_bytes": stat.st_size,
            }
            with self._lock:
                self._cache[run_id] = (key, info)
            info = dict(info)
        info["updated_at"] = stat.st_mtime
        info["idle_s"] = max(0.0, time.time() - stat.st_mtime)
        return info


def make_handler(index: RunIndex) -> type:
    class Handler(BaseHTTPRequestHandler):
        server_version = "embodify_mcp-monitor"

        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: D401 - silenced
            # The page polls every half second, and the default access log would flood stderr. The
            # MCP server's stderr is where the host reads diagnostics, so it must not drown.
            return

        def do_GET(self) -> None:  # noqa: N802 - http.server convention
            url = urlparse(self.path)
            parts = [unquote(p) for p in url.path.split("/") if p]
            query = parse_qs(url.query)
            try:
                if not parts:
                    return self._page()
                if parts == ["api", "runs"]:
                    return self._json({"root": str(index.root.resolve()), "now": time.time(), "runs": index.list_runs()})
                if len(parts) == 5 and parts[:2] == ["api", "run"] and parts[4] == "events":
                    return self._events(parts[2], parts[3], query)
                if len(parts) == 7 and parts[:2] == ["api", "run"] and parts[4] == "frame":
                    return self._frame(parts[2], parts[3], parts[5], parts[6])
                self._error(404, "no such endpoint")
            except (BrokenPipeError, ConnectionResetError):
                pass  # browsers drop unused image requests when frames change quickly; this is normal

        def _page(self) -> None:
            body = PAGE.read_bytes()
            self._send(200, "text/html; charset=utf-8", body, cache="no-store")

        def _events(self, stamp: str, run: str, query: Dict[str, List[str]]) -> None:
            run_dir = index.run_dir(stamp, run)
            if run_dir is None:
                return self._error(404, "no such run")
            try:
                offset = max(0, int(query.get("offset", ["0"])[0]))
            except ValueError:
                return self._error(400, "offset must be an integer")
            events_path = run_dir / "events.jsonl"
            events, next_offset = read_events(events_path, offset)
            try:
                size = events_path.stat().st_size
            except OSError:
                size = next_offset
            payload: Dict[str, Any] = {
                "events": events,
                "offset": next_offset,
                "size": size,
                "summary": _read_json(run_dir / "summary.json"),
            }
            if offset == 0:
                payload["manifest"] = _read_json(run_dir / "manifest.json")
            self._json(payload)

        def _frame(self, stamp: str, run: str, label: str, camera: str) -> None:
            run_dir = index.run_dir(stamp, run)
            if run_dir is None or not (_SAFE.match(label) and _SAFE.match(camera)):
                return self._error(404, "no such frame")
            # Frames are WebP, or PNG without Pillow; a run has only one kind, and the page need not know which.
            for ext, mime in (("webp", "image/webp"), ("png", "image/png")):
                path = run_dir / "images" / f"{label}_{camera}.{ext}"
                if path.is_file():
                    # A written frame never changes, so let the browser cache it; scrubbing stays smooth.
                    return self._send(200, mime, path.read_bytes(), cache="public, max-age=31536000, immutable")
            self._error(404, "no such frame")

        def _json(self, payload: Any) -> None:
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self._send(200, "application/json; charset=utf-8", body, cache="no-store")

        def _error(self, code: int, message: str) -> None:
            body = json.dumps({"error": message}).encode("utf-8")
            self._send(code, "application/json; charset=utf-8", body, cache="no-store")

        def _send(self, code: int, mime: str, body: bytes, *, cache: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            self.end_headers()
            self.wfile.write(body)

    return Handler


def make_server(root: Path, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(RunIndex(root)))
    server.daemon_threads = True
    return server


def start_in_background(root: Path, port: int, host: str = "127.0.0.1") -> Optional[ThreadingHTTPServer]:
    """For the MCP server: serve the monitor page from a background thread. On failure, print one line and never affect MCP itself.

    The usual failure is a port already in use, for example by another MCP process or a monitor
    started separately. That monitor reads the same directory and shows the same things, so this
    one quietly steps aside.
    """
    try:
        server = make_server(root, host, port)
    except OSError as exc:
        print(f"[embodify-mcp] Monitor page not started ({exc}); run python -m embodify_mcp.monitor separately", file=sys.stderr)
        return None
    thread = threading.Thread(target=server.serve_forever, name="mcp-monitor", daemon=True)
    thread.start()
    print(f"[embodify-mcp] Monitor page: http://{host}:{server.server_address[1]}/", file=sys.stderr)
    return server


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Embodify monitor: a read-only web page for live view and replay")
    parser.add_argument("--root", default="~/.embodify/runs", help="The MCP server's --output-root")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="Open a browser after starting")
    args = parser.parse_args(argv)
    args.root = Path(args.root).expanduser()
    server = make_server(args.root, args.host, args.port)
    url = f"http://{args.host}:{server.server_address[1]}/"
    print(f"Monitor page: {url}   (data directory {Path(args.root).resolve()}; Ctrl+C to quit)")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
