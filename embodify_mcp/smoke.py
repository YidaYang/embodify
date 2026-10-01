"""Exercise an installed package over real stdio, without any simulator."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def run(root, remote=False):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "embodify_mcp", "--image-size", "16",
               "--output-root", str(root / "runs")]
    if remote:
        config = root / "backend.local.json"
        config.write_text(json.dumps({"transport": "stdio", "command": [sys.executable, "-m",
            "embodify_mcp.backend.serve", "--backend", "fake", "--image-size", "16"]}), encoding="utf-8")
        command += ["--backend", "remote", "--remote-config", str(config)]
    else:
        command += ["--backend", "fake"]
    requests = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
        "protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "smoke", "version": "1"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}]
    calls = [("reset_task", {}), ("observe", {}), ("move_relative", {"delta_xyz": [0, 0, .02]}),
             ("set_gripper", {"state": "close"}), ("stop_episode", {}),
             ("reset_task", {}), ("stop_episode", {})]
    for ident, (name, args) in enumerate(calls, 3):
        requests.append({"jsonrpc": "2.0", "id": ident, "method": "tools/call",
                         "params": {"name": name, "arguments": args}})
    proc = subprocess.run(command, input="".join(json.dumps(x)+"\n" for x in requests),
                          encoding="utf-8", capture_output=True, timeout=60)
    if proc.returncode:
        raise RuntimeError("MCP subprocess failed: " + proc.stderr)
    results = [json.loads(line) for line in proc.stdout.splitlines()]
    if [x.get("id") for x in results] != list(range(1, 10)):
        raise RuntimeError("Missing or unexpected protocol responses")
    for response in results:
        if "error" in response or response.get("result", {}).get("isError"):
            raise RuntimeError("Smoke tool failed: " + json.dumps(response))
    if not any(x.get("type") == "image" for x in results[2]["result"]["content"]):
        raise RuntimeError("Reset did not include an image")
    for response in (results[6], results[8]):
        summary = Path(response["result"]["structuredContent"]["artifacts"]["summary"])
        if not summary.is_file():
            raise RuntimeError("Episode summary missing")
    from .monitor import PAGE
    if not PAGE.is_file():
        raise RuntimeError("Installed package is missing the replay page")
    return {"ok": True, "backend": "remote-fake" if remote else "fake", "episodes": 2,
            "responses": len(results), "package": str(Path(__file__).resolve().parent),
            "gpu_validated": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", action="store_true", help="Exercise the backend subprocess transport too")
    parser.add_argument("--output-root", help="Keep smoke logs here; otherwise use a temporary directory")
    args = parser.parse_args()
    if args.output_root:
        result = run(args.output_root, args.remote)
    else:
        with tempfile.TemporaryDirectory(prefix="embodify-smoke-") as root:
            result = run(root, args.remote)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
