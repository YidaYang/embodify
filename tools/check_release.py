"""Conservative source-tree checks; not a complete secret or license audit."""
from __future__ import annotations
import ast
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".git", "__pycache__", ".pytest_cache", "build", "dist", "out"}
TEXT = {".py", ".md", ".toml", ".json", ".yml", ".txt", ".html"}
PATTERNS = {
    "personal path": re.compile(r"(?:/home/(?!USER/)[A-Za-z0-9_.-]+/|[A-Za-z]:[/\\]Users[/\\](?!USER\b)[A-Za-z0-9_.-]+)"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "access token": re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{32,})"),
}
#: Optional extra patterns (a JSON list of regular expressions), such as your own host names. The
#: file is ignored by git, so private patterns never end up in the repository.
LOCAL_PATTERNS = ROOT / "tools" / "release-patterns.local.json"
OWN_FILES = {"tools/check_release.py", "tools/release-patterns.local.json"}


def patterns():
    found = dict(PATTERNS)
    if LOCAL_PATTERNS.is_file():
        for index, regex in enumerate(json.loads(LOCAL_PATTERNS.read_text(encoding="utf-8")), 1):
            found["local pattern %d" % index] = re.compile(regex, re.I)
    return found


def versions():
    """Every place that carries the package version, so a release cannot publish them out of step."""
    def find(path, pattern):
        if not (ROOT / path).is_file():
            return None
        match = re.search(pattern, (ROOT / path).read_text(encoding="utf-8"), re.M)
        return match.group(1) if match else None
    server = json.loads((ROOT / "server.json").read_text(encoding="utf-8"))
    return server["name"], {
        "pyproject.toml": find("pyproject.toml", r'^version = "([^"]+)"'),
        "embodify_mcp/__init__.py": find("embodify_mcp/__init__.py", r'^__version__ = "([^"]+)"'),
        "server.json": server["version"],
        "server.json package": server["packages"][0]["version"],
        "packaging/embodify": find("packaging/embodify/pyproject.toml", r'^version = "([^"]+)"'),
        "packaging/embodify dependency": find("packaging/embodify/pyproject.toml", r'"embodify-mcp==([^"]+)"'),
        "plugin/.claude-plugin/plugin.json": find("plugin/.claude-plugin/plugin.json", r'"version": "([^"]+)"'),
        "plugin/.mcp.json": find("plugin/.mcp.json", r'"embodify-mcp==([^"]+)"'),
    }


def check():
    issues = []
    count = 0
    checks = patterns()
    for p in ROOT.rglob("*"):
        rel = p.relative_to(ROOT)
        if any(x in SKIP or x.startswith(".venv") or x.endswith(".egg-info") for x in rel.parts):
            continue
        if not p.is_file():
            continue
        if p.is_symlink():
            issues.append(str(rel) + ": symlink")
            continue
        count += 1
        if p.suffix.lower() not in TEXT:
            continue
        text = p.read_text(encoding="utf-8-sig")
        # The checker and the local pattern file contain the patterns themselves.
        if rel.as_posix() not in OWN_FILES:
            for label, pattern in checks.items():
                if pattern.search(text):
                    issues.append(str(rel) + ": " + label)
        if p.suffix == ".py":
            ast.parse(text, filename=str(rel), feature_version=(3, 8))
        if p.suffix == ".json":
            json.loads(text)
        if p.suffix == ".md":
            for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
                if "://" in target or target.startswith("#"):
                    continue
                path = target.split("#", 1)[0]
                if not (p.parent / path).exists():
                    issues.append(str(rel) + ": broken link " + target)
    for required in ("LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md", "README.md", "pyproject.toml",
                     "embodify_mcp/monitor_page.html", "plugin/.mcp.json"):
        if not (ROOT / required).is_file():
            issues.append("Missing " + required)
    server_name, found = versions()
    if len(set(found.values())) != 1:
        issues.append("Versions differ: " + json.dumps(found))
    # The MCP Registry accepts the PyPI package only if its README names the server.
    if "mcp-name: " + server_name not in (ROOT / "README.md").read_text(encoding="utf-8"):
        issues.append("README.md lacks mcp-name: " + server_name)
    if issues:
        raise SystemExit("\n".join(issues))
    print(json.dumps({"ok": True, "source_files": count, "python38_syntax": True,
                      "scope": "targeted patterns, metadata and local links; not exhaustive"}))

if __name__ == "__main__":
    check()
