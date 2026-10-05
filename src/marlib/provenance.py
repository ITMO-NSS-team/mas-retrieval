"""Local provenance without importing model frameworks or contacting services."""
from __future__ import annotations

import hashlib
from importlib.metadata import distributions
from pathlib import Path


def file_hash(path: Path) -> str | None:
    if not path.is_file():
        return None
    checksum = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def runtime_provenance(systems_dir: Path) -> dict:
    paths = list(Path("src/marlib").rglob("*.py")) + list(systems_dir.rglob("*.py"))
    paths += [Path("pyproject.toml")]
    return {"source_sha256": {str(p): file_hash(p) for p in sorted(paths)},
            "packages": dict(sorted((d.metadata["Name"], d.version) for d in distributions()
                                    if d.metadata["Name"]))}
