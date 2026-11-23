"""Small helper functions for data handling."""

from __future__ import annotations

import hashlib
import importlib.metadata
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_versions(names: tuple[str, ...]) -> dict[str, str | None]:
    """Return installed versions without importing optional GPU packages."""
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def first_existing_sha256(directory: str | Path, names: tuple[str, ...]) -> str | None:
    """Hash the first matching adapter file, if one exists."""
    root = Path(directory)
    for name in names:
        path = root / name
        if path.is_file():
            return file_sha256(path)
    return None

