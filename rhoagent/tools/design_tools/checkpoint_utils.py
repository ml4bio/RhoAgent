"""Utilities for checkpoint-directory based design ensembles."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, Sequence


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def dotenv_candidates() -> tuple[Path, ...]:
    candidates: list[Path] = [
        Path.cwd() / ".env",
        _repo_root() / ".env",
    ]
    for env_name in ("RHOAGENT_ROOT", "WORKSPACE_ROOT"):
        root = os.environ.get(env_name)
        if root:
            candidates.append(Path(root).expanduser() / ".env")

    seen: set[str] = set()
    unique: list[Path] = []
    for path in candidates:
        key = str(path.expanduser())
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return tuple(unique)


def _read_dotenv_value(name: str) -> str | None:
    for env_path in dotenv_candidates():
        if not env_path.is_file():
            continue
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key.strip() != name:
                    continue
                value = value.strip().strip('"').strip("'")
                return value or None
        except OSError:
            continue
    return None


def get_env_value(names: str | Sequence[str]) -> str | None:
    if isinstance(names, str):
        names = [names]
    for name in names:
        value = os.environ.get(name) or _read_dotenv_value(name)
        if value:
            return value
    return None


def require_env_value(names: str | Sequence[str]) -> str:
    value = get_env_value(names)
    if value:
        return value
    if isinstance(names, str):
        names = [names]
    joined = " or ".join(names)
    raise ValueError(
        f"Missing required environment variable: {joined}. "
        "Set it in .env or the shell environment."
    )


def parse_checkpoint_paths(value: str | Iterable[str] | None) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
    else:
        parts = [str(p).strip() for p in value if str(p).strip()]
    return [Path(p).expanduser().resolve() for p in parts]


def resolve_checkpoint_paths(
    *,
    checkpoint_dir: str | None = None,
    checkpoint_paths: str | Iterable[str] | None = None,
    env_dir_vars: str | Sequence[str],
    fallback_path: str | Path | None = None,
) -> list[Path]:
    explicit_paths = [p for p in parse_checkpoint_paths(checkpoint_paths) if p.is_file()]
    if explicit_paths:
        return sorted(explicit_paths)

    directory_value = checkpoint_dir or get_env_value(env_dir_vars)
    if directory_value:
        checkpoint_root = Path(directory_value).expanduser().resolve()
        if checkpoint_root.is_file() and checkpoint_root.suffix == ".pth":
            return [checkpoint_root]
        if checkpoint_root.is_dir():
            paths = sorted(checkpoint_root.rglob("*.pth"))
            if paths:
                return paths

    if fallback_path:
        fallback = Path(fallback_path).expanduser().resolve()
        if fallback.is_file():
            return [fallback]
    return []


def split_samples(total_samples: int, n_chunks: int) -> list[int]:
    if total_samples <= 0:
        raise ValueError("total_samples must be positive")
    if n_chunks <= 0:
        raise ValueError("n_chunks must be positive")
    base = total_samples // n_chunks
    remainder = total_samples % n_chunks
    return [base + (1 if i < remainder else 0) for i in range(n_chunks)]


def checkpoint_tag(path: Path, index: int) -> str:
    stem = path.stem
    parent = path.parent.name
    raw = f"{index:02d}_{parent}_{stem}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)[:80]
