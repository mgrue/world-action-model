"""Run and provenance helpers shared by `wam.compressor` (Step 4) and the world action model (Steps 5–8).

Run directories, repo-relative paths, git state, hashes, timestamps, atomic resume files and the stdout tee.
"""

from __future__ import annotations

import datetime
import hashlib
import io
import os
import subprocess
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
CHECKPOINTS = Path("checkpoints")


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rel(path: str | Path) -> str:
    """Repo-relative path (never absolute)."""
    p = Path(path).resolve()
    try:
        return p.relative_to(REPO).as_posix()
    except ValueError:
        return Path(os.path.relpath(p, REPO)).as_posix()


def data_path(data: str | Path) -> Path:
    """A dataset path from a config or card (repo-relative) or the command line (as given)."""
    p = Path(data)
    return p if p.is_absolute() or p.exists() else REPO / p


def _git_info() -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True)
        status = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return commit.stdout.strip(), bool(status.stdout.strip())


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


class _Tee(io.TextIOBase):
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, s: str) -> int:
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self) -> None:
        for st in self.streams:
            if not st.closed:
                st.flush()

    def close(self) -> None:  # the wrapped streams belong to the caller
        pass


def _new_run_dir(root: Path, name: str) -> Path:
    """`root/<YYYYMMDD-HHMM>-<name>`; a clash gets a numeric suffix."""
    base = Path(root) / f"{datetime.datetime.now():%Y%m%d-%H%M}-{name}"
    out, k = base, 1
    while out.exists():
        k += 1
        out = base.with_name(f"{base.name}-{k}")
    out.mkdir(parents=True)
    return out


def _save_atomic(obj, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def _read_weights(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)
