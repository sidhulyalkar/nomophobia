from __future__ import annotations

import os
import subprocess
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .artifacts import atomic_write_json, runtime_manifest, sha256_file


MANIFEST_SCHEMA_VERSION = 1
DEFAULT_DATA_FILES = ("train.csv", "test.csv", "sample_submission.csv")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return value


def git_provenance(repo_root: str | Path | None = None) -> dict[str, Any]:
    """Best-effort git identity that also works in Kaggle/CI snapshots without `.git`."""

    env_sha = os.environ.get("NOMOPHOBIA_GIT_SHA") or os.environ.get("GITHUB_SHA")
    result: dict[str, Any] = {
        "sha": env_sha,
        "branch": os.environ.get("GITHUB_REF_NAME"),
        "dirty": None,
        "source": "environment" if env_sha else None,
    }
    root = Path(repo_root or Path.cwd())
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=3,
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=3,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
                timeout=3,
            ).stdout.strip()
        )
        result.update({"sha": sha, "branch": branch, "dirty": dirty, "source": "git"})
    except (OSError, subprocess.SubprocessError):
        pass
    return result


def data_provenance(
    data_dir: str | Path,
    *,
    files: Iterable[str] = DEFAULT_DATA_FILES,
    hash_files: bool = True,
) -> dict[str, Any]:
    root = Path(data_dir)
    records: dict[str, Any] = {}
    for name in files:
        path = root / name
        record: dict[str, Any] = {"path": str(path), "exists": path.exists()}
        if path.exists():
            record["bytes"] = int(path.stat().st_size)
            if hash_files:
                record["sha256"] = sha256_file(path)
        records[name] = record
    return {"root": str(root), "files": records, "hash_files": bool(hash_files)}


def output_provenance(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    records = []
    seen: set[str] = set()
    for value in paths:
        path = Path(value)
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        record: dict[str, Any] = {"path": key, "exists": path.exists()}
        if path.exists() and path.is_file():
            record.update({"bytes": int(path.stat().st_size), "sha256": sha256_file(path)})
        records.append(record)
    return records


@dataclass
class ExperimentRecorder:
    """Universal, failure-safe manifest recorder for research and production runs."""

    path: str | Path
    experiment_id: str
    evidence_tier: str
    hypothesis: str
    falsifier: str
    accept_rule: str
    kill_rule: str
    config: dict[str, Any] = field(default_factory=dict)
    data_dir: str | Path | None = None
    hash_inputs: bool = True
    repo_root: str | Path | None = None
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.started_at = _utc_now()
        self._clock = time.perf_counter()
        self.metrics: dict[str, Any] = {}
        self.outputs: list[str | Path] = []
        self.status = "RUNNING"

    def add_metrics(self, **metrics: Any) -> None:
        self.metrics.update(metrics)

    def add_output(self, *paths: str | Path) -> None:
        self.outputs.extend(paths)

    def add_note(self, note: str) -> None:
        self.notes.append(str(note))

    def _payload(self, *, error: BaseException | None = None) -> dict[str, Any]:
        elapsed = float(time.perf_counter() - self._clock)
        payload: dict[str, Any] = {
            "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
            "experiment_id": self.experiment_id,
            "evidence_tier": self.evidence_tier,
            "status": self.status,
            "scientific_contract": {
                "hypothesis": self.hypothesis,
                "falsifier": self.falsifier,
                "accept_rule": self.accept_rule,
                "kill_rule": self.kill_rule,
            },
            "started_at_utc": self.started_at,
            "finished_at_utc": _utc_now(),
            "elapsed_seconds": elapsed,
            "git": git_provenance(self.repo_root),
            "runtime": runtime_manifest(),
            "config": _jsonable(self.config),
            "metrics": _jsonable(self.metrics),
            "outputs": output_provenance(self.outputs),
            "notes": list(self.notes),
        }
        if self.data_dir is not None:
            payload["data"] = data_provenance(self.data_dir, hash_files=self.hash_inputs)
        if error is not None:
            payload["error"] = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": "".join(
                    traceback.format_exception(type(error), error, error.__traceback__)
                )[-12000:],
            }
        return payload

    def write(self, *, error: BaseException | None = None) -> dict[str, Any]:
        payload = self._payload(error=error)
        atomic_write_json(self.path, payload)
        return payload

    def __enter__(self) -> "ExperimentRecorder":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.write()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is None:
            if self.status == "RUNNING":
                self.status = "COMPLETE"
            self.write()
            return False
        self.status = "FAILED"
        self.write(error=exc)
        return False
