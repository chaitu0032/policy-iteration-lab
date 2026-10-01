"""Per-user run storage (repository pattern). Runs are stored gzip-compressed JSON.

* ``LocalRunRepository``  — files under ./data/runs/<user>/ (local dev).
* ``BlobRunRepository``   — Vercel Blob (see blob_store.py), used when BLOB_READ_WRITE_TOKEN is set.

The server never trusts the client's summary: it decodes, validates and summarises every upload.
"""

from __future__ import annotations

import gzip
import json
import re
import zlib
from pathlib import Path
from typing import Any, Protocol

from .rl.env_specs import SPECS

RUN_ID_PATTERN = re.compile(r"[a-f0-9]{12}")  # always used with fullmatch (no trailing-newline loophole)
USER_PATTERN = re.compile(r"[a-z0-9_-]{2,32}")
MAX_RUN_BYTES = 40 * 1024 * 1024  # decompressed JSON; guards against zip bombs
MAX_UPLOAD_BYTES = 4 * 1024 * 1024  # compressed; Vercel functions accept <= 4.5 MB request bodies
MAX_RUNS_PER_USER = 100


class InvalidRun(ValueError):
    """Upload is not a valid run document."""


class QuotaExceeded(RuntimeError):
    """User already stores MAX_RUNS_PER_USER runs."""


def _check_user(user: str) -> str:
    if not isinstance(user, str) or not USER_PATTERN.fullmatch(user):
        raise KeyError(user)
    return user


def _check_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id):
        raise KeyError(run_id)
    return run_id


def decode_run(compressed: bytes) -> dict[str, Any]:
    """Decompress (bounded), parse and validate a run document."""
    try:
        d = zlib.decompressobj(wbits=31)  # gzip container
        raw = d.decompress(compressed, MAX_RUN_BYTES + 1)
        if len(raw) > MAX_RUN_BYTES or d.unconsumed_tail:
            raise InvalidRun("Run is too large")
        doc = json.loads(raw)
    except RecursionError as exc:
        raise InvalidRun("Run JSON is nested too deeply") from exc
    except (zlib.error, ValueError) as exc:
        if isinstance(exc, InvalidRun):
            raise
        raise InvalidRun(f"Run must be gzip-compressed JSON ({exc})") from exc
    if not isinstance(doc, dict):
        raise InvalidRun("Run must be a JSON object")
    try:
        _check_id(doc.get("id"))
    except KeyError as exc:
        raise InvalidRun("Run id must be 12 hex characters") from exc
    env = doc.get("env")
    if not isinstance(env, dict) or env.get("key") not in SPECS:
        raise InvalidRun("Run env.key must be FrozenLake, CliffWalking or Taxi")
    its = doc.get("iterations")
    if not isinstance(its, list) or not its or not all(
        isinstance(it, dict) and isinstance(it.get("policy"), list) and isinstance(it.get("V"), list) for it in its
    ):
        raise InvalidRun("Run needs at least one iteration with policy and V")
    if not isinstance(doc.get("layout"), dict) or not isinstance(doc.get("config"), dict):
        raise InvalidRun("Run needs layout and config")
    return doc


def summarize_run(doc: dict[str, Any]) -> dict[str, Any]:
    try:
        return _summarize(doc)
    except (TypeError, ValueError, KeyError, AttributeError) as exc:
        raise InvalidRun(f"Run fields are malformed ({exc})") from exc


def _summarize(doc: dict[str, Any]) -> dict[str, Any]:
    its = doc["iterations"]
    idx = doc.get("optimal_index", len(its) - 1)
    best = its[idx] if isinstance(idx, int) and 0 <= idx < len(its) else its[-1]
    env, cfg = doc["env"], doc["config"]
    return {
        "id": doc["id"],
        "created_at": float(doc.get("created_at") or 0),
        "env_key": env["key"],
        "title": str(env.get("title", env["key"]))[:40],
        "options": env.get("options", {}),
        "gamma": cfg.get("gamma"),
        "eval_mode": cfg.get("eval_mode"),
        "n_iterations": len(its),
        "status": str(doc.get("status", "converged"))[:20],
        "v_start": (best.get("stats") or {}).get("v_start"),
    }


class RunRepository(Protocol):
    def list(self, user: str) -> list[dict[str, Any]]: ...
    def get_compressed(self, user: str, run_id: str) -> bytes: ...
    def save(self, user: str, doc: dict[str, Any], compressed: bytes) -> dict[str, Any]: ...
    def delete(self, user: str, run_id: str) -> None: ...


class LocalRunRepository:
    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    def _dir(self, user: str) -> Path:
        return self._root / _check_user(user)

    def list(self, user: str) -> list[dict[str, Any]]:
        folder = self._dir(user)
        if not folder.exists():
            return []
        out = [json.loads(p.read_text()) for p in folder.glob("*.summary.json")]
        return sorted(out, key=lambda s: s["created_at"], reverse=True)

    def get_compressed(self, user: str, run_id: str) -> bytes:
        path = self._dir(user) / f"{_check_id(run_id)}.json.gz"
        if not path.exists():
            raise KeyError(run_id)
        return path.read_bytes()

    def save(self, user: str, doc: dict[str, Any], compressed: bytes) -> dict[str, Any]:
        folder = self._dir(user)
        run_id = _check_id(doc["id"])
        exists = (folder / f"{run_id}.json.gz").exists()
        if not exists and len(list(folder.glob("*.summary.json"))) >= MAX_RUNS_PER_USER:
            raise QuotaExceeded(f"At most {MAX_RUNS_PER_USER} saved runs per user; delete some first")
        folder.mkdir(parents=True, exist_ok=True)
        summary = summarize_run(doc)
        _atomic_write(folder / f"{run_id}.json.gz", compressed)
        _atomic_write(folder / f"{run_id}.summary.json", json.dumps(summary).encode())
        return summary

    def delete(self, user: str, run_id: str) -> None:
        folder = self._dir(user)
        data = folder / f"{_check_id(run_id)}.json.gz"
        if not data.exists():
            raise KeyError(run_id)
        data.unlink()
        (folder / f"{run_id}.summary.json").unlink(missing_ok=True)


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def gzip_json(doc: dict[str, Any]) -> bytes:
    return gzip.compress(json.dumps(doc, separators=(",", ":")).encode(), compresslevel=6)
