import gzip
import json

import pytest

from backend.storage import (
    MAX_RUNS_PER_USER,
    InvalidRun,
    LocalRunRepository,
    decode_run,
    summarize_run,
)


def make_run(run_id="abcdef123456", env_key="FrozenLake", n=2):
    return {
        "id": run_id, "created_at": 1.0, "status": "converged", "optimal_index": n - 1,
        "env": {"key": env_key, "title": "Frozen Lake", "options": {"map_name": "4x4"}},
        "layout": {"kind": "grid"}, "config": {"gamma": 0.9, "eval_mode": "iterative"}, "train_seconds": 0.1,
        "iterations": [{"index": k, "policy": [0] * 16, "V": [0.0] * 16, "stats": {"v_start": 0.5}} for k in range(n)],
    }


def gz(doc):
    return gzip.compress(json.dumps(doc).encode())


def test_decode_run_validates_and_rejects_garbage():
    assert decode_run(gz(make_run()))["id"] == "abcdef123456"
    with pytest.raises(InvalidRun):
        decode_run(b"not gzip")
    with pytest.raises(InvalidRun):
        decode_run(gzip.compress(b"[1,2]"))
    with pytest.raises(InvalidRun):
        decode_run(gz({**make_run(), "id": "../../etc"}))
    with pytest.raises(InvalidRun):
        decode_run(gz({**make_run(), "env": {"key": "Pong"}}))
    with pytest.raises(InvalidRun):
        decode_run(gz({**make_run(), "iterations": []}))


def test_decode_run_rejects_zip_bomb(monkeypatch):
    import backend.storage as st
    monkeypatch.setattr(st, "MAX_RUN_BYTES", 1000)
    with pytest.raises(InvalidRun):
        decode_run(gzip.compress(b"{" + b" " * 5000 + b"}"))


def test_summary_is_computed_server_side():
    s = summarize_run(make_run())
    assert s["id"] == "abcdef123456" and s["env_key"] == "FrozenLake" and s["n_iterations"] == 2 and s["v_start"] == 0.5


def test_local_repository_crud_and_user_isolation(tmp_path):
    repo = LocalRunRepository(tmp_path)
    doc = make_run()
    repo.save("alice", doc, gz(doc))
    assert [s["id"] for s in repo.list("alice")] == ["abcdef123456"]
    assert repo.list("bob") == []
    assert json.loads(gzip.decompress(repo.get_compressed("alice", "abcdef123456")))["id"] == "abcdef123456"
    with pytest.raises(KeyError):
        repo.get_compressed("bob", "abcdef123456")
    repo.delete("alice", "abcdef123456")
    assert repo.list("alice") == []
    with pytest.raises(KeyError):
        repo.delete("alice", "abcdef123456")


def test_local_repository_rejects_bad_ids_and_users(tmp_path):
    repo = LocalRunRepository(tmp_path)
    with pytest.raises(KeyError):
        repo.get_compressed("alice", "../../x")
    with pytest.raises(KeyError):
        repo.list("../evil")


def test_local_repository_quota(tmp_path, monkeypatch):
    import backend.storage as st
    monkeypatch.setattr(st, "MAX_RUNS_PER_USER", 2)
    repo = LocalRunRepository(tmp_path)
    for i in range(2):
        d = make_run(f"{i:012x}")
        repo.save("alice", d, gz(d))
    d = make_run("ffffffffffff")
    with pytest.raises(st.QuotaExceeded):
        repo.save("alice", d, gz(d))
    assert MAX_RUNS_PER_USER >= 50


# ---------------------------------------------------------------- Vercel Blob (fake HTTP)

class FakeBlobApi:
    """In-memory stand-in for the Vercel Blob REST API (private store)."""

    def __init__(self):
        self.files = {}
        self.calls = []

    def __call__(self, method, url, headers, body):
        from urllib.parse import parse_qs, unquote, urlparse
        from backend.blob_store import HttpResponse
        self.calls.append((method, url))
        u = urlparse(url)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if "private.blob.vercel-storage.com" in u.netloc:
            assert headers["authorization"].startswith("Bearer vercel_blob_rw_")
            path = unquote(u.path.lstrip("/"))
            return HttpResponse(200, self.files[path]) if path in self.files else \
                HttpResponse(404, b'{"error":{"code":"not_found"}}')
        assert headers["x-api-version"] == "12"
        if method == "PUT":
            assert headers["x-vercel-blob-access"] == "private"
            self.files[q["pathname"]] = body
            return HttpResponse(200, b"{}")
        if method == "GET":
            blobs = [{"pathname": p, "url": "x"} for p in sorted(self.files) if p.startswith(q["prefix"])]
            return HttpResponse(200, json.dumps({"blobs": blobs, "hasMore": False}).encode())
        if method == "POST" and u.path.endswith("/delete"):
            for full in json.loads(body)["urls"]:
                self.files.pop(unquote(urlparse(full).path.lstrip("/")), None)
            return HttpResponse(200, b"{}")
        return HttpResponse(400, b"{}")


TOKEN = "vercel_blob_rw_store123_secretpart"


def test_blob_repository_crud_isolation_and_quota(monkeypatch):
    import backend.storage as st
    from backend.blob_store import BlobRunRepository
    api = FakeBlobApi()
    repo = BlobRunRepository(TOKEN, transport=api)
    doc = make_run()
    repo.save("alice", doc, gz(doc))
    assert "runs/alice/abcdef123456.json.gz" in api.files
    assert [s["id"] for s in repo.list("alice")] == ["abcdef123456"] and repo.list("bob") == []
    assert json.loads(gzip.decompress(repo.get_compressed("alice", "abcdef123456")))["id"] == "abcdef123456"
    with pytest.raises(KeyError):
        repo.get_compressed("bob", "abcdef123456")
    monkeypatch.setattr(st, "MAX_RUNS_PER_USER", 1)
    other = make_run("bbbbbbbbbbbb")
    with pytest.raises(st.QuotaExceeded):
        repo.save("alice", other, gz(other))
    repo.save("alice", doc, gz(doc))  # overwriting an existing run is not a new run
    repo.delete("alice", "abcdef123456")
    assert repo.list("alice") == [] and not api.files
    with pytest.raises(KeyError):
        repo.delete("alice", "abcdef123456")


def test_blob_repository_rejects_bad_token_and_surfaces_errors():
    from backend.blob_store import BlobError, BlobRunRepository, HttpResponse
    with pytest.raises(ValueError):
        BlobRunRepository("not-a-token")
    repo = BlobRunRepository(TOKEN, transport=lambda *a: HttpResponse(500, b'{"error":{"code":"internal"}}'))
    with pytest.raises(BlobError):
        repo.list("alice")


def test_ids_with_trailing_newline_are_rejected(tmp_path):
    repo = LocalRunRepository(tmp_path)
    with pytest.raises(KeyError):
        repo.get_compressed("alice", "abcdef123456\n")
    with pytest.raises(KeyError):
        repo.list("alice\n")
    with pytest.raises(InvalidRun):
        decode_run(gz({**make_run(), "id": "abcdef123456\n"}))


def test_deeply_nested_or_malformed_runs_are_invalid():
    deep = '{"id":"abcdef123456","x":' + "[" * 100_000 + "]" * 100_000 + "}"
    with pytest.raises(InvalidRun):
        decode_run(gzip.compress(deep.encode()))
    with pytest.raises(InvalidRun):
        summarize_run({**make_run(), "created_at": "yesterday"})
