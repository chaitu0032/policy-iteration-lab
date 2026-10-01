import gzip
import importlib
import json

import pytest
from fastapi.testclient import TestClient

from backend.auth import hash_password

SECRET = "s" * 48
HDR = {"X-Requested-With": "pi-lab"}


@pytest.fixture()
def make_client(tmp_path, monkeypatch):
    def factory():
        users = ",".join(f"{u}:{hash_password(u + '-pw', iterations=1000)}" for u in ("alice", "bob"))
        monkeypatch.setenv("PI_LAB_USERS", users)
        monkeypatch.setenv("PI_LAB_SECRET", SECRET)
        monkeypatch.setenv("PI_LAB_DATA_DIR", str(tmp_path))
        monkeypatch.delenv("BLOB_READ_WRITE_TOKEN", raising=False)
        monkeypatch.delenv("VERCEL", raising=False)
        import backend.accounts
        import backend.app
        importlib.reload(backend.accounts)
        importlib.reload(backend.app)
        return TestClient(backend.app.app)
    yield factory
    monkeypatch.delenv("PI_LAB_USERS")
    import backend.accounts
    import backend.app
    importlib.reload(backend.accounts)
    importlib.reload(backend.app)


def login(client, user="alice"):
    res = client.post("/api/auth/login", json={"username": user, "password": f"{user}-pw"}, headers=HDR)
    assert res.status_code == 200, res.text
    return res


def run_doc(run_id="abcdef123456"):
    return {"id": run_id, "created_at": 2.0, "status": "converged", "optimal_index": 0,
            "env": {"key": "FrozenLake", "title": "Frozen Lake", "options": {}}, "layout": {}, "config": {"gamma": 0.9},
            "iterations": [{"index": 0, "policy": [0] * 16, "V": [0.0] * 16, "stats": {"v_start": 0.1}}]}


def put_run(client, doc):
    return client.put(f"/api/runs/{doc['id']}", content=gzip.compress(json.dumps(doc).encode()),
                      headers={**HDR, "Content-Type": "application/gzip"})


def test_session_requires_login_and_protects_compute(make_client):
    c = make_client()
    assert c.get("/api/session").json() == {"auth": True, "user": None, "storage": "server"}
    assert c.get("/api/health").status_code == 200
    assert c.get("/api/envs").status_code == 401
    assert c.post("/api/pi/step", json={"env_key": "FrozenLake"}, headers=HDR).status_code == 401


def test_login_logout_and_cookie_flags(make_client):
    c = make_client()
    res = login(c)
    cookie = res.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=lax" in cookie
    assert c.get("/api/session").json()["user"] == "alice"
    assert c.get("/api/envs").status_code == 200
    assert c.post("/api/auth/logout", headers=HDR).status_code == 200
    assert c.get("/api/envs").status_code == 401


def test_wrong_password_is_generic_and_rate_limited(make_client):
    c = make_client()
    for _ in range(5):
        res = c.post("/api/auth/login", json={"username": "alice", "password": "nope"}, headers=HDR)
        assert res.status_code == 401 and res.json()["detail"] == "Wrong username or password"
    unknown = c.post("/api/auth/login", json={"username": "mallory", "password": "x"}, headers=HDR)
    assert unknown.json()["detail"] == "Wrong username or password"
    blocked = c.post("/api/auth/login", json={"username": "alice", "password": "alice-pw"}, headers=HDR)
    assert blocked.status_code == 429


def test_csrf_header_required_for_writes(make_client):
    c = make_client()
    login(c)
    assert c.post("/api/auth/logout").status_code == 403
    assert c.post("/api/pi/step", json={"env_key": "FrozenLake"}).status_code == 403


def test_tampered_cookie_rejected(make_client):
    c = make_client()
    login(c)
    tok = c.cookies.get("pi_session")
    c.cookies.set("pi_session", tok[:-3] + "AAA")
    assert c.get("/api/envs").status_code == 401


def test_runs_are_per_user_and_shared_across_browsers(make_client):
    alice_laptop = make_client()
    login(alice_laptop)
    assert put_run(alice_laptop, run_doc()).status_code == 200
    import backend.app
    alice_phone = TestClient(backend.app.app)  # second browser, separate cookie jar
    login(alice_phone)
    assert [r["id"] for r in alice_phone.get("/api/runs").json()] == ["abcdef123456"]
    blob = alice_phone.get("/api/runs/abcdef123456")
    assert blob.headers["content-type"] == "application/gzip"
    assert json.loads(gzip.decompress(blob.content))["id"] == "abcdef123456"
    bob = TestClient(backend.app.app)
    login(bob, "bob")
    assert bob.get("/api/runs").json() == []
    assert bob.get("/api/runs/abcdef123456").status_code == 404
    assert bob.delete("/api/runs/abcdef123456", headers=HDR).status_code == 404
    assert alice_phone.delete("/api/runs/abcdef123456", headers=HDR).status_code == 200
    assert alice_laptop.get("/api/runs").json() == []


def test_run_upload_validation(make_client):
    c = make_client()
    login(c)
    assert put_run(c, {**run_doc(), "env": {"key": "Pong"}}).status_code == 422
    mismatch = put_run(c, run_doc("bbbbbbbbbbbb")).status_code
    assert mismatch == 200
    wrong_path = c.put("/api/runs/cccccccccccc", content=gzip.compress(json.dumps(run_doc()).encode()),
                       headers={**HDR, "Content-Type": "application/gzip"})
    assert wrong_path.status_code == 422
    garbage = c.put("/api/runs/abcdef123456", content=b"xx", headers={**HDR, "Content-Type": "application/gzip"})
    assert garbage.status_code == 422


def test_local_dev_without_accounts_is_open(monkeypatch, tmp_path):
    monkeypatch.delenv("PI_LAB_USERS", raising=False)
    monkeypatch.delenv("VERCEL", raising=False)
    monkeypatch.setenv("PI_LAB_DATA_DIR", str(tmp_path))
    import backend.accounts
    import backend.app
    importlib.reload(backend.accounts)
    importlib.reload(backend.app)
    c = TestClient(backend.app.app)
    assert c.get("/api/session").json() == {"auth": False, "user": "local", "storage": "server"}
    assert c.get("/api/envs").status_code == 200
    assert put_run(c, run_doc()).status_code == 200
    assert len(c.get("/api/runs").json()) == 1


def test_password_rotation_revokes_existing_sessions(make_client, monkeypatch):
    c = make_client()
    login(c)
    cookie = c.cookies.get("pi_session")
    users = ",".join(f"{u}:{hash_password(u + '-NEW', iterations=1000)}" for u in ("alice", "bob"))
    monkeypatch.setenv("PI_LAB_USERS", users)
    import backend.accounts
    import backend.app
    importlib.reload(backend.accounts)
    importlib.reload(backend.app)
    c2 = TestClient(backend.app.app)
    c2.cookies.set("pi_session", cookie)
    assert c2.get("/api/envs").status_code == 401


def test_stored_run_is_re_encoded_not_raw(make_client, tmp_path):
    c = make_client()
    login(c)
    raw = gzip.compress(json.dumps(run_doc()).encode()) + b"TRAILING-JUNK"
    res = c.put("/api/runs/abcdef123456", content=raw, headers={**HDR, "Content-Type": "application/gzip"})
    assert res.status_code == 200
    stored = (tmp_path / "alice" / "abcdef123456.json.gz").read_bytes()
    assert b"TRAILING-JUNK" not in stored and json.loads(gzip.decompress(stored))["id"] == "abcdef123456"


def test_misconfiguration_message_is_generic(monkeypatch, tmp_path):
    monkeypatch.setenv("VERCEL", "1")
    monkeypatch.delenv("PI_LAB_USERS", raising=False)
    monkeypatch.setenv("PI_LAB_DATA_DIR", str(tmp_path))
    import backend.accounts
    import backend.app
    importlib.reload(backend.accounts)
    importlib.reload(backend.app)
    res = TestClient(backend.app.app).get("/api/envs")
    assert res.status_code == 503 and "PI_LAB_USERS" not in res.text
    monkeypatch.delenv("VERCEL")
    importlib.reload(backend.accounts)
    importlib.reload(backend.app)


def test_open_local_mode_refuses_non_local_hosts(monkeypatch, tmp_path):
    monkeypatch.delenv("PI_LAB_USERS", raising=False)
    monkeypatch.delenv("VERCEL", raising=False)
    monkeypatch.setenv("PI_LAB_DATA_DIR", str(tmp_path))
    import backend.accounts
    import backend.app
    importlib.reload(backend.accounts)
    importlib.reload(backend.app)
    c = TestClient(backend.app.app)
    assert c.get("/api/envs", headers={"Host": "attacker.example"}).status_code == 403
    assert c.get("/api/envs").status_code == 200


def test_api_docs_disabled(make_client):
    c = make_client()
    assert c.get("/docs").status_code == 404 and c.get("/openapi.json").status_code == 404
