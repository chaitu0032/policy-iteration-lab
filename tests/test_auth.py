import time

import pytest

from backend.auth import (
    AuthConfig,
    LoginLimiter,
    hash_password,
    issue_token,
    load_config,
    parse_users,
    read_token,
    verify_password,
)

SECRET = "x" * 48


def test_password_hash_roundtrip_and_rejects_wrong():
    h = hash_password("correct horse", iterations=1000)
    assert h.startswith("pbkdf2_sha256$1000$")
    assert verify_password("correct horse", h)
    assert not verify_password("wrong", h)
    assert not verify_password("correct horse", "garbage")


def test_parse_users():
    a, b = hash_password("pw1", iterations=1000), hash_password("pw2", iterations=1000)
    users = parse_users(f"alice:{a}, bob:{b}")
    assert set(users) == {"alice", "bob"}
    with pytest.raises(ValueError):
        parse_users("alice")
    with pytest.raises(ValueError):
        parse_users(f"bad name!:{a}")


def test_token_roundtrip_tamper_and_expiry():
    tok = issue_token("alice", SECRET, ttl_s=60)
    assert read_token(tok, SECRET)["u"] == "alice"
    assert read_token(tok + "x", SECRET) is None
    assert read_token(tok, "y" * 48) is None
    payload, sig = tok.split(".")
    assert read_token(payload[:-2] + "AA." + sig, SECRET) is None
    expired = issue_token("alice", SECRET, ttl_s=-1)
    assert read_token(expired, SECRET) is None
    assert read_token("", SECRET) is None and read_token("a.b.c", SECRET) is None


def test_login_limiter_blocks_after_failures():
    lim = LoginLimiter(max_failures=3, window_s=60)
    for _ in range(3):
        assert lim.allowed("alice")
        lim.failed("alice")
    assert not lim.allowed("alice")
    assert lim.allowed("bob")
    lim.succeeded("alice")
    assert lim.allowed("alice")


def test_login_limiter_window_expires(monkeypatch):
    lim = LoginLimiter(max_failures=1, window_s=10)
    now = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    lim.failed("a")
    assert not lim.allowed("a")
    now[0] += 11
    assert lim.allowed("a")


def test_load_config(monkeypatch):
    for k in ("PI_LAB_USERS", "PI_LAB_SECRET", "VERCEL"):
        monkeypatch.delenv(k, raising=False)
    assert load_config().enabled is False  # local dev without accounts
    monkeypatch.setenv("VERCEL", "1")
    with pytest.raises(RuntimeError):
        load_config()  # fail closed in production without accounts
    monkeypatch.setenv("PI_LAB_USERS", f"alice:{hash_password('pw', iterations=1000)}")
    with pytest.raises(RuntimeError):
        load_config()  # secret missing
    monkeypatch.setenv("PI_LAB_SECRET", "short")
    with pytest.raises(RuntimeError):
        load_config()  # secret too short
    monkeypatch.setenv("PI_LAB_SECRET", SECRET)
    cfg = load_config()
    assert isinstance(cfg, AuthConfig) and cfg.enabled and "alice" in cfg.users


def test_try_acquire_reserves_atomically_against_bursts():
    import threading
    lim = LoginLimiter(max_failures=5, window_s=60)
    results = []
    threads = [threading.Thread(target=lambda: results.append(lim.try_acquire("k"))) for _ in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results.count(True) == 5  # never more attempts than the limit, even in parallel


def test_limiter_memory_is_bounded():
    lim = LoginLimiter(max_failures=1, window_s=60, max_keys=10)
    for i in range(100):
        lim.try_acquire(f"user{i}")
    assert len(lim._failures) <= 10
    assert lim.allowed("never-seen") and "never-seen" not in lim._failures


def test_username_pattern_rejects_trailing_newline():
    with pytest.raises(ValueError):
        parse_users(f"alice\n:{hash_password('x', iterations=1000)}")
