"""Accounts + stateless sessions (safe on serverless: any instance can verify any request).

* Users come from the env var ``PI_LAB_USERS`` = "name:pbkdf2_sha256$iters$salt$hash, ..." — only
  salted PBKDF2 hashes are ever stored; generate them with ``scripts/create_users.py``.
* A session is an HMAC-SHA256-signed token (``PI_LAB_SECRET``) in an HttpOnly cookie, so the same
  user can be logged in from several browsers at once and no server memory is needed.
* Locally, with no users configured, auth is disabled (single "local" user). On Vercel the app
  refuses to start without accounts (fail closed).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

PBKDF2_ITERATIONS = 390_000
MIN_SECRET_LENGTH = 32
SESSION_TTL_S = 3 * 24 * 3600
USERNAME_PATTERN = re.compile(r"[a-z0-9_-]{2,32}")  # always used with fullmatch
LOCAL_USER = "local"


# ---------------------------------------------------------------- passwords

def hash_password(password: str, iterations: int = PBKDF2_ITERATIONS) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return f"pbkdf2_sha256${iterations}${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algo, iterations, salt, expected = encoded.split("$")
        if algo != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac("sha256", password.encode(), _unb64(salt), int(iterations))
        return hmac.compare_digest(digest, _unb64(expected))
    except (ValueError, TypeError):
        return False


def parse_users(raw: str) -> dict[str, str]:
    users: dict[str, str] = {}
    for entry in filter(None, (e.strip() for e in raw.split(","))):
        name, sep, encoded = entry.partition(":")
        if not sep or not encoded:
            raise ValueError("PI_LAB_USERS entries must look like name:pbkdf2_sha256$...")
        if not USERNAME_PATTERN.fullmatch(name):
            raise ValueError(f"Invalid username '{name}' (use a-z, 0-9, _ or -, 2-32 chars)")
        users[name] = encoded
    return users


# ---------------------------------------------------------------- tokens

def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _sign(payload: str, secret: str) -> str:
    return _b64(hmac.new(secret.encode(), payload.encode(), hashlib.sha256).digest())


def credential_version(encoded_hash: str) -> str:
    """Changes whenever the user's password hash changes -> rotating a password revokes old sessions."""
    return hashlib.sha256(encoded_hash.encode()).hexdigest()[:16]


# Verified against when the username is unknown, so login timing does not reveal valid names.
DUMMY_HASH = hash_password(secrets.token_urlsafe(16), iterations=PBKDF2_ITERATIONS)


def issue_token(username: str, secret: str, ttl_s: int = SESSION_TTL_S, version: str = "") -> str:
    body = {"u": username, "v": version, "exp": int(time.time()) + ttl_s, "sid": secrets.token_hex(8)}
    payload = _b64(json.dumps(body, separators=(",", ":")).encode())
    return f"{payload}.{_sign(payload, secret)}"


def read_token(token: str, secret: str) -> dict | None:
    """Return the token body if the signature is valid and it has not expired, else None."""
    try:
        payload, sig = token.split(".")
        if not hmac.compare_digest(sig, _sign(payload, secret)):
            return None
        body = json.loads(_unb64(payload))
        if not isinstance(body, dict) or int(body.get("exp", 0)) < time.time():
            return None
        return body
    except (ValueError, TypeError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------- login rate limit

@dataclass
class LoginLimiter:
    """Sliding-window limit on login attempts per key (username + client IP).

    ``try_acquire`` reserves a slot atomically *before* the password is checked, so parallel bursts
    cannot slip past; a successful login clears the key. Per process: on serverless each warm
    instance keeps its own window (PBKDF2 cost and strong passwords do the rest).
    """

    max_failures: int = 5
    window_s: float = 600.0
    max_keys: int = 10_000
    _failures: dict[str, deque] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def _pruned(self, key: str) -> deque | None:
        q = self._failures.get(key)
        if q is None:
            return None
        cutoff = time.monotonic() - self.window_s
        while q and q[0] < cutoff:
            q.popleft()
        if not q:
            del self._failures[key]
            return None
        return q

    def allowed(self, key: str) -> bool:
        with self._lock:
            q = self._pruned(key)
            return q is None or len(q) < self.max_failures

    def try_acquire(self, key: str) -> bool:
        """Atomically check the limit and count this attempt as a failure until it succeeds."""
        with self._lock:
            q = self._pruned(key)
            if q is not None and len(q) >= self.max_failures:
                return False
            if q is None:
                if len(self._failures) >= self.max_keys:
                    self._failures.pop(next(iter(self._failures)))  # bound memory
                q = self._failures[key] = deque()
            q.append(time.monotonic())
            return True

    def failed(self, key: str) -> None:
        self.try_acquire(key)

    def succeeded(self, key: str) -> None:
        with self._lock:
            self._failures.pop(key, None)


# ---------------------------------------------------------------- config

@dataclass(frozen=True)
class AuthConfig:
    enabled: bool
    users: dict[str, str]
    secret: str


def load_config() -> AuthConfig:
    raw = os.environ.get("PI_LAB_USERS", "").strip()
    secret = os.environ.get("PI_LAB_SECRET", "")
    on_vercel = bool(os.environ.get("VERCEL"))
    if not raw:
        if on_vercel:
            raise RuntimeError("PI_LAB_USERS is not set: refusing to run without accounts on Vercel")
        return AuthConfig(enabled=False, users={}, secret="")  # local dev; Host is restricted to localhost
    if len(secret) < MIN_SECRET_LENGTH:
        raise RuntimeError(f"PI_LAB_SECRET must be set to at least {MIN_SECRET_LENGTH} random characters")
    users = parse_users(raw)
    if not users:
        raise RuntimeError("PI_LAB_USERS contains no users")
    return AuthConfig(enabled=True, users=users, secret=secret)
