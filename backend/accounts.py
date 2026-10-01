"""Login/session endpoints, the auth + CSRF middleware, and per-user run storage endpoints."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from .auth import (
    DUMMY_HASH,
    LOCAL_USER,
    SESSION_TTL_S,
    AuthConfig,
    LoginLimiter,
    credential_version,
    issue_token,
    load_config,
    read_token,
    verify_password,
)
from .blob_store import BlobError, BlobRunRepository
from .storage import (
    MAX_UPLOAD_BYTES,
    InvalidRun,
    LocalRunRepository,
    QuotaExceeded,
    RunRepository,
    decode_run,
    gzip_json,
    summarize_run,
)

logger = logging.getLogger("pi_lab.accounts")

COOKIE = "pi_session"
CSRF_HEADER = "x-requested-with"
CSRF_VALUE = "pi-lab"
PUBLIC_PATHS = {"/api/health", "/api/session", "/api/auth/login", "/api/auth/logout"}
UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
LOGIN_FAILED = "Wrong username or password"
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "runs"
LOCAL_HOSTS = {"localhost", "127.0.0.1", "[::1]", "testserver"}


@dataclass(frozen=True)
class Security:
    config: AuthConfig | None
    config_error: str | None
    limiter: LoginLimiter
    repo: RunRepository | None
    storage: str  # "server" | "browser"


def _build_repo() -> tuple[RunRepository | None, str]:
    token = os.environ.get("BLOB_READ_WRITE_TOKEN")
    if token:
        return BlobRunRepository(token), "server"
    if os.environ.get("VERCEL"):
        logger.warning("No BLOB_READ_WRITE_TOKEN on Vercel: runs stay in each browser (IndexedDB)")
        return None, "browser"
    return LocalRunRepository(Path(os.environ.get("PI_LAB_DATA_DIR", DEFAULT_DATA_DIR))), "server"


def build_security() -> Security:
    try:
        config, error = load_config(), None
    except (RuntimeError, ValueError) as exc:  # misconfiguration: refuse every request, explain why
        config, error = None, str(exc)
        logger.error("Auth misconfigured: %s", exc)
    repo, storage = _build_repo()
    return Security(config, error, LoginLimiter(), repo, storage)


def _user_from_request(sec: Security, request: Request) -> str | None:
    if not sec.config.enabled:
        return LOCAL_USER
    body = read_token(request.cookies.get(COOKIE, ""), sec.config.secret)
    if not body or body.get("u") not in sec.config.users:
        return None
    if body.get("v") != credential_version(sec.config.users[body["u"]]):
        return None  # password was rotated after this session was issued
    return body["u"]


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-real-ip") or request.headers.get("x-forwarded-for", "").split(",")[0]
    return forwarded.strip() or (request.client.host if request.client else "unknown")


def install(app: FastAPI, sec: Security) -> None:
    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        path = request.url.path
        if not path.startswith("/api/"):
            return await call_next(request)
        if sec.config is None:  # details are in the server log only
            return JSONResponse({"detail": "Server is not configured. Check the deployment's environment variables."},
                                status_code=503)
        if not sec.config.enabled and request.url.hostname not in LOCAL_HOSTS:
            # Auth is off only for local development: refuse other hosts (DNS rebinding, exposed ports).
            return JSONResponse({"detail": "Accounts are not configured on this server"}, status_code=403)
        if request.method in UNSAFE_METHODS and request.headers.get(CSRF_HEADER) != CSRF_VALUE:
            return JSONResponse({"detail": "Missing X-Requested-With header"}, status_code=403)
        user = _user_from_request(sec, request)
        if user is None and path not in PUBLIC_PATHS:
            return JSONResponse({"detail": "Please sign in"}, status_code=401)
        request.state.user = user
        return await call_next(request)

    app.include_router(_router(sec))


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


def _router(sec: Security) -> APIRouter:
    r = APIRouter(prefix="/api")

    def repo() -> RunRepository:
        if sec.repo is None:
            raise HTTPException(status_code=409, detail="Server-side run storage is not configured")
        return sec.repo

    @r.get("/session")
    def session(request: Request) -> dict[str, Any]:
        return {"auth": sec.config.enabled, "user": request.state.user, "storage": sec.storage}

    @r.post("/auth/login")
    def login(req: LoginRequest, request: Request) -> Response:
        if not sec.config.enabled:
            return JSONResponse({"user": LOCAL_USER})
        name = req.username.strip().lower()
        key = f"{name}|{_client_ip(request)}"
        if not sec.limiter.try_acquire(key):  # reserves the attempt before the (slow) password check
            raise HTTPException(status_code=429, detail="Too many failed attempts. Wait 10 minutes and try again.")
        encoded = sec.config.users.get(name)
        ok = verify_password(req.password, encoded or DUMMY_HASH)  # same cost for unknown names
        if encoded is None or not ok:
            raise HTTPException(status_code=401, detail=LOGIN_FAILED)
        sec.limiter.succeeded(key)
        res = JSONResponse({"user": name})
        secure = bool(os.environ.get("VERCEL")) or "https" in request.headers.get("x-forwarded-proto", request.url.scheme)
        token = issue_token(name, sec.config.secret, version=credential_version(encoded))
        res.set_cookie(COOKIE, token, max_age=SESSION_TTL_S, httponly=True, samesite="lax", secure=secure, path="/")
        logger.info("login ok user=%s", name)
        return res

    @r.post("/auth/logout")
    def logout() -> Response:
        res = JSONResponse({"ok": True})
        res.delete_cookie(COOKIE, path="/")
        return res

    @r.get("/runs")
    def list_runs(request: Request) -> list[dict[str, Any]]:
        try:
            return repo().list(request.state.user)
        except BlobError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @r.get("/runs/{run_id}")
    def get_run(run_id: str, request: Request) -> Response:
        try:
            data = repo().get_compressed(request.state.user, run_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Run not found") from None
        except BlobError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return Response(data, media_type="application/gzip", headers={"Cache-Control": "no-store"})

    @r.put("/runs/{run_id}")
    async def put_run(run_id: str, request: Request) -> dict[str, Any]:
        body = await request.body()
        if len(body) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Run is too large to save")
        try:
            doc = decode_run(body)
            summarize_run(doc)  # malformed fields -> 422, not 500
        except InvalidRun as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if doc["id"] != run_id:
            raise HTTPException(status_code=422, detail="Run id in the URL and the document differ")
        try:
            # Store what was validated (re-encoded), never the raw upload bytes.
            return repo().save(request.state.user, doc, gzip_json(doc))
        except QuotaExceeded as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except BlobError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @r.delete("/runs/{run_id}")
    def delete_run(run_id: str, request: Request) -> dict[str, Any]:
        try:
            repo().delete(request.state.user, run_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Run not found") from None
        except BlobError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return {"deleted": run_id}

    return r
