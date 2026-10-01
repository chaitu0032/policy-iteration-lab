"""Vercel Blob run repository over the REST API used by @vercel/blob (API version 12).

Use a *private* Blob store: uploads send ``x-vercel-blob-access: private`` and downloads need the
bearer token, so run files are never publicly reachable. Layout per user:
    runs/<user>/<id>.json.gz        gzip-compressed run document
    runs/<user>/<id>.summary.json   small server-computed summary (for fast listing)
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

from . import storage
from .storage import QuotaExceeded, _check_id, _check_user, summarize_run

logger = logging.getLogger(__name__)

API_URL = os.environ.get("VERCEL_BLOB_API_URL", "https://vercel.com/api/blob")
API_VERSION = "12"
LIST_LIMIT = 1000
SUMMARY_FETCH_WORKERS = 8
TIMEOUT_S = 20


@dataclass(frozen=True)
class HttpResponse:
    status: int
    body: bytes


Transport = Callable[[str, str, dict[str, str], bytes | None], HttpResponse]


def urllib_transport(method: str, url: str, headers: dict[str, str], body: bytes | None) -> HttpResponse:
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as res:
            return HttpResponse(res.status, res.read())
    except urllib.error.HTTPError as exc:
        return HttpResponse(exc.code, exc.read())


class BlobError(RuntimeError):
    pass


class BlobRunRepository:
    def __init__(self, token: str, transport: Transport = urllib_transport) -> None:
        parts = token.split("_")
        if len(parts) < 5 or not token.startswith("vercel_blob_rw_"):
            raise ValueError("BLOB_READ_WRITE_TOKEN does not look like a Vercel Blob read/write token")
        self._token = token
        self._store_id = parts[3]
        self._http = transport

    # ---------------------------------------------------------------- low level
    def _api_headers(self) -> dict[str, str]:
        return {
            "authorization": f"Bearer {self._token}",
            "x-api-version": API_VERSION,
            "x-vercel-blob-store-id": self._store_id,
            "x-api-blob-request-id": f"{self._store_id}:{int(time.time() * 1000)}:{secrets.token_hex(4)}",
        }

    def _blob_url(self, pathname: str) -> str:
        return f"https://{self._store_id}.private.blob.vercel-storage.com/{urllib.parse.quote(pathname)}"

    def _check(self, res: HttpResponse, what: str) -> HttpResponse:
        if 200 <= res.status < 300:
            return res
        try:
            code = json.loads(res.body).get("error", {}).get("code", "")
        except (ValueError, AttributeError):
            code = ""
        if res.status == 404 or code in ("not_found", "blob_not_found"):
            raise KeyError(what)
        logger.error("Vercel Blob %s failed: HTTP %s %s", what, res.status, res.body[:300])
        raise BlobError(f"Run storage error ({res.status}) while trying to {what}")

    def _put(self, pathname: str, body: bytes, content_type: str) -> None:
        headers = {
            **self._api_headers(),
            "x-vercel-blob-access": "private",
            "x-content-type": content_type,
            "x-add-random-suffix": "0",
            "x-allow-overwrite": "1",
            "x-cache-control-max-age": "60",
        }
        url = f"{API_URL}/?{urllib.parse.urlencode({'pathname': pathname})}"
        self._check(self._http("PUT", url, headers, body), f"upload {pathname}")

    def _download(self, pathname: str) -> bytes:
        url = self._blob_url(pathname) + "?cache=0"  # skip the CDN cache: reads must see the latest write
        res = self._http("GET", url, {"authorization": f"Bearer {self._token}"}, None)
        return self._check(res, f"read {pathname}").body

    def _list(self, prefix: str) -> list[dict[str, Any]]:
        blobs: list[dict[str, Any]] = []
        cursor = None
        while True:
            params = {"prefix": prefix, "limit": str(LIST_LIMIT), **({"cursor": cursor} if cursor else {})}
            res = self._check(self._http("GET", f"{API_URL}?{urllib.parse.urlencode(params)}",
                                         self._api_headers(), None), f"list {prefix}")
            page = json.loads(res.body)
            blobs.extend(page.get("blobs", []))
            cursor = page.get("cursor")
            if not page.get("hasMore") or not cursor:
                return blobs

    # ---------------------------------------------------------------- repository API
    def list(self, user: str) -> list[dict[str, Any]]:
        prefix = f"runs/{_check_user(user)}/"
        names = [b["pathname"] for b in self._list(prefix) if b.get("pathname", "").endswith(".summary.json")]
        with ThreadPoolExecutor(max_workers=SUMMARY_FETCH_WORKERS) as pool:
            summaries = list(pool.map(lambda n: json.loads(self._download(n)), names))
        return sorted(summaries, key=lambda s: s["created_at"], reverse=True)

    def get_compressed(self, user: str, run_id: str) -> bytes:
        return self._download(f"runs/{_check_user(user)}/{_check_id(run_id)}.json.gz")

    def save(self, user: str, doc: dict[str, Any], compressed: bytes) -> dict[str, Any]:
        base = f"runs/{_check_user(user)}/{_check_id(doc['id'])}"
        existing = {b["pathname"] for b in self._list(f"runs/{user}/")}
        n_runs = sum(1 for p in existing if p.endswith(".summary.json"))
        if f"{base}.summary.json" not in existing and n_runs >= storage.MAX_RUNS_PER_USER:
            raise QuotaExceeded(f"At most {storage.MAX_RUNS_PER_USER} saved runs per user; delete some first")
        summary = summarize_run(doc)
        self._put(f"{base}.json.gz", compressed, "application/gzip")
        self._put(f"{base}.summary.json", json.dumps(summary).encode(), "application/json")
        return summary

    def delete(self, user: str, run_id: str) -> None:
        base = f"runs/{_check_user(user)}/{_check_id(run_id)}"
        existing = {b["pathname"] for b in self._list(f"runs/{user}/")}
        if f"{base}.json.gz" not in existing:
            raise KeyError(run_id)
        urls = [self._blob_url(f"{base}.json.gz"), self._blob_url(f"{base}.summary.json")]
        res = self._http("POST", f"{API_URL}/delete", {**self._api_headers(), "content-type": "application/json"},
                         json.dumps({"urls": urls}).encode())
        self._check(res, f"delete {base}")
