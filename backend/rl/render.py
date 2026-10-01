"""Real gymnasium frames: env.render() in rgb_array mode for any state (needs pygame)."""

from __future__ import annotations

import base64
import importlib.util
import logging
import multiprocessing
import os
import threading
import zlib
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from concurrent.futures.process import BrokenProcessPool
from typing import Any

import numpy as np

from .environments import TAXI, make_env

# Headless pygame (servers / serverless functions have no display or audio device).
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

MOVE_ACTIONS = (0, 1, 2, 3)


logger = logging.getLogger(__name__)


class RenderUnavailable(RuntimeError):
    """pygame is not installed in this deployment."""


class RenderCrashed(RuntimeError):
    """The isolated render process died or timed out (the server itself is unaffected)."""


def _png_bytes(frame: np.ndarray) -> bytes:
    """Encode an RGB uint8 array as PNG with only the standard library."""
    height, width, _ = frame.shape
    raw = b"".join(b"\x00" + frame[y].tobytes() for y in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return len(data).to_bytes(4, "big") + body + zlib.crc32(body).to_bytes(4, "big")

    header = width.to_bytes(4, "big") + height.to_bytes(4, "big") + bytes([8, 2, 0, 0, 0])
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")


def _render_in_worker(key: str, options: dict[str, Any], state: int, last_action: int | None) -> str:
    """Runs inside the dedicated render process (pygame on that process's main thread)."""
    import pygame  # noqa: F401  (ImportError is reported back to the server)

    env = make_env(key, options)
    env.unwrapped.render_mode = "rgb_array"
    try:
        env.reset(seed=0)
        u = env.unwrapped
        u.s = int(state)
        u.lastaction = last_action
        if key == TAXI and last_action in MOVE_ACTIONS:
            u.taxi_orientation = last_action
        frame = np.asarray(u.render(), dtype=np.uint8)
        return "data:image/png;base64," + base64.b64encode(_png_bytes(frame)).decode()
    finally:
        env.close()


# pygame/SDL is not thread-safe, and the API serves requests from a thread pool: overlapping calls
# segfaulted the whole server. All rendering therefore happens in ONE separate process, one frame
# at a time. If that process dies, only it is lost; it is restarted on the next request.
RENDER_TIMEOUT_S = 20
_pool: ProcessPoolExecutor | None = None
_pool_lock = threading.Lock()


def _get_pool() -> ProcessPoolExecutor:
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
        return _pool


def _reset_pool() -> None:
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.shutdown(wait=False, cancel_futures=True)
        _pool = None


def kill_worker_for_tests() -> None:
    """Terminate the render process abruptly (simulates a native crash)."""
    pool = _get_pool()
    pool.submit(int, 0).result(timeout=RENDER_TIMEOUT_S)  # make sure the worker exists
    for proc in list(getattr(pool, "_processes", {}).values()):
        proc.kill()


def render_frame(key: str, options: dict[str, Any], state: int, last_action: int | None) -> str:
    """Return a data: URL with the PNG that gymnasium itself draws for ``state``."""
    if importlib.util.find_spec("pygame") is None:
        raise RenderUnavailable("pygame is not installed on this server")
    try:
        return _get_pool().submit(_render_in_worker, key, options, state, last_action).result(timeout=RENDER_TIMEOUT_S)
    except BrokenProcessPool as exc:
        logger.error("render process crashed; restarting it")
        _reset_pool()
        raise RenderCrashed("The renderer restarted; please try again") from exc
    except FuturesTimeout as exc:
        _reset_pool()
        raise RenderCrashed("Rendering timed out; please try again") from exc
