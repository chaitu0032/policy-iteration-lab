"""Real gymnasium frames: env.render() in rgb_array mode for any state (needs pygame)."""

from __future__ import annotations

import base64
import io
import os
import zlib
from typing import Any

import numpy as np

from .environments import TAXI, make_env

# Headless pygame (servers / serverless functions have no display or audio device).
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

MOVE_ACTIONS = (0, 1, 2, 3)


class RenderUnavailable(RuntimeError):
    """pygame is not installed in this deployment."""


def _png_bytes(frame: np.ndarray) -> bytes:
    """Encode an RGB uint8 array as PNG with only the standard library."""
    height, width, _ = frame.shape
    raw = b"".join(b"\x00" + frame[y].tobytes() for y in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return len(data).to_bytes(4, "big") + body + zlib.crc32(body).to_bytes(4, "big")

    header = width.to_bytes(4, "big") + height.to_bytes(4, "big") + bytes([8, 2, 0, 0, 0])
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")


def render_frame(key: str, options: dict[str, Any], state: int, last_action: int | None) -> str:
    """Return a data: URL with the PNG that gymnasium itself draws for ``state``."""
    try:
        import pygame  # noqa: F401
    except ImportError as exc:
        raise RenderUnavailable("pygame is not installed on this server") from exc
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
